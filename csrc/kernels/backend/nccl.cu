#include <cstring>
#include <vector>
#include <string>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <sstream>

#include <nccl.h>
#include <nccl_device/core.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>

#include "api.cuh"
#include "../../utils/system.hpp"


namespace deep_ep::nccl {

pybind11::bytearray get_local_unique_id() {
    ncclUniqueId unique_id;
    NCCL_CHECK(ncclGetUniqueId(&unique_id));
    std::vector<char> result(sizeof(ncclUniqueId));
    std::memcpy(result.data(), &unique_id, sizeof(ncclUniqueId));
    return {result.data(), result.size()};
}

int64_t create_nccl_comm(const pybind11::bytearray& root_unique_id_bytes,
                         const int& num_ranks, const int& rank_idx) {
    // Copy unique ID
    ncclUniqueId root_unique_id;
    const auto root_unique_id_str = root_unique_id_bytes.cast<std::string>();
    std::memcpy(&root_unique_id, root_unique_id_str.c_str(), sizeof(ncclUniqueId));

    // Init
    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, num_ranks, root_unique_id, rank_idx));
    if (get_env<int>("EP_BUFFER_DEBUG"))
        printf("New NCCL host communicator created (%d/%d)\n", rank_idx, num_ranks);
    return reinterpret_cast<int64_t>(comm);
}

void destroy_nccl_comm(const int64_t& nccl_comm) {
    NCCL_CHECK(ncclCommAbort(reinterpret_cast<ncclComm_t>(nccl_comm)));
    if (get_env<int>("EP_BUFFER_DEBUG"))
        printf("NCCL host communicator aborted\n");
}

std::tuple<int, int> get_physical_domain_size(const int64_t& nccl_comm) {
    const auto comm = reinterpret_cast<ncclComm_t>(nccl_comm);
    const int num_ranks = ncclTeamWorld(comm).nRanks, num_nvl_ranks = ncclTeamLsa(comm).nRanks;
    EP_HOST_ASSERT(num_ranks % num_nvl_ranks == 0);
    return {num_ranks / num_nvl_ranks, num_nvl_ranks};
}

std::tuple<int, int> get_logical_domain_size(const int64_t& nccl_comm, const bool& allow_hybrid_mode) {
    const auto [num_rdma_ranks, num_nvl_ranks] = get_physical_domain_size(nccl_comm);
    return {allow_hybrid_mode ? num_rdma_ranks : 1,
            allow_hybrid_mode ? num_nvl_ranks : num_rdma_ranks * num_nvl_ranks};
}

NCCLSymmetricMemoryContext::NCCLSymmetricMemoryContext(const int64_t& nccl_comm, const symmetric::cpu_comm_t& cpu_comm,
                                                       const int& num_ranks, const int& rank_idx,
                                                       const int64_t& num_bytes, const int64_t& num_cpu_bytes,
                                                       const bool& allow_hybrid_mode,
                                                       const int& sl_idx, const int& num_allocated_qps,
                                                       const int& num_max_sms,
                                                       const int& dispatch_token_bytes, const int& combine_token_bytes,
                                                       const int& notify_smem_bytes, const int& smem_budget_bytes,
                                                       const bool& prefer_overlap_with_compute):
    rank_idx(rank_idx), num_ranks(num_ranks), num_allocated_qps(num_allocated_qps), num_max_sms(num_max_sms) {
    if (get_env("EP_BUFFER_DEBUG", 0)) {
        int nccl_version;
        NCCL_CHECK(ncclGetVersion(&nccl_version));
        printf("DeepEP initialized with NCCL version: %d.%d.%d (loaded library)\n",
               nccl_version / 10000, (nccl_version % 10000) / 100, nccl_version % 100);
    }

    // Reuse the NCCL communicator
    comm = reinterpret_cast<ncclComm_t>(nccl_comm);

    // Print number of allocated QPs
    if (get_env<int>("EP_BUFFER_DEBUG"))
        printf("EP NCCL device communicator has %d allocated QPs\n", num_allocated_qps);

    const bool gin_disabled = get_env("EP_DISABLE_GIN", 0) != 0;
    if (not gin_disabled) {
        // Query NCCL supported Gin Type
        ncclCommProperties props = NCCL_COMM_PROPERTIES_INITIALIZER;
        NCCL_CHECK(ncclCommQueryProperties(comm, &props));
        EP_HOST_ASSERT(
            (allow_hybrid_mode ? props.railedGinType : props.ginType) != NCCL_GIN_TYPE_NONE and
            "NCCL GIN is unavailable. This is usually due to a network configuration issue, "
            "such as `allow_hybrid_mode=0` (disable direct RDMA kernels) in multi-plane network.");
    }

    // Initialize NCCL device communicator
    ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    if (num_ranks > 1 and not gin_disabled) {
        const int num_nvl_ranks_early = ncclTeamLsa(comm).nRanks;
        EP_HOST_ASSERT(num_ranks % num_nvl_ranks_early == 0);
        const int num_rdma_ranks = num_ranks / num_nvl_ranks_early;
        // Cap down the number of SMs, as we are limited in resources. The ctor
        // parameter `num_max_sms` is a const reference, but the member
        // `this->num_max_sms` (initialized from it) is mutable, so we override
        // the member — same pattern as `num_allocated_qps` below.
        if (this->num_max_sms > elastic::gin_alloc::kMaxSM) {
            printf("DeepEP WARNING: requested num_max_sms=%d exceeds the GIN resource "
                   "limit (kMaxSM=%d); overriding to %d\n",
                   this->num_max_sms, elastic::gin_alloc::kMaxSM, elastic::gin_alloc::kMaxSM);
            this->num_max_sms = elastic::gin_alloc::kMaxSM;
        }

        constexpr int kNumNotifyWarps = 4;
        constexpr int kNumMaxChannelsPerSM = 8;
        if (dispatch_token_bytes <= 0) {
            const int scaleout_warp_cap = (this->num_max_sms > 0)
                ? (this->num_max_sms * elastic::gin_alloc::kMaxWarpsPerSM)
                : elastic::gin_alloc::kMaxScaleoutWarps;
            gin_config = elastic::gin_alloc::compute_gin_resources(scaleout_warp_cap);
            this->num_channels_per_sm = 0;
        } else {
            auto channels_per_sm_for_smem = [&]() -> int {
                int cpsm = std::min(
                    (smem_budget_bytes - notify_smem_bytes) / std::max(1, dispatch_token_bytes),
                    32 - kNumNotifyWarps);
                cpsm = std::min(cpsm, smem_budget_bytes / std::max(1, combine_token_bytes));
                cpsm = std::min(cpsm / 2, kNumMaxChannelsPerSM);
                if (not prefer_overlap_with_compute)
                    cpsm = std::min(cpsm, elastic::gin_alloc::kMaxWarpsPerSM);
                return std::max(1, cpsm);
            };
            const int smem_cpsm = channels_per_sm_for_smem();

            const int scaleout_warp_cap = (this->num_max_sms > 0)
                ? (this->num_max_sms * elastic::gin_alloc::kMaxWarpsPerSM)
                : elastic::gin_alloc::kMaxScaleoutWarps;
            gin_config = elastic::gin_alloc::compute_gin_resources(scaleout_warp_cap);

            const int num_available_qps_est = std::max(1,
                (this->num_allocated_qps > 0 ? this->num_allocated_qps : gin_config.gin_context_cnt)
                - (kNumNotifyWarps > 0 ? 1 : 0));
            const int qps = std::min(num_available_qps_est, gin_config.gin_context_cnt);
            const auto part_alloc = elastic::gin_alloc::compute_part_allocation(
                gin_config, std::max(1, this->num_max_sms), std::max(1, qps), smem_cpsm);
            this->num_channels_per_sm = part_alloc.num_channels_per_sm;
        }

        if (get_env<int>("EP_BUFFER_DEBUG"))
            printf("DeepEP GIN fixed point: channels_per_sm=%d, gin_context_cnt=%d, "
                   "gin_indexed_signals_cnt=%d\n",
                   this->num_channels_per_sm, gin_config.gin_context_cnt, gin_config.gin_indexed_signals_cnt);

        // Auto-tuner sanity checks, each peer rail team (not including self) must have dedicated signals
        EP_HOST_ASSERT(gin_config.gin_indexed_signals_cnt >= (num_rdma_ranks - 1));

        // If the user did not set --num-qp (=0), default to the auto-tuner's context
        // budget. Otherwise it must fit within it; if it exceeds the budget we
        // override it down to the budget and warn instead of asserting.
        if (this->num_allocated_qps == 0) {
            this->num_allocated_qps = gin_config.gin_context_cnt;
        } else if (this->num_allocated_qps > gin_config.gin_context_cnt) {
            printf("DeepEP WARNING: requested num_allocated_qps=%d exceeds the GIN context "
                   "budget (%d); overriding to %d\n",
                   this->num_allocated_qps, gin_config.gin_context_cnt, gin_config.gin_context_cnt);
            this->num_allocated_qps = gin_config.gin_context_cnt;
        }

        printf("GIN auto-tuner: gin_context_cnt=%d, gin_indexed_signals_cnt=%d, num_qp=%d, num_max_sms=%d\n",
               gin_config.gin_context_cnt,
               gin_config.gin_indexed_signals_cnt,
               this->num_allocated_qps,
               this->num_max_sms);

        reqs.ginExclusiveContexts = false;
        // TODO: currently a WA, depth-1024 SQ rings across 17 contexts exceed the EFA DMA-pinning
        // budget at comm create; these rings should live in card memory, not
        // MR-mapped host memory. Revert to 1024 once the GDA-side fix lands.
        reqs.ginQueueDepth = 256;
        reqs.ginTrafficClass = sl_idx;
        reqs.ginContextCount = gin_config.gin_context_cnt;
        reqs.ginSignalCount = gin_config.gin_indexed_signals_cnt;
        reqs.ginCounterCount = gin_config.gin_indexed_signals_cnt;

        reqs.ginConnectionType = allow_hybrid_mode ? NCCL_GIN_CONNECTION_RAIL: NCCL_GIN_CONNECTION_FULL;
        // VA signals are not supported by EFA GDA, in addition EFA GDA does not support strong signals.
        reqs.ginVaSignalsRequired = false;
        reqs.ginStrongSignalsRequired = false;
    }
    NCCL_CHECK(ncclDevCommCreate(comm, &reqs, &dev_comm));

    // Now we know the NVLink domain size
    num_nvl_ranks = dev_comm.lsaSize, nvl_rank_idx = dev_comm.lsaRank;
    num_rdma_ranks = num_ranks / num_nvl_ranks, rdma_rank_idx = rank_idx / num_nvl_ranks;
    EP_HOST_ASSERT(num_ranks % num_nvl_ranks == 0 and nvl_rank_idx == rank_idx % num_nvl_ranks);
    EP_HOST_ASSERT(rank_idx == rdma_rank_idx * num_nvl_ranks + nvl_rank_idx);

    // Calculate scaleout/up domain size
    if (allow_hybrid_mode) {
        num_scaleout_ranks = num_rdma_ranks, num_scaleup_ranks = num_nvl_ranks;
        scaleout_rank_idx = rdma_rank_idx, scaleup_rank_idx = nvl_rank_idx;
    } else {
        num_scaleout_ranks = 1, num_scaleup_ranks = num_ranks;
        scaleout_rank_idx = 0, scaleup_rank_idx = rank_idx;
    }
    is_scaleup_nvlink = num_scaleup_ranks == num_nvl_ranks;

    // Create symmetric memory
    // num_bytes = GPU + CPU, derive GPU portion
    this->symmetric_memory = symmetric::alloc(
        num_bytes - num_cpu_bytes, num_cpu_bytes,
        allow_hybrid_mode, num_scaleup_ranks, scaleout_rank_idx,
        cpu_comm);

    // Create window
    // NOTES: `ncclCommWindowRegister` is collective: it internally calls bootstrapBarrier
    // across all ranks, so no explicit barrier is needed after this call.
    raw_window_ptr = this->symmetric_memory->ptr;
    this->num_gpu_bytes = this->symmetric_memory->num_gpu_bytes;
    this->num_cpu_bytes = this->symmetric_memory->num_cpu_bytes;
    NCCL_CHECK(ncclCommWindowRegister(comm, raw_window_ptr, this->symmetric_memory->num_bytes, &window, NCCL_WIN_DEFAULT));
    NCCL_CHECK(ncclGetLsaDevicePointer(window, 0, nvl_rank_idx, &mapped_window_ptr));

    // Get LSA pointers for all LSA peers
    // TODO: check whether this is correct for network with RDMA
    nvl_window_ptrs.resize(num_nvl_ranks);
    for (int i = 0; i < num_nvl_ranks; ++ i)
        NCCL_CHECK(ncclGetLsaDevicePointer(window, 0, i, &nvl_window_ptrs[i]));
}

void* NCCLSymmetricMemoryContext::get_sym_ptr(void* ptr, const int& dst_rank_idx) const {
    const auto offset = static_cast<uint8_t*>(ptr) - static_cast<uint8_t*>(mapped_window_ptr);
    return static_cast<uint8_t*>(nvl_window_ptrs[dst_rank_idx]) + offset;
}

void NCCLSymmetricMemoryContext::finalize() {
    // Deregister window
    NCCL_CHECK(ncclCommWindowDeregister(comm, window));
    symmetric_memory.reset();

    // Destroy device communicator
    NCCL_CHECK(ncclDevCommDestroy(comm, &dev_comm));
}

}  // namespace deep_ep::nccl
