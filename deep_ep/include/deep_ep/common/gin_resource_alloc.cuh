#pragma once

#include <algorithm>

#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>

namespace deep_ep::elastic::gin_alloc {

// NCCL GIN EFA GDA: Resource limitation.
static constexpr int kTotalQPBudget              = 256;
static constexpr int kMaxGinContextBudget        = 17;

// Minimum GIN context sharing factor.
static constexpr int kMinGinContextSharingFactor = 10;

// Design constant: ScaleOut warps per SM used by the auto-tuner's budget math.
// Matches the buffer's `!prefer_overlap_with_compute` cap on `num_channels_per_sm`
// (one ScaleOut warp per channel). Under-provisions vs. the runtime peak (which
// can go up to `kNumMaxChannelsPerSM = 8`); worst case just means heavier
// warps-per-context sharing, not a correctness issue.
static constexpr int kMaxWarpsPerSM              = 4;

// Aggressive-setup upper bound on ScaleOut warps used by any single kernel:
// 55 SMs × 4 warps = 220.
//
// Each ScaleOut warp (== channel) needs its own dedicated signal: e.g. when
// warp0 (channel0) signals its peer channel0 warp on a remote node, it uses a
// signal reserved for that channel. That same signal is reused for the channel
// across all nodes in the rail group, so it is shared node-wide — the signal
// count scales with the number of channels (warps), not with the number of
// nodes.
static constexpr int kMaxSM           = (kTotalQPBudget - 2*kMaxGinContextBudget)/kMaxWarpsPerSM;
static constexpr int kMaxScaleoutWarps           = kMaxSM * kMaxWarpsPerSM;

struct GinResourceConfig {
    int gin_indexed_signals_cnt;      // NGinIndexedSignalsPerGinContext (per NCCL context)
    int gin_context_cnt;              // ScaleOut contexts + one Notify context
};

// Compute the full context budget and per-context signal layout from the fixed
// design max warp count. The per-context internal layout is deterministic —
// same warp count always yields the same layout.
//
// Hardware ceiling: total indexed signals (gin_context_cnt × gin_indexed_signals_cnt)
// must be >= max_scaleout_warps so every warp can own a dedicated signal id.
// With the current budgets this caps max_scaleout_warps at 238
// (17 contexts × 14 signals/ctx = 238).
__forceinline__ __device__ __host__ constexpr GinResourceConfig compute_gin_resources(int max_scaleout_warps) {

    GinResourceConfig cfg{};

    // return value between 2:kMaxGinContextBudget GIN context
    const int by_sharing = math::constexpr_ceil_div(max_scaleout_warps, kMinGinContextSharingFactor);
    const int capped = by_sharing < kMaxGinContextBudget ? by_sharing : kMaxGinContextBudget;
    cfg.gin_context_cnt = capped > 2 ? capped : 2;

    // Internal EFA GDA implementation: Each gin context consume:
    //      >> Internal QP dedicated for the putValue.
    //      >> Data QP.
    //      >> nSignalQP.
    cfg.gin_indexed_signals_cnt = (kTotalQPBudget - 2 * cfg.gin_context_cnt) / cfg.gin_context_cnt;

    // Each ScaleOut warp needs its own indexed signal id. If this fails, reduce
    // `max_scaleout_warps` (e.g. lower `--num-sms`) so the total signal budget
    // covers every warp.
#ifndef __CUDA_ARCH__
    EP_HOST_ASSERT(cfg.gin_context_cnt * cfg.gin_indexed_signals_cnt >= max_scaleout_warps and
                   "GIN indexed-signal budget cannot give each ScaleOut warp a dedicated "
                   "signal id (max supported at current QP budget: 238 warps)");
#endif

    return cfg;
}

// Preferred (and workspace-sizing) maximum for per-part signalling.
static constexpr int kMaxParts = 4;

static constexpr int kScaleoutSlotRoundingReserve = kMaxParts;

struct GinPartAllocation {
    int num_parts;           // per-channel part count (>= 1)
    int num_channels_per_sm;
};

// Per-part signal allocation: pick the largest num_parts (up to kMaxParts) that fits
//   ceil(num_channels / num_available_qps) * num_parts <= gin_indexed_signals_cnt
// at one channel per SM, then cap channels_per_sm until the budget holds.
__forceinline__ __device__ __host__ constexpr GinPartAllocation compute_part_allocation(
        const GinResourceConfig& cfg, int num_sms, int num_available_qps, int num_channels_per_sm) {
    const int gin_signals = cfg.gin_indexed_signals_cnt;
    const int ch_per_ctx_min = math::constexpr_ceil_div(num_sms, num_available_qps);
    const int budget_parts = gin_signals / (ch_per_ctx_min > 1 ? ch_per_ctx_min : 1);
    GinPartAllocation alloc{};
    alloc.num_parts = budget_parts < kMaxParts ? budget_parts : kMaxParts;
    alloc.num_parts = alloc.num_parts > 1 ? alloc.num_parts : 1;
    alloc.num_channels_per_sm = num_channels_per_sm;
    while (alloc.num_channels_per_sm > 1 and
           static_cast<long long>(math::constexpr_ceil_div(num_sms * alloc.num_channels_per_sm,
                                                           num_available_qps)) * alloc.num_parts > gin_signals)
        --alloc.num_channels_per_sm;
#ifndef __CUDA_ARCH__
    EP_HOST_ASSERT(static_cast<long long>(math::constexpr_ceil_div(num_sms * alloc.num_channels_per_sm,
                                                                   num_available_qps)) * alloc.num_parts <= gin_signals and
                   "GIN signal budget cannot host even 1 part-signal per channel "
                   "at 1 channel/SM. Reduce --num-sms.");
#endif
    return alloc;
}

// Kernel-side entry points: derive the per-channel part count (and verify the launched
// channel count) as compile-time constants from the provisioned indexed-signal budget.
__forceinline__ __device__ __host__ constexpr int constexpr_num_parts(
        int gin_signals, int num_sms, int num_qps, bool with_notify, int channels_per_sm) {
    GinResourceConfig cfg{};
    cfg.gin_indexed_signals_cnt = gin_signals;
    const int avail = (num_qps - (with_notify ? 1 : 0)) > 0 ? (num_qps - (with_notify ? 1 : 0)) : 1;
    return compute_part_allocation(cfg, num_sms > 0 ? num_sms : 1, avail, channels_per_sm).num_parts;
}

__forceinline__ __device__ __host__ constexpr int constexpr_channels_per_sm(
        int gin_signals, int num_sms, int num_qps, bool with_notify, int channels_per_sm) {
    GinResourceConfig cfg{};
    cfg.gin_indexed_signals_cnt = gin_signals;
    const int avail = (num_qps - (with_notify ? 1 : 0)) > 0 ? (num_qps - (with_notify ? 1 : 0)) : 1;
    return compute_part_allocation(cfg, num_sms > 0 ? num_sms : 1, avail, channels_per_sm).num_channels_per_sm;
}

}  // namespace deep_ep::elastic::gin_alloc
