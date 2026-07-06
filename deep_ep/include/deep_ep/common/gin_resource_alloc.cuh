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
inline GinResourceConfig compute_gin_resources(int max_scaleout_warps) {

    GinResourceConfig cfg{};

    // return value between 2:kMaxGinContextBudget GIN context
    cfg.gin_context_cnt = std::max(2, std::min(
        math::ceil_div(max_scaleout_warps, kMinGinContextSharingFactor),
        kMaxGinContextBudget));

    // Internal EFA GDA implementation: Each gin context consume:
    //      >> Internal QP dedicated for the putValue.
    //      >> Data QP.
    //      >> nSignalQP.
    cfg.gin_indexed_signals_cnt = (kTotalQPBudget - 2 * cfg.gin_context_cnt) / cfg.gin_context_cnt;

    // Each ScaleOut warp needs its own indexed signal id. If this fails, reduce
    // `max_scaleout_warps` (e.g. lower `--num-sms`) so the total signal budget
    // covers every warp.
    EP_HOST_ASSERT(cfg.gin_context_cnt * cfg.gin_indexed_signals_cnt >= max_scaleout_warps and
                   "GIN indexed-signal budget cannot give each ScaleOut warp a dedicated "
                   "signal id (max supported at current QP budget: 238 warps)");

    return cfg;
}

}  // namespace deep_ep::elastic::gin_alloc
