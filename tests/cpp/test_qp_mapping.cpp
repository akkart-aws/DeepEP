// MIT License
//
// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// Host-side unit test for the channel <-> (QP, signal-id) mapping that
// `comm/qp_mapping.cuh` ships to the kernels. The header is deliberately free of
// CUDA and NCCL types so the exact shipped logic is what runs here.
//
// The property that matters most is QP-0 reservation. `kWithNotifyWarps` decides
// whether QP 0 is set aside for the notify warp and the rail barrier. Cached
// dispatch (the backward pass) launches no notify warps, and upstream therefore
// let data channels fall onto QP 0 -- the same GIN context the rail barrier
// signals on. The two sets of signal ids then collided, a barrier arrival was
// counted as a data arrival, and the forwarder consumed a slot whose payload had
// not landed. That is the Xid 31 FAULT_PDE in NCCLOFITICKET-2895. `QP0Reserved`
// below is the regression test for it.
//
// Build and run (no GPU, no CUDA toolkit needed):
//   g++ -std=c++20 -I deep_ep/include tests/cpp/test_qp_mapping.cpp -o /tmp/test_qp_mapping
//   /tmp/test_qp_mapping

#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <deep_ep/comm/qp_mapping.cuh>

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (not ok) {
        std::printf("  FAIL  %s\n", what.c_str());
        ++g_failures;
    }
}

// ---------------------------------------------------------------- partition

void test_balanced_partition() {
    std::printf("balanced_partition\n");
    for (int n = 1; n <= 64; ++n) {
        for (int q = 1; q <= 32; ++q) {
            std::map<int, int> bin_counts;
            int previous_bin = -1;
            for (int idx = 0; idx < n; ++idx) {
                const auto slot = deep_ep::comm::balanced_partition(idx, n, q);

                check(slot.bin >= 0 and slot.bin < q,
                      "bin in range (n=" + std::to_string(n) + ", q=" + std::to_string(q) + ")");
                check(slot.local >= 0 and slot.local < slot.bin_size,
                      "local index within its bin (n=" + std::to_string(n) + ", q=" + std::to_string(q) + ")");
                // Contiguous and monotonic: a producer's consecutive items stay grouped
                check(slot.bin >= previous_bin,
                      "bin index non-decreasing (n=" + std::to_string(n) + ", q=" + std::to_string(q) + ")");
                previous_bin = slot.bin;
                ++bin_counts[slot.bin];
            }

            // Balanced: any two bins differ by at most one item
            int smallest = n + 1, largest = 0;
            for (const auto& [bin, count] : bin_counts) {
                smallest = count < smallest ? count : smallest;
                largest = count > largest ? count : largest;
            }
            check(largest - smallest <= 1,
                  "bins balanced within 1 (n=" + std::to_string(n) + ", q=" + std::to_string(q) + ")");

            // No idle bin once there are at least as many items as bins
            if (q <= n)
                check(static_cast<int>(bin_counts.size()) == q,
                      "no idle bin (n=" + std::to_string(n) + ", q=" + std::to_string(q) + ")");

            // Backward compatibility: on even division this must be byte-identical to
            // the old ceil-block mapping, which every shipped power-of-2 config used.
            if (n % q == 0) {
                const int base = n / q;
                for (int idx = 0; idx < n; ++idx)
                    check(deep_ep::comm::balanced_partition(idx, n, q).bin == idx / base,
                          "even division matches idx/base (n=" + std::to_string(n) +
                              ", q=" + std::to_string(q) + ")");
            }
        }
    }
}

// ------------------------------------------------------- QP-0 reservation

// One config's full channel -> (qp, signal_id) map, for both notify-warp settings.
template <int kNumSMs, int kNumQPs, int kNumChannelsPerSM>
void test_config(const std::string& label) {
    std::printf("%s\n", label.c_str());

    const auto walk = [&](bool with_notify) {
        std::set<std::pair<int, int>> seen;
        std::set<int> qps_used;
        int max_signal_id = -1;
        bool any_on_qp0 = false;

        for (int sm = 0; sm < kNumSMs; ++sm) {
            for (int ch = 0; ch < kNumChannelsPerSM; ++ch) {
                const int qp = with_notify
                    ? deep_ep::comm::channel_to_qp<kNumSMs, kNumQPs, kNumChannelsPerSM, true>(sm, ch)
                    : deep_ep::comm::channel_to_qp<kNumSMs, kNumQPs, kNumChannelsPerSM, false>(sm, ch);
                const int sig = with_notify
                    ? deep_ep::comm::channel_to_signal_id<kNumSMs, kNumQPs, kNumChannelsPerSM, true>(sm, ch)
                    : deep_ep::comm::channel_to_signal_id<kNumSMs, kNumQPs, kNumChannelsPerSM, false>(sm, ch);

                check(qp >= 0 and qp < kNumQPs, label + ": qp in range");
                check(sig >= 0, label + ": signal id non-negative");
                // (qp, signal_id) must be unique, or two channels share a counter
                check(seen.insert({qp, sig}).second, label + ": (qp, signal_id) unique");

                qps_used.insert(qp);
                max_signal_id = sig > max_signal_id ? sig : max_signal_id;
                any_on_qp0 = any_on_qp0 or qp == 0;
            }
        }

        // The provisioned per-context signal budget is sized from
        // `gin_alloc::channels_per_context()`, so the bound here has to BE that function.
        // It is two-regime: when the QPs outnumber the SMs, each SM owns its own block of
        // QPs and balances only its OWN channels across them, so spare QPs belonging to
        // other SMs cannot absorb this SM's channels. A single-regime
        // `ceil(total_channels / qps)` is WRONG there -- it under-counts, which would let
        // this check pass on a layout the real provisioning rejects.
        //
        // Duplicated rather than included: `gin_resource_alloc.cuh` is not host-compilable
        // (every function carries an unconditional `__device__ __host__`, and it pulls in
        // `common/math.cuh`). KEEP IN SYNC with `channels_per_context()` in that file.
        const auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };
        const int available_qps = kNumQPs - static_cast<int>(with_notify);
        const int avail = available_qps > 1 ? available_qps : 1;
        const int sms = kNumSMs > 1 ? kNumSMs : 1;
        int bound;
        if (kNumQPs == 1) {
            bound = kNumSMs * kNumChannelsPerSM;
        } else if (sms <= avail) {
            const int qps_in_sm = avail / sms;
            bound = ceil_div(kNumChannelsPerSM, qps_in_sm > 1 ? qps_in_sm : 1);
        } else {
            bound = ceil_div(sms * kNumChannelsPerSM, avail);
        }
        check(max_signal_id < bound, label + ": signal id within the per-context budget");

        return any_on_qp0;
    };

    // The regression property: with notify warps, QP 0 belongs to the notify warp
    // and the rail barrier alone, so no data channel may land there.
    const bool qp0_used_with_notify = walk(/*with_notify=*/true);
    if constexpr (kNumQPs > 1)
        check(not qp0_used_with_notify, label + ": QP 0 reserved when notify warps are present");

    // Without notify warps (cached dispatch) data channels do reach QP 0. This is
    // the collision, asserted here so the behaviour is documented rather than
    // discovered in production: the fix is to keep with_notify true in cached mode.
    const bool qp0_used_without_notify = walk(/*with_notify=*/false);
    if constexpr (kNumQPs > 1)
        check(qp0_used_without_notify,
              label + ": QP 0 IS used without notify warps (the ctx-0 collision)");
}

}  // namespace

int main() {
    test_balanced_partition();

    // The customer's production shape: 28 SMs x 7 channels, 11 GIN contexts
    test_config<28, 11, 7>("config: 28 SMs, 11 QPs, 7 channels/SM (NCCLOFITICKET-2895)");
    // The THREAD-mode threshold from NCCLOFI-2080: 12 SMs at 11 vs 13 contexts
    test_config<12, 11, 1>("config: 12 SMs, 11 QPs, 1 channel/SM (NCCLOFI-2080, before)");
    test_config<12, 13, 1>("config: 12 SMs, 13 QPs, 1 channel/SM (NCCLOFI-2080, after)");
    // More QPs than SMs: exercises the per-SM QP-block regime
    test_config<4, 17, 2>("config: 4 SMs, 17 QPs, 2 channels/SM (QPs > SMs regime)");
    // Pins the two-regime bound. With 7 SMs and 13 QPs the single-regime
    // `ceil(channels / qps)` gives ceil(21/13) = 2, but each SM owns only
    // `13 / 7 = 1` QP and must fit all 3 of its own channels on it, so the real
    // bound is 3. The old formula rejected a layout the provisioning allows.
    test_config<7, 13, 3>("config: 7 SMs, 13 QPs, 3 channels/SM (two-regime bound)");
    // Degenerate single-QP fast path
    test_config<8, 1, 4>("config: 8 SMs, 1 QP, 4 channels/SM (single-QP fast path)");

    if (g_failures == 0) {
        std::printf("\nALL PASSED\n");
        return 0;
    }
    std::printf("\n%d CHECK(S) FAILED\n", g_failures);
    return 1;
}
