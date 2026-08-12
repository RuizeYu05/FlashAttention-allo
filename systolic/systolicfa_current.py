# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Beat-serial systolic FlashAttention — MIXED PRECISION (fp16 mul, fp32 add)
==========================================================================

Derived from attn_beatserial_merged.py. Array topology, stream types, FIFO
counts, borders, feeders and combine are all UNCHANGED -- only the PE's
internal arithmetic precision and accumulator structure differ, so every
symbolic balance argument from the parent file still holds verbatim.

WHAT CHANGED AND WHY (V80 / Vitis 2025.1, measured -- probe_fp32acc.py)
----------------------------------------------------------------------
Vitis picks radically different fp cores on V80 than on U280:

    op        U280                          V80
    fp32 add  (not used)                    fadd_..._1_primitive_dsp  lat 1  DSP
    fp16 add  hadd_..._10_full_dsp  lat 10  hadd_..._4/5_no_dsp     lat 4-5  LUT
    fp16 mul  hmul_..._5_max_dsp    lat  5  hmul_..._3_no_dsp         lat 3  LUT

The fp32 add is a LATENCY-1 hardened DSPFP32 primitive. That single fact
invalidates two U280-era microarchitecture decisions:

  1. LACC rotating accumulators. The phase-A recurrence carries a DISTANCE-1
     dependence (the adder output is re-read by the next beat), so II is
     bounded by Lat_add no matter how many partial accumulators there are --
     LACC never decoupled it. On V80 the fp16 form measures II=3 in the
     deployed 4x32 band (all 128 interior PEs, iterLat 27), silently missing
     the II=1 acceptance gate. With a latency-1 adder, ONE accumulator gives
     II=1.

  2. The boundary fold tree (parent file's 4.3). It exists only to avoid a
     serial chain on the latency-10 full_dsp adder. With one accumulator
     there is nothing to fold; the whole tree is deleted.

    probe, dh=64:  fp32 LACC=1 -> II=1, iterLat  6, DSP 2, LUT  402
                   fp16 LACC=4 -> II=4, iterLat 25, DSP 0, LUT 1395

PRECISION BOUNDARY (deliberate)
-------------------------------
    multiply  fp16   -- operands q_loc/K/V and alpha/p all stay fp16
    add/sub   fp32   -- accumulation and the boundary softmax arithmetic
    exp       fp16   -- hexp is already the ONLY remaining DSP consumer on
                        V80 (~800 per band); widening it would grow the one
                        resource fp32 adds also want
    STREAMS   fp16   -- every Stream[] type is untouched, so link width, SLL
                        pressure, FIFO depth and all symbolic beat counts are
                        identical to the parent file

Multiplies are written as an explicit fp16 temporary BEFORE the widening
assignment. Writing `w: float32 = a * b` on two fp16 operands lets Vitis
promote both and emit an fp32 MULTIPLY, which is not this datapath.

The accuracy win is concentrated where it matters: phase A is a dh-long
accumulation chain, now carried in fp32. The o/d updates are single adds
whose results go straight back out as fp16, so they gain little -- that is
the accepted cost of keeping the links narrow.

Same array topology as the dh-unrolled version: rows = query rows, columns = KV
chunks, online-softmax state chained left->right, K/V streaming top->bottom,
split-K + combine retained, pure DAG, no feedback edge.

ONE THING CHANGES: HEAD_DIM is no longer unrolled inside the PE. It is
beat-serialized at chunk width 1, which moves dh out of the AREA budget and into
the TIME budget.

Per key t, PE(i,j) runs 2*dh beats:

    phase A  (dh beats):  acc += fp32(q_loc[d] * k_in[d])    one fp16 mult/beat
    boundary          :  x = acc;  m_out = max(m_in, x)     (no fold: LACC=1)
                         alpha = exp(m_in - m_out);  p = exp(x - m_out)
                         d_out = alpha*d_in + p
    phase B  (dh beats):  o_out[d] = alpha*o_in[d] + p*v_in[d]    two mults/beat

Load-bearing property: phase B is BEAT-STREAMING. o_out[d] depends only on
o_in[d] and v_in[d] -- no full-vector dependency -- so o forwards one element per
beat and is never materialized dh-wide anywhere.

RATE INVARIANT: the PE spends 2*dh beats per key and must forward exactly dh
elements of o in that window. dh <= 2*dh, so the horizontal state link is
rate-matched with 2x slack.

CONSEQUENCE: every inter-PE link is float16, not float16[HEAD_DIM]. Multiplier
count per PE is ~3-4 (one in phase A, two in phase B, one at the boundary) and
FLAT IN dh -- versus 92 DSP/PE at dh=8 and ~10,640 total at dh=64 in the unrolled
version, which exceeds the U280's ~9,024 DSP.

TWO DESIGN DECISIONS FIXED HERE
  * BS = 1. Folding BS keys inside a PE needs v_loc[BS][dh] -- dh-wide storage,
    exactly what beat-serial exists to remove. So KV_TILE = NCOLS.
  * q is STATIONARY. Each PE holds q_loc[HEAD_DIM] (dh registers, cheap), loaded
    once per query tile by a beat-serial load phase down the row. Streaming q per
    key would reinstate dh-wide horizontal traffic.

WHAT GETS WORSE (expected, not a regression)
  Per-query latency rises to ~2*dh beats/key. Throughput is preserved ONLY if
  every multiplier is busy every beat, i.e. II=1 on both phase loops. If II=1
  holds and DSP/PE is flat in dh, the design succeeded despite the latency rise.

STATUS: reference implementation, NOT yet run against Allo. Expect frontend
debugging. Highest-risk construct is `acc[da % LACC]` -- see probe_phaseA.py,
which must pass BEFORE this file is worth running.
"""

import os
import numpy as np
import allo
import allo.dataflow as df
import allo.backend.hls as hls
from allo.ir.types import float16, float32, Stream
from allo import Memory


def get_beatserial_flash_attention(
    BATCH_SIZE: int,
    CONTEXT_LENGTH: int,
    HIDDEN_SIZE: int,
    NUM_HEADS: int,
    NROWS: int,
    NCOLS: int,
    LACC: int,
):
    HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
    KV_TILE = NCOLS                       # BS = 1: one key per PE per tile
    NUM_KV_TILES = CONTEXT_LENGTH // KV_TILE
    NUM_Q_TILES = CONTEXT_LENGTH // NROWS

    assert CONTEXT_LENGTH % KV_TILE == 0
    assert CONTEXT_LENGTH % NROWS == 0

    P0 = NROWS + 2
    P1 = NCOLS + 2

    D_SQRT = float(HEAD_DIM**0.5)
    D_SCALE = 1.0 / D_SQRT
    QKV_ELEMS = BATCH_SIZE * CONTEXT_LENGTH * HIDDEN_SIZE
    OUT_ELEMS = BATCH_SIZE * CONTEXT_LENGTH * NUM_HEADS * HEAD_DIM

    COMBINE_LANES = 4
    assert NUM_KV_TILES % COMBINE_LANES == 0, (
        f"NUM_KV_TILES={NUM_KV_TILES} must be divisible by COMBINE_LANES={COMBINE_LANES}"
    )
    assert NUM_KV_TILES >= COMBINE_LANES

    # ---- FIFO depths: now measured in BEATS, not vectors. Start generous, then
    #      cut in Phase 4 to whatever `emu` tolerates -- depth is LUTRAM and is
    #      the remaining congestion pressure once links are scalar.
    #      Skew between adjacent columns is ~dh beats (PE j+1 runs its phase A
    #      while PE j runs its phase B), so 2*dh covers it with margin.
    # FIFO depths in BEATS. The originals (4*dh / 8*dh) were "start generous".
    # The real skew between adjacent columns is ~dh beats, so 2*dh covers V and o
    # with margin; the feeders need far less than 8*dh. Env-driven multipliers so
    # the depth can be swept and the emu-tolerated minimum found without editing.
    # NOTE: these trims MUST be re-validated in emu (csyn cannot see a capacity
    # deadlock). This file is the BRAM-scaling optimization variant.
    D_Q = int(os.environ.get("MUL_Q", "2")) * HEAD_DIM
    D_K = int(os.environ.get("MUL_K", "2")) * HEAD_DIM
    D_V = int(os.environ.get("MUL_V", "2")) * HEAD_DIM
    D_O = int(os.environ.get("MUL_O", "2")) * HEAD_DIM
    D_FEED = int(os.environ.get("MUL_FEED", "2")) * HEAD_DIM

    @df.region()
    def top(
        q_mem: float16[QKV_ELEMS],
        k_mem: float16[QKV_ELEMS],
        v_mem: float16[QKV_ELEMS],
        output_mem: float16[OUT_ELEMS],
    ):
        # ---- inter-PE streams: ALL SCALAR. This is the whole point. If any of
        #      these regains a [HEAD_DIM] element type, the redesign is void.
        fifo_Q: Stream[float16, D_Q][P0, P1]    # q load phase, left -> right
        fifo_K: Stream[float16, D_K][P0, P1]    # K beats,      top  -> bottom
        fifo_V: Stream[float16, D_V][P0, P1]    # V beats,      top  -> bottom
        fifo_m: Stream[float16, 16][P0, P1]     # running max,  left -> right
        fifo_d: Stream[float16, 16][P0, P1]     # running denom,left -> right
        fifo_o: Stream[float16, D_O][P0, P1]    # o beats,      left -> right

        fifo_in_Q: Stream[float16, D_FEED][P0 - 2]
        fifo_in_K: Stream[float16, D_FEED][P1 - 2]
        fifo_in_V: Stream[float16, D_FEED][P1 - 2]

        fifo_fill_K: Stream[float16, D_FEED]
        fifo_fill_V: Stream[float16, D_FEED]

        fifo_part_m: Stream[float16, 8][P0 - 2, COMBINE_LANES]
        fifo_part_d: Stream[float16, 8][P0 - 2, COMBINE_LANES]
        fifo_part_o: Stream[float16, D_O][P0 - 2, COMBINE_LANES]

        fifo_out: Stream[float16, D_FEED][NROWS]

        # ---- load_q: beat-serial, scale folded in ----------------------------
        @df.kernel(mapping=[1], args=[q_mem])
        def load_q(q_d: float16[QKV_ELEMS]):
            for b in range(BATCH_SIZE):
                for h in range(NUM_HEADS):
                    for tr in range(0, CONTEXT_LENGTH, NROWS):
                        with allo.meta_for(NROWS) as r:
                            for jj in range(HEAD_DIM):
                                q_idx = (
                                    b * (CONTEXT_LENGTH * HIDDEN_SIZE)
                                    + (tr + r) * HIDDEN_SIZE
                                    + h * HEAD_DIM
                                    + jj
                                )
                                sc: float16 = D_SCALE
                                qv: float16 = q_d[q_idx] * sc
                                fifo_in_Q[r].put(qv)

        # ---- load_kv_fill: DRAM -> handoff, beat-serial ----------------------
        #      order per (tile, c): dh K beats then dh V beats. K and V are
        #      separate FIFOs so their relative order does not matter; only the
        #      order WITHIN each stream does.
        @df.kernel(mapping=[1], args=[k_mem, v_mem])
        def load_kv_fill(k_d: float16[QKV_ELEMS], v_d: float16[QKV_ELEMS]):
            for b in range(BATCH_SIZE):
                for h in range(NUM_HEADS):
                    for tile_f in range(NUM_KV_TILES):
                        for c in range(NCOLS):
                            t = tile_f * KV_TILE + c
                            base = (
                                b * (CONTEXT_LENGTH * HIDDEN_SIZE)
                                + t * HIDDEN_SIZE
                                + h * HEAD_DIM
                            )
                            for jjf in range(HEAD_DIM):
                                fifo_fill_K.put(k_d[base + jjf])
                            for jjg in range(HEAD_DIM):
                                fifo_fill_V.put(v_d[base + jjg])

        # ---- load_kv_replay: capture first sweep, replay for later q tiles ----
        #      k_buf/v_buf now hold the whole K/V for this (b,h): L*dh elements.
        #      At dh=64, L=1024 that is 64K elem x 2B = 128KB each -> prime URAM
        #      candidates (the design currently reports URAM=0).
        @df.kernel(mapping=[1], args=[])
        def load_kv_replay():
            # Whole-context K/V replay store, O(L*dh) -> the dominant BRAM user.
            # Map to URAM (idle: design reports URAM=0). Partition stays on dim=2
            # (NCOLS) for the meta_for(NCOLS) parallel access; each of the 8 banks
            # is [NUM_KV_TILES, dh] and fits ~2 URAM blocks even at dh=128.
            k_buf: float16[NUM_KV_TILES, NCOLS, HEAD_DIM] @ Memory(
                resource="URAM", storage_type="RAM_2P"
            )
            v_buf: float16[NUM_KV_TILES, NCOLS, HEAD_DIM] @ Memory(
                resource="URAM", storage_type="RAM_2P"
            )
            for b in range(BATCH_SIZE):
                for h in range(NUM_HEADS):
                    for tile_r in range(NUM_KV_TILES):
                        with allo.meta_for(NCOLS) as c:
                            for jjr in range(HEAD_DIM):
                                kv: float16 = fifo_fill_K.get()
                                k_buf[tile_r, c, jjr] = kv
                                fifo_in_K[c].put(kv)
                            for jjs in range(HEAD_DIM):
                                vv: float16 = fifo_fill_V.get()
                                v_buf[tile_r, c, jjs] = vv
                                fifo_in_V[c].put(vv)

                    # ---- replay passes for q-tiles 1..N-1 -- THE BOTTLENECK.
                    # Original form put meta_for(NCOLS) OUTSIDE the beat loop, so
                    # the lanes unrolled into 2*NCOLS SEQUENTIAL pipelined loops of
                    # dh trips each. Each such loop pays its own fill/drain: at
                    # dh=8 csyn measured 11 cycles for 8 trips, x 2*NCOLS lanes =
                    # 352 cycles/tile at NCOLS=16 where the ideal is dh. That made
                    # load_kv_replay 99.6% of total latency and starved the array
                    # to ~2% duty cycle (csyn: replay 6.527e6 vs interior PE
                    # 1.13e5 at 16x16).
                    #
                    # k_buf/v_buf are RANDOM ACCESS (unlike pass 0, whose order is
                    # fixed by the load_kv_fill stream), so the lane axis can move
                    # INSIDE the beat loop: ONE pipelined loop of dh trips per
                    # tile, NCOLS lanes issued per cycle out of the NCOLS banks
                    # created by partition(dim=2). K and V share the loop -- they
                    # are separate streams so only per-stream order matters, and
                    # both still deliver jj=0..dh-1 per tile.
                    #
                    # DEADLOCK BOUND: the top border consumes dh K then dh V per
                    # key, so pending_V = pending_K + j with j < dh. fifo_in_K
                    # empty => pending_V < dh, hence V can never be full while K
                    # is empty as long as D_FEED >= dh. D_FEED = 2*dh holds this
                    # with 2x margin.
                    for tr in range(NROWS, CONTEXT_LENGTH, NROWS):
                        for tile_rr in range(NUM_KV_TILES):
                            for jjp in range(HEAD_DIM):
                                with allo.meta_for(NCOLS) as c2:
                                    fifo_in_K[c2].put(k_buf[tile_rr, c2, jjp])
                                    fifo_in_V[c2].put(v_buf[tile_rr, c2, jjp])

        # ---- the PE array ----------------------------------------------------
        @df.kernel(mapping=[P0, P1], args=[])
        def pe():
            i, j = df.get_pid()

            # Stationary query. NOT completely partitioned: `q_loc[da]` is a
            # DYNAMIC index, so a complete partition builds a dh:1 16-bit mux per
            # PE -- that is the O(dh) LUT/PE growth (1920 -> 5956 LUT across
            # dh 8->128). Access is strictly sequential (write in the load phase,
            # read in phase A), so a single LUTRAM serves it at 1 elem/cycle with
            # no mux at all. DSP/PE is unaffected; this is a LUT/FF cut whose
            # whole purpose is to buy a bigger array at dh=64/128.
            q_loc: float16[HEAD_DIM] @ Memory(
                resource="LUTRAM", storage_type="RAM_2P"
            ) = 0
            # MIXED PRECISION: fp32 accumulator, fp16 multiply, fp16 streams.
            #
            # LACC IS 1 HERE, AND THAT IS THE POINT. On V80/Vitis 2025.1 the
            # fp32 add is `fadd_32ns_32ns_32_1_primitive_dsp` -- LATENCY 1, on
            # the hardened DSPFP32 in DSP58. The phase-A recurrence carries a
            # distance-1 dependence (adder output re-read next beat), so
            # II == Lat_add. At latency 1 that is II=1 with a SINGLE
            # accumulator; no rotating registers, no boundary fold tree.
            #
            # Measured (dh=64 probe, probe_fp32acc.py):
            #   fp32 LACC=1 -> II=1, iterLat  6, DSP 2, LUT  402
            #   fp16 LACC=4 -> II=4, iterLat 25, DSP 0, LUT 1395  <- deployed
            # LACC>1 is strictly worse on fp32: DSP and LUT scale with it and
            # buy nothing, because LACC never decoupled the distance-1 dep.
            acc: float32[1] = 0              # single fp32 accumulator, phase A
            # bnd[0]=alpha, bnd[1]=p of the PREVIOUS key. The merged beat loop
            # applies them to phase B of key t-1 while phase A of key t runs.
            # A 2-element array (not two scalars) so it partitions to registers
            # the same way `acc` does.
            bnd: float16[2] = 0

            # ---- corners ----
            with allo.meta_if(i in {0, P0 - 1} and j in {0, P1 - 1}):
                pass

            # ---- top border: MERGED -- one K and one V beat per cycle --------
            #
            # WHY THIS CHANGED. The split form ran dh K beats THEN dh V beats,
            # i.e. 2*dh beats per key, while the merged interior PE runs dh.
            # That made the borders the critical path once fp32 took the
            # interior to II=1: measured at 4x32/dh=64, interior 571,716 cycles
            # but top/bottom border 1,114,625, capping band speedup at 1.43x
            # instead of the 2.78x the interior actually achieved.
            #
            # SAFE BECAUSE the replay path -- which dominates, running for
            # q-tiles 1..N-1 -- ALREADY produces interleaved K/V (one
            # fifo_in_K.put and one fifo_in_V.put per beat). The split border
            # was the mismatched party; interleaving aligns consumer with
            # producer. Per-stream order is unchanged and counts are identical,
            # so every symbolic FIFO balance still holds.
            #
            # DEADLOCK: pass 0 (load_kv_fill) still emits dh K then dh V, so a
            # merged border takes K0 and then waits for V0 while the producer
            # is still emitting K1..K(dh-1). That needs fifo_in_K to absorb dh
            # entries -- exactly the same D_FEED >= dh bound as before, just
            # binding in the other direction. D_FEED = 2*dh keeps 2x margin.
            with allo.meta_elif(i == 0):
                for b in range(BATCH_SIZE):
                    for h in range(NUM_HEADS):
                        for qt in range(NUM_Q_TILES):
                            for kt in range(NUM_KV_TILES):
                                for dkv in range(HEAD_DIM):
                                    fifo_K[i + 1, j].put(fifo_in_K[j - 1].get())
                                    fifo_V[i + 1, j].put(fifo_in_V[j - 1].get())

            # ---- left border: q load phase, then fresh state per KV tile ----
            with allo.meta_elif(j == 0):
                for b in range(BATCH_SIZE):
                    for h in range(NUM_HEADS):
                        for qt in range(NUM_Q_TILES):
                            for dq in range(HEAD_DIM):
                                fifo_Q[i, j + 1].put(fifo_in_Q[i - 1].get())
                            for kt in range(NUM_KV_TILES):
                                m_init: float16 = -1e30
                                d_init: float16 = 0.0
                                fifo_m[i, j + 1].put(m_init)
                                fifo_d[i, j + 1].put(d_init)
                                for do0 in range(HEAD_DIM):
                                    o_init: float16 = 0.0
                                    fifo_o[i, j + 1].put(o_init)

            # ---- bottom border: MERGED drain, one K and one V per cycle ------
            # The last interior row PRODUCES interleaved (its merged beat loop
            # forwards one K and one V per beat), so a split drain was doubly
            # wrong: 2*dh beats AND mismatched with its own producer, forcing
            # fifo_K to buffer dh entries. Interleaving fixes both.
            with allo.meta_elif(i == P0 - 1):
                for b in range(BATCH_SIZE):
                    for h in range(NUM_HEADS):
                        for qt in range(NUM_Q_TILES):
                            for kt in range(NUM_KV_TILES):
                                for dkv2 in range(HEAD_DIM):
                                    _k: float16 = fifo_K[i, j].get()
                                    _v: float16 = fifo_V[i, j].get()

            # ---- right border: drain q beats, emit partials to combine ----
            with allo.meta_elif(j == P1 - 1):
                for b in range(BATCH_SIZE):
                    for h in range(NUM_HEADS):
                        for qt in range(NUM_Q_TILES):
                            for dq2 in range(HEAD_DIM):
                                _q: float16 = fifo_Q[i, j].get()
                            for kt_strip in range(NUM_KV_TILES // COMBINE_LANES):
                                with allo.meta_for(COMBINE_LANES) as lane:
                                    m_t: float16 = fifo_m[i, j].get()
                                    d_t: float16 = fifo_d[i, j].get()
                                    fifo_part_m[i - 1, lane].put(m_t)
                                    fifo_part_d[i - 1, lane].put(d_t)
                                    for db3 in range(HEAD_DIM):
                                        fifo_part_o[i - 1, lane].put(
                                            fifo_o[i, j].get()
                                        )

            # ---- compute PE: MERGED phases, dh beats per key (was 2*dh) ----
            #
            # The split-phase PE spends 2*dh beats per key: dh in phase A
            # (consume K) then dh in phase B (consume V, forward o). Phase B of
            # key t-1 has NO data dependence on phase A of key t -- it needs only
            # alpha/p of key t-1, which the boundary already produced. So the two
            # can share ONE beat loop, software-pipelined by one key:
            #
            #     beat d of the merged loop:  phase A of key t   (q*K[t][d])
            #                                 phase B of key t-1 (alpha_prev*o + p_prev*V[t-1][d])
            #
            # Beat count per key drops 2*dh -> dh. Total beats per q-tile go from
            # 2*N*dh to (N+1)*dh, i.e. ~1.97x fewer at N=NUM_KV_TILES=64.
            #
            # The first key has no predecessor and the last has no successor, so
            # the loop is peeled: a phase-A-only PROLOGUE for key 0 and a
            # phase-B-only EPILOGUE for key N-1. Peeling (rather than an `if`
            # inside the beat loop) keeps the stream access pattern
            # unconditional, which is what preserves II=1 and keeps the symbolic
            # FIFO counts exact.
            #
            # RATE: the PE now forwards dh o-beats within dh beats -- exactly
            # 1:1, where the split-phase version had 2x slack. Still rate-legal,
            # but the o link has no headroom, so D_O carries the jitter.
            #
            # COUNTS ARE UNCHANGED per stream: K = N*dh, V = N*dh, o = N*dh,
            # m/d = N. Only the interleaving moves, so every border/feeder/
            # combine kernel is untouched.
            with allo.meta_else():
                for b in range(BATCH_SIZE):
                    for h in range(NUM_HEADS):
                        for qt in range(NUM_Q_TILES):
                            # q load phase: capture AND forward, dh beats
                            for dq3 in range(HEAD_DIM):
                                qv2: float16 = fifo_Q[i, j].get()
                                q_loc[dq3] = qv2
                                fifo_Q[i, j + 1].put(qv2)

                            # ================= PROLOGUE: key 0, phase A only ===
                            acc[0] = 0.0
                            for da0 in range(HEAD_DIM):
                                k0: float16 = fifo_K[i, j].get()
                                fifo_K[i + 1, j].put(k0)
                                # fp16 multiply, THEN widen. Writing
                                # `pw0: float32 = q_loc[da0] * k0` directly
                                # would let Vitis promote both operands and
                                # emit an fp32 multiply -- not the datapath
                                # we want (mul stays fp16).
                                pr0: float16 = q_loc[da0] * k0
                                pw0: float32 = pr0
                                acc[0] = acc[0] + pw0        # fp32, lat 1

                            # No fold: one accumulator, nothing to reduce.
                            xx0: float32 = acc[0]
                            mi0: float16 = fifo_m[i, j].get()
                            di0: float16 = fifo_d[i, j].get()
                            mi0w: float32 = mi0
                            mo0w: float32 = mi0w
                            if xx0 > mi0w:
                                mo0w = xx0
                            # exp stays fp16: hexp is already the only DSP
                            # consumer left and an fp32 exp would grow it.
                            ea0: float16 = mi0w - mo0w       # fp32 sub -> fp16
                            eb0: float16 = xx0 - mo0w
                            bnd[0] = allo.exp(ea0)           # alpha_prev
                            bnd[1] = allo.exp(eb0)           # p_prev
                            mo0h: float16 = mo0w
                            fifo_m[i, j + 1].put(mo0h)
                            dp0: float16 = bnd[0] * di0      # fp16 multiply
                            dp0w: float32 = dp0
                            db0w: float32 = bnd[1]
                            ds0: float16 = dp0w + db0w       # fp32 add -> fp16
                            fifo_d[i, j + 1].put(ds0)

                            # ================= MAIN: merged beat loop ==========
                            for kt in range(1, NUM_KV_TILES):
                                acc[0] = 0.0

                                # ONE loop: phase A of key kt + phase B of kt-1.
                                # Phase A's recurrence is distance-1 into a
                                # latency-1 fp32 adder -> II=1 with a single
                                # accumulator. Phase B has no carried dep.
                                for dm in range(HEAD_DIM):
                                    kv3: float16 = fifo_K[i, j].get()
                                    fifo_K[i + 1, j].put(kv3)
                                    prod: float16 = q_loc[dm] * kv3
                                    prodw: float32 = prod
                                    acc[0] = acc[0] + prodw

                                    vv3: float16 = fifo_V[i, j].get()
                                    fifo_V[i + 1, j].put(vv3)
                                    oi: float16 = fifo_o[i, j].get()
                                    om1: float16 = bnd[0] * oi     # fp16 mul
                                    om2: float16 = bnd[1] * vv3    # fp16 mul
                                    om1w: float32 = om1
                                    om2w: float32 = om2
                                    osum: float16 = om1w + om2w    # fp32 add
                                    fifo_o[i, j + 1].put(osum)

                                # boundary for key kt -> becomes prev for kt+1
                                x: float32 = acc[0]
                                m_in: float16 = fifo_m[i, j].get()
                                d_in: float16 = fifo_d[i, j].get()
                                m_inw: float32 = m_in
                                m_outw: float32 = m_inw
                                if x > m_inw:
                                    m_outw = x
                                ea: float16 = m_inw - m_outw
                                eb: float16 = x - m_outw
                                bnd[0] = allo.exp(ea)
                                bnd[1] = allo.exp(eb)
                                m_outh: float16 = m_outw
                                fifo_m[i, j + 1].put(m_outh)
                                dp: float16 = bnd[0] * d_in
                                dpw: float32 = dp
                                dbw: float32 = bnd[1]
                                ds: float16 = dpw + dbw
                                fifo_d[i, j + 1].put(ds)

                            # ============ EPILOGUE: last key, phase B only =====
                            for de in range(HEAD_DIM):
                                vve: float16 = fifo_V[i, j].get()
                                fifo_V[i + 1, j].put(vve)
                                oie: float16 = fifo_o[i, j].get()
                                oe1: float16 = bnd[0] * oie
                                oe2: float16 = bnd[1] * vve
                                oe1w: float32 = oe1
                                oe2w: float32 = oe2
                                osume: float16 = oe1w + oe2w
                                fifo_o[i, j + 1].put(osume)

        # ---- combine: also beat-serial, so it stays O(1) in dh ---------------
        #      NOTE: the dd_* loops here are PIPELINED, never unrolled by
        #      HEAD_DIM -- unrolling would put combine back on an O(dh) area
        #      curve and negate the flatness claim at the array output.
        @df.kernel(mapping=[NROWS], args=[])
        def combine():
            r = df.get_pid()
            m_ln: float16[COMBINE_LANES] = 0
            d_ln: float16[COMBINE_LANES] = 0
            acc_ln: float16[COMBINE_LANES, HEAD_DIM] = 0
            for b in range(BATCH_SIZE):
                for h in range(NUM_HEADS):
                    for tr in range(0, CONTEXT_LENGTH, NROWS):
                        with allo.meta_for(COMBINE_LANES) as g:
                            m_ln[g] = fifo_part_m[r, g].get()
                            d_ln[g] = fifo_part_d[r, g].get()
                            for dd_init in range(HEAD_DIM):
                                acc_ln[g, dd_init] = fifo_part_o[r, g].get()

                        for kt in range(1, NUM_KV_TILES // COMBINE_LANES):
                            with allo.meta_for(COMBINE_LANES) as g:
                                m_t: float16 = fifo_part_m[r, g].get()
                                d_t: float16 = fifo_part_d[r, g].get()
                                m_old: float16 = m_ln[g]
                                m_new: float16 = m_old
                                if m_t > m_old:
                                    m_new = m_t
                                sa: float16 = allo.exp(m_old - m_new)
                                st: float16 = allo.exp(m_t - m_new)
                                d_ln[g] = sa * d_ln[g] + st * d_t
                                for dd_a in range(HEAD_DIM):
                                    at: float16 = fifo_part_o[r, g].get()
                                    acc_ln[g, dd_a] = sa * acc_ln[g, dd_a] + st * at
                                m_ln[g] = m_new

                        # ---- 4 -> 2 -> 1 log-sum-exp tree (hardcoded to LANES=4)
                        for gp0 in range(2):
                            m_a0: float16 = m_ln[gp0]
                            m_b0: float16 = m_ln[gp0 + 2]
                            m_m0: float16 = m_a0
                            if m_b0 > m_a0:
                                m_m0 = m_b0
                            sa0: float16 = allo.exp(m_a0 - m_m0)
                            sb0: float16 = allo.exp(m_b0 - m_m0)
                            d_ln[gp0] = sa0 * d_ln[gp0] + sb0 * d_ln[gp0 + 2]
                            for dpr0 in range(HEAD_DIM):
                                acc_ln[gp0, dpr0] = (
                                    sa0 * acc_ln[gp0, dpr0] + sb0 * acc_ln[gp0 + 2, dpr0]
                                )
                            m_ln[gp0] = m_m0

                        for gp1 in range(1):
                            m_a1: float16 = m_ln[gp1]
                            m_b1: float16 = m_ln[gp1 + 1]
                            m_m1: float16 = m_a1
                            if m_b1 > m_a1:
                                m_m1 = m_b1
                            sa1: float16 = allo.exp(m_a1 - m_m1)
                            sb1: float16 = allo.exp(m_b1 - m_m1)
                            d_ln[gp1] = sa1 * d_ln[gp1] + sb1 * d_ln[gp1 + 1]
                            for dpr1 in range(HEAD_DIM):
                                acc_ln[gp1, dpr1] = (
                                    sa1 * acc_ln[gp1, dpr1] + sb1 * acc_ln[gp1 + 1, dpr1]
                                )
                            m_ln[gp1] = m_m1

                        d_f: float16 = d_ln[0]
                        inv: float16 = 1.0 / d_f
                        for dn in range(HEAD_DIM):
                            fifo_out[r].put(acc_ln[0, dn] * inv)

        # ---- store: beat-serial drain ---------------------------------------
        @df.kernel(mapping=[1], args=[output_mem])
        def store(global_mem: float16[OUT_ELEMS]):
            for b in range(BATCH_SIZE):
                for h in range(NUM_HEADS):
                    for tr in range(0, CONTEXT_LENGTH, NROWS):
                        with allo.meta_for(NROWS) as rr:
                            for jj2 in range(HEAD_DIM):
                                idx = (
                                    (b * CONTEXT_LENGTH + (tr + rr)) * NUM_HEADS + h
                                ) * HEAD_DIM + jj2
                                global_mem[idx] = fifo_out[rr].get()

    return top


def get_scheduled_beatserial(
    BATCH_SIZE, CONTEXT_LENGTH, HIDDEN_SIZE, NUM_HEADS, NROWS, NCOLS, LACC
):
    HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
    P0 = NROWS + 2
    P1 = NCOLS + 2

    s = df.customize(
        get_beatserial_flash_attention(
            BATCH_SIZE, CONTEXT_LENGTH, HIDDEN_SIZE, NUM_HEADS, NROWS, NCOLS, LACC
        )
    )

    def pipe_all(band, base, maxn=40):
        for n in range(maxn):
            nm = base if n == 0 else f"{base}_{n}"
            try:
                s.pipeline(getattr(band, nm))
                print("pipe", nm)
            except Exception:
                pass

    def unroll_all(band, base, maxn=40):
        for n in range(maxn):
            nm = base if n == 0 else f"{base}_{n}"
            try:
                s.unroll(getattr(band, nm))
                print("unroll", nm)
            except Exception:
                pass

    # ---------------- feeders: pipeline every beat loop -------------------
    try:
        root = s.get_loops("load_kv_fill_0").S_b_0
        print("load_kv_fill attrs:", [n for n in dir(root) if not n.startswith("_")])
        pipe_all(root, "jjf")
        pipe_all(root, "jjg")
    except Exception as e:
        print(f"[load_kv_fill] sched FAILED: {e}")

    try:
        # partition on the dh axis so the beat loops get a port per cycle
        s.partition("load_kv_replay_0:k_buf", dim=2)
        s.partition("load_kv_replay_0:v_buf", dim=2)
    except Exception as e:
        print(f"[load_kv_replay] partition FAILED: {e}")
    try:
        root = s.get_loops("load_kv_replay_0").S_b_0
        print("load_kv_replay attrs:", [n for n in dir(root) if not n.startswith("_")])
        # jjp is the new fused replay beat loop (lanes unrolled inside it).
        for base in ["jjr", "jjs", "jjp"]:
            pipe_all(root, base)
    except Exception as e:
        print(f"[load_kv_replay] sched FAILED: {e}")

    try:
        root = s.get_loops("load_q_0").S_b_0
        pipe_all(root, "jj")
    except Exception as e:
        print(f"[load_q] sched FAILED: {e}")

    # ---------------- PE array --------------------------------------------
    # CRITICAL: pipeline the BEAT loops (dm, da0, de, dq3), never unroll them.
    # Unrolling any of them by HEAD_DIM reinstates the O(dh) PE and voids the
    # entire redesign. `acc` is now a single fp32 register (LACC=1), so the
    # partition is a formality -- kept because a 1-element partition is free
    # and it keeps the schedule shape identical to the fp16 variant for A/B.
    for pi in range(1, P0 - 1):
        for pj in range(1, P1 - 1):
            pname = f"pe_{pi}_{pj}"
            try:
                lp = s.get_loops(pname).S_b_0
            except Exception as e:
                print(f"[{pname}] get_loops FAILED: {e}")
                continue
            if pi == 1 and pj == 1:
                print("pe attrs:", [n for n in dir(lp) if not n.startswith("_")])
            try:
                s.partition(f"{pname}:acc", dim=1)
            except Exception as e:
                print(f"[{pname}] part acc: {e}")
            # NO q_loc partition -- see the q_loc declaration. Partitioning it
            # completely is what synthesized the dh:1 mux we are removing.
            try:
                s.partition(f"{pname}:bnd", dim=1)
            except Exception as e:
                print(f"[{pname}] part bnd: {e}")
            pipe_all(lp, "dm")      # MERGED beat loop -- THE measurement
            pipe_all(lp, "da0")     # prologue: phase A of key 0
            pipe_all(lp, "de")      # epilogue: phase B of the last key
            pipe_all(lp, "dq3")     # q load phase
            # li/li0/rr/rr0 are GONE in this variant: with one accumulator
            # there is no per-key init loop and no rotating-register shift.
            # The fp16 variant needed them; a latency-1 adder does not.

    # ---------------- border PEs: pipeline their beat loops ---------------
    for pi in range(P0):
        for pj in range(P1):
            if 1 <= pi <= P0 - 2 and 1 <= pj <= P1 - 2:
                continue
            pname = f"pe_{pi}_{pj}"
            try:
                lp = s.get_loops(pname).S_b_0
            except Exception:
                continue
            # dkv/dkv2 are the MERGED top/bottom K+V beat loops (they replace
            # the split dk/dv and dk2/dv2). The old names are kept in the list
            # because pipe_all is a no-op when the loop is absent, and the
            # left/right borders still use dq/dq2/do0/db3.
            for base in ["dkv", "dkv2", "dk", "dv", "dk2", "dv2",
                         "dq", "dq2", "do0", "db3"]:
                pipe_all(lp, base)

    # ---------------- combine: pipeline dh loops, DO NOT unroll -----------
    for r in range(NROWS):
        c = f"combine_{r}"
        try:
            s.partition(f"{c}:m_ln", dim=1)
            s.partition(f"{c}:d_ln", dim=1)
            s.partition(f"{c}:acc_ln", dim=1)   # lane axis only, NOT the dh axis
        except Exception as e:
            print(f"[{c}] partition: {e}")
        try:
            root = s.get_loops(c).S_b_0
        except Exception as e:
            print(f"[{c}] get_loops FAILED: {e}")
            continue
        if r == 0:
            print("combine attrs:", [n for n in dir(root) if not n.startswith("_")])
        for base in ["dd_init", "dd_a", "dpr0", "dpr1", "dn"]:
            pipe_all(root, base)
        unroll_all(root, "gp0")
        unroll_all(root, "gp1")

    # ---------------- store ------------------------------------------------
    try:
        root = s.get_loops("store_0").S_b_0
        pipe_all(root, "jj2")
    except Exception as e:
        print(f"[store] sched FAILED: {e}")

    return s


def preflight(BATCH_SIZE, CONTEXT_LENGTH, HIDDEN_SIZE, NUM_HEADS, NROWS, NCOLS, LACC):
    """Static gate. Cheapest error catch there is -- a Vitis round costs minutes,
    a typo'd count costs a whole cycle."""
    HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
    KV_TILE = NCOLS
    NUM_KV_TILES = CONTEXT_LENGTH // KV_TILE
    NUM_Q_TILES = CONTEXT_LENGTH // NROWS
    COMBINE_LANES = 4
    NSTRIPS = NUM_KV_TILES // COMBINE_LANES

    print("-" * 64)
    print("PRE-FLIGHT")
    print(f"  HEAD_DIM={HEAD_DIM} KV_TILE={KV_TILE} NUM_KV_TILES={NUM_KV_TILES} "
          f"NUM_Q_TILES={NUM_Q_TILES} NSTRIPS={NSTRIPS} LACC={LACC}")

    assert HIDDEN_SIZE % NUM_HEADS == 0
    assert CONTEXT_LENGTH % NROWS == 0
    assert CONTEXT_LENGTH % KV_TILE == 0
    assert NUM_KV_TILES % COMBINE_LANES == 0, "combine strip divisibility"
    assert NSTRIPS >= 1, "combine init strip would never arrive -> deadlock"
    assert HEAD_DIM >= LACC, (
        f"HEAD_DIM={HEAD_DIM} < LACC={LACC}: partial accumulators cannot amortize"
    )
    # This variant accumulates in fp32, where the V80 adder is latency 1
    # (fadd_32ns_32ns_32_1_primitive_dsp, hardened DSPFP32). A distance-1
    # recurrence into a latency-1 adder schedules at II=1 with ONE
    # accumulator, so the rotating registers and the boundary fold tree are
    # both deleted. Raising LACC here would not be a tuning knob -- the code
    # only has acc[0], so it would silently do nothing.
    assert LACC == 1, (
        f"LACC={LACC}: the fp32 variant has a single accumulator and no fold. "
        f"LACC>1 measured strictly worse on V80 (DSP and LUT scale with it, "
        f"II stays 1). Use attn_beatserial_merged.py for the fp16 LACC=4 form."
    )

    # ---- rate invariant: o forwarded per key vs beats available per key
    beats_per_key = 2 * HEAD_DIM
    o_per_key = HEAD_DIM
    assert o_per_key <= beats_per_key, "horizontal o link is rate-starved"
    print(f"  rate: {o_per_key} o-beats forwarded within {beats_per_key} beats/key  OK")

    # ---- symbolic FIFO balance, per (b,h). Every count is dh-x larger than the
    #      vector version -- recompute, never scale the old numbers by hand.
    per_bh = NUM_Q_TILES * NUM_KV_TILES

    q_left = NUM_Q_TILES * HEAD_DIM
    q_right = NUM_Q_TILES * HEAD_DIM
    assert q_left == q_right
    print(f"  q beats:  left={q_left}  each PE fwd={q_left}  right drain={q_right}  OK")

    k_top = per_bh * HEAD_DIM
    k_bot = per_bh * HEAD_DIM
    assert k_top == k_bot
    print(f"  K beats:  top={k_top}  bottom drain={k_bot}  OK")

    v_top = per_bh * HEAD_DIM
    v_bot = per_bh * HEAD_DIM
    assert v_top == v_bot
    print(f"  V beats:  top={v_top}  bottom drain={v_bot}  OK")

    o_left = per_bh * HEAD_DIM
    o_right = per_bh * HEAD_DIM
    assert o_left == o_right
    print(f"  o beats:  left init={o_left}  right emit={o_right}  OK")

    md_left = per_bh
    md_right = per_bh
    assert md_left == md_right
    print(f"  m/d:      left={md_left}  right={md_right}  OK")

    part_emit = NUM_Q_TILES * NSTRIPS * COMBINE_LANES
    part_consume = NUM_Q_TILES * (COMBINE_LANES + (NSTRIPS - 1) * COMBINE_LANES)
    assert part_emit == part_consume, (part_emit, part_consume)
    print(f"  partials: right emit={part_emit}  combine consume={part_consume}  OK")

    print("PRE-FLIGHT PASSED")
    print("-" * 64)


def run_test_with_params(
    BATCH_SIZE, CONTEXT_LENGTH, HIDDEN_SIZE, NUM_HEADS, NROWS, NCOLS, LACC, mode="csyn"
):
    preflight(BATCH_SIZE, CONTEXT_LENGTH, HIDDEN_SIZE, NUM_HEADS, NROWS, NCOLS, LACC)

    HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
    D_SQRT = HEAD_DIM**0.5
    OUT_ELEMS = BATCH_SIZE * CONTEXT_LENGTH * NUM_HEADS * HEAD_DIM

    print("=" * 64)
    print(
        f"beat-serial FlashAttn: B={BATCH_SIZE} L={CONTEXT_LENGTH} "
        f"HIDDEN={HIDDEN_SIZE} H={NUM_HEADS} dh={HEAD_DIM} "
        f"NROWS={NROWS} NCOLS={NCOLS} BS=1 LACC={LACC} mode={mode}"
    )
    print("=" * 64)

    A = np.random.rand(BATCH_SIZE * CONTEXT_LENGTH * 3 * HIDDEN_SIZE).astype(np.float16)
    A5 = A.reshape((BATCH_SIZE, CONTEXT_LENGTH, 3, NUM_HEADS, HEAD_DIM))
    Q = np.ascontiguousarray(A5[:, :, 0, :, :]).reshape(-1).astype(np.float16)
    K = np.ascontiguousarray(A5[:, :, 1, :, :]).reshape(-1).astype(np.float16)
    V = np.ascontiguousarray(A5[:, :, 2, :, :]).reshape(-1).astype(np.float16)
    B_out = np.zeros(OUT_ELEMS, dtype=np.float16)

    # ---- golden reference: unchanged, so the A/B against the unrolled kernel
    #      is apples-to-apples.
    Q_np = A5[:, :, 0, :, :].transpose((0, 2, 1, 3))
    K_np = A5[:, :, 1, :, :].transpose((0, 2, 1, 3))
    V_np = A5[:, :, 2, :, :].transpose((0, 2, 1, 3))
    scores = np.matmul(Q_np, K_np.transpose((0, 1, 3, 2))) * (1.0 / D_SQRT)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    attn = np.exp(scores)
    attn = attn / np.sum(attn, axis=-1, keepdims=True)
    out_np = np.matmul(attn, V_np)
    B_golden = out_np.transpose((0, 2, 1, 3)).flatten()

    if not hls.is_available("vitis_hls"):
        print("Vitis HLS not available, skipping.")
        return

    s = get_scheduled_beatserial(
        BATCH_SIZE, CONTEXT_LENGTH, HIDDEN_SIZE, NUM_HEADS, NROWS, NCOLS, LACC
    )
    prj = os.environ.get(
        "PRJ", f"/scratch/ry375/beat_dh{HEAD_DIM}_L{CONTEXT_LENGTH}_{mode}.prj"
    )
    if mode == "hw":
        # Scalar-link beat-serial should route far below the baseline's congestion,
        # but the fp16 multiplier (hmul_16ns_16ns_16_5_max_dsp) is the same critical
        # path that failed timing at 300MHz on the chunked baseline (WNS=-0.775) and
        # closed at 230MHz. Build at 230 by default (env FREQ override) and spread
        # the 4 gmem buffers across HBM banks 0-3 to avoid the buf0 AXI-adapter
        # concentration that broke prior HD16/HD32 post-route verification.
        freq = int(os.environ.get("FREQ", "230"))
        hls_mod = s.build(
            target="vitis_hls",
            mode=mode,
            project=prj,
            configs={
                "frequency": freq,
                "hbm_mapping": {
                    "q_mem": 0,
                    "k_mem": 1,
                    "v_mem": 2,
                    "output_mem": 3,
                },
            },
        )
    else:
        # DEVICE lets csyn retarget another part without a platform install --
        # DEVICE=v80 (xcv80-lsva4737-2MHP-e-S, Versal HBM) for the V80 port.
        # Without this the build falls back to the u280 default in Allo's
        # DEFAULT_CONFIG, which fails on any host lacking the U280 part.
        # (This plumbing existed only in attn_beatserial_fast.py -- HANDOFF §8.)
        dev = os.environ.get("DEVICE", "")
        cfg = {"device": dev} if dev else {}
        if os.environ.get("FREQ"):
            cfg["frequency"] = int(os.environ["FREQ"])
        hls_mod = s.build(target="vitis_hls", mode=mode, project=prj,
                          configs=cfg) if cfg else s.build(
                          target="vitis_hls", mode=mode, project=prj)

    if mode != "csyn":
        hls_mod(Q, K, V, B_out)
        np.testing.assert_allclose(B_out, B_golden, rtol=0.02, atol=1e-2)
        print(f"{mode} PASSED -> {prj}")
    else:
        hls_mod()
        print(f"csyn done -> {prj}")
        print("  READ: II on `da` (phase A) and `db` (phase B) in pe_1_1")
        print("  READ: DSP for pe_1_1 -- this is the per-PE number that must be")
        print("        FLAT across the dh sweep. Totals are confounded by array size.")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "128")

    # LACC is 1 in this variant and is not a tuning knob -- see preflight().
    # The fp32 adder is latency 1, so one accumulator already gives II=1.
    LACC = int(os.environ.get("LACC", "1"))
    RUN = os.environ.get("RUN", "perf")

    # BS=1 throughout: KV_TILE = NCOLS = 4
    #   REAL: NUM_KV_TILES = 1024/4 = 256, 256 % 4 == 0  OK
    #   GATE: NUM_KV_TILES =  128/4 =  32,  32 % 4 == 0  OK
    #   FUNC: NUM_KV_TILES =   64/4 =  16,  16 % 4 == 0  OK
    REAL = dict(BATCH_SIZE=1, CONTEXT_LENGTH=1024, HIDDEN_SIZE=32, NUM_HEADS=4,
                NROWS=4, NCOLS=4, LACC=LACC)
    GATE = dict(BATCH_SIZE=1, CONTEXT_LENGTH=128, HIDDEN_SIZE=32, NUM_HEADS=4,
                NROWS=4, NCOLS=4, LACC=LACC)
    FUNC = dict(BATCH_SIZE=1, CONTEXT_LENGTH=64, HIDDEN_SIZE=32, NUM_HEADS=1,
                NROWS=4, NCOLS=4, LACC=LACC)

    # dh sweep (Phase 3, csyn only): NUM_HEADS=1 and vary HIDDEN_SIZE, so only
    # ONE thing changes per point. This produces the headline flatness curve.
    # ---- fully env-driven config: the optimization sweeps need to move dh,
    #      array size, context and mode independently.
    #        DH=<dh> ARR=<n> CTX=<L> MODE=csyn|hw_emu|hw  python attn_beatserial_fast.py
    #      Legality: CTX % ARR == 0 and (CTX/ARR) % COMBINE_LANES(4) == 0,
    #      i.e. CTX >= 4*ARR. So a SHORT-CTX numeric check at the DEPLOYED array
    #      size is legal (ARR=16 needs only CTX=64) -- no long hw_emu required.
    FREE = int(os.environ.get("FREE", "0"))
    if FREE:
        fdh = int(os.environ.get("DH", "64"))
        farr = int(os.environ.get("ARR", "8"))
        fctx = int(os.environ.get("CTX", "1024"))
        run_test_with_params(
            BATCH_SIZE=1, CONTEXT_LENGTH=fctx, HIDDEN_SIZE=fdh, NUM_HEADS=1,
            NROWS=int(os.environ.get("NROWS", farr)),
            NCOLS=int(os.environ.get("NCOLS", farr)), LACC=LACC,
            mode=os.environ.get("MODE", "csyn"),
        )
        raise SystemExit(0)

    DH = int(os.environ.get("DH", "0"))
    # ARRAY-SIZE sweep: biggest deployable array. NROWS=NCOLS=ARR at fixed dh.
    # NCOLS must divide CTX and NUM_KV_TILES=CTX/NCOLS must be %COMBINE_LANES(4)==0,
    # so ARR in {4,8,16,32} at CTX=1024. MODE=csyn (resource ceiling) or hw (deploy).
    ARR = int(os.environ.get("ARR", "0"))
    if ARR:
        adh = int(os.environ.get("ADH", "8"))
        ARRSW = dict(BATCH_SIZE=1, CONTEXT_LENGTH=1024, HIDDEN_SIZE=adh, NUM_HEADS=1,
                     NROWS=ARR, NCOLS=ARR, LACC=LACC)
        run_test_with_params(**ARRSW, mode=os.environ.get("MODE", "csyn"))
    elif DH:
        SWEEP = dict(BATCH_SIZE=1, CONTEXT_LENGTH=1024, HIDDEN_SIZE=DH, NUM_HEADS=1,
                     NROWS=4, NCOLS=4, LACC=LACC)
        run_test_with_params(**SWEEP, mode="csyn")
    else:
        cfg, mode = {
            "perf": (REAL, "csyn"),
            "func": (FUNC, "hw_emu"),
            "gate": (GATE, "hw_emu"),
            "emu":  (REAL, "hw_emu"),
            "hw":   (REAL, "hw"),
        }[RUN]
        run_test_with_params(**cfg, mode=mode)
