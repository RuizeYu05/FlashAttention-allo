# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Systolic Array Flash Attention — split-K + combine (feed-forward, acyclic)
==========================================================================

Replaces the cyclic feedback design with a split-K / flash-decoding scheme so
the dataflow graph is a pure DAG (no right->left feedback FIFO):

  * queries are tiled (NROWS rows per query tile, looped over NUM_Q_TILES),
  * keys/values are tiled into KV tiles of (NCOLS * BS) entries,
  * EACH KV tile is processed INDEPENDENTLY by the array starting from a fresh
    state (m=-inf, d'=0, o'=0); the right border emits the tile's *partial*
    (m_t, d'_t, o'_t) instead of feeding it back,
  * a `combine` kernel reduces the NUM_KV_TILES partials per query row with the
    standard log-sum-exp merge and writes the final output.

Because every KV-tile pass is independent, there is no loop-carried dependency
across tiles in the array, so no feedback edge, so no cycle — and KV tiles can
even pipeline through the array back-to-back.

Array geometry (compute core NROWS x NCOLS, 1-wide border):

       j=0          j=1 .. NCOLS          j=P1-1
  i=0   corner   [ K/V feed (top) ]       corner
  i=1  [Q + fresh                          [ collect partial
        init  ]  [   compute PEs   ]         (m,d,o) -> combine ]
   ..             (online softmax)
  i=NROWS
  i=P0-1 corner  [   drain K/V     ]       corner

Per-tile partial (online-softmax output is already normalised by the tile
denominator):
  m_t = max_{i in tile} x_i
  d_t = sum_{i in tile} exp(x_i - m_t)
  o_t = ( sum_{i in tile} exp(x_i - m_t) v_i ) / d_t

Merge of accumulator (m_a,d_a,o_a) with tile (m_t,d_t,o_t):
  m   = max(m_a, m_t)
  sa  = exp(m_a - m),  st = exp(m_t - m)
  d   = sa*d_a + st*d_t
  o   = ( sa*d_a*o_a + st*d_t*o_t ) / d        # un-normalise, rescale, re-normalise

NOTE on dtype: this is float32 end-to-end. The csim/nanobind host boundary
cannot bind `half`, so float16 only works on the hw_emu/hw (XRT) path. Validate
logic in float32 first, then switch the stream/interface types back to float16
for synthesis if you want the narrow datapath.
"""

import os
import tempfile
import argparse

import numpy as np
import allo
import allo.dataflow as df
import allo.backend.hls as hls
from allo.ir.types import float16, Stream


def get_kv_tiled_flash_attention(
    BATCH_SIZE: int,
    CONTEXT_LENGTH: int,
    HIDDEN_SIZE: int,
    NUM_HEADS: int,
    NROWS: int,
    NCOLS: int,
    BS: int,
):
    HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
    KV_TILE = NCOLS * BS
    NUM_KV_TILES = CONTEXT_LENGTH // KV_TILE
    NUM_Q_TILES = CONTEXT_LENGTH // NROWS

    assert CONTEXT_LENGTH % KV_TILE == 0, "CONTEXT_LENGTH must be divisible by NCOLS*BS"
    assert CONTEXT_LENGTH % NROWS == 0, "CONTEXT_LENGTH must be divisible by NROWS"

    P0 = NROWS + 2
    P1 = NCOLS + 2

    D_SQRT = float(HEAD_DIM**0.5)
    D_SCALE = 1.0 / D_SQRT  # 1/sqrt(head_dim), applied to Q (matches the reference)
    QKV_ELEMS = BATCH_SIZE * CONTEXT_LENGTH * HIDDEN_SIZE   # each of Q,K,V, contiguous [B,T,H,Dh]
    OUT_ELEMS = BATCH_SIZE * CONTEXT_LENGTH * NUM_HEADS * HEAD_DIM
    M_INIT = -1e4
    COMBINE_LANES = 4
    assert NUM_KV_TILES % COMBINE_LANES == 0
    assert NUM_KV_TILES >= COMBINE_LANES


    @df.region()
    def top(
        q_mem: float16[QKV_ELEMS],
        k_mem: float16[QKV_ELEMS],
        v_mem: float16[QKV_ELEMS],
        output_mem: float16[OUT_ELEMS],
    ):
        # ---- inter-PE streams ----
        fifo_Q: Stream[float16[HEAD_DIM], 32][P0, P1]   # Q, left->right; depth 32 shrinks capacity req (NCOLS-1)*32*BS
        fifo_K: Stream[float16[HEAD_DIM], 64][P0, P1]  # K,          top  -> bottom
        fifo_V: Stream[float16[HEAD_DIM], 64][P0, P1]  # V,          top  -> bottom
        fifo_m: Stream[float16, 64][P0, P1]             # running max m,  left -> right
        fifo_d: Stream[float16, 64][P0, P1]             # running denom d, left -> right
        fifo_o: Stream[float16[HEAD_DIM], 64][P0, P1]   # running out  o, left -> right

        # ---- feeders from load ----
        fifo_in_Q: Stream[float16[HEAD_DIM], 512][P0 - 2]
        fifo_in_K: Stream[float16[HEAD_DIM], 512][P1 - 2]    # 512 > (NCOLS-1)*depth(fifo_Q)*BS=(4-1)*32*4=384: capacity-safe, stays out of LUTRAM-heavy depth
        fifo_in_V: Stream[float16[HEAD_DIM], 512][P1 - 2]    # match K (same lockstep feeder)

        # ---- fill -> replay handoff (one full sweep of all KV tiles, in
        #      replay's first-query-tile consumption order: tile, kf, c) ----
        fifo_fill_K: Stream[float16[HEAD_DIM], 512]
        fifo_fill_V: Stream[float16[HEAD_DIM], 512]

        # ---- per-tile partials: right border -> combine (one set per query row, per lane) ----
        fifo_part_m: Stream[float16, 8][P0 - 2, COMBINE_LANES]
        fifo_part_d: Stream[float16, 8][P0 - 2, COMBINE_LANES]
        fifo_part_o: Stream[float16[HEAD_DIM], 8][P0 - 2, COMBINE_LANES]

        fifo_out: Stream[float16[HEAD_DIM], 512][NROWS]

        @df.kernel(mapping=[1], args=[q_mem])
        def load_q(q_d: float16[QKV_ELEMS]):
            for b in range(BATCH_SIZE):
                for h in range(NUM_HEADS):
                    for tr in range(0, CONTEXT_LENGTH, NROWS):
                        with allo.meta_for(NROWS) as r:
                            q_vec: float16[HEAD_DIM] = 0
                            t_q = tr + r
                            for jj in range(HEAD_DIM):
                                q_idx = b*(CONTEXT_LENGTH*HIDDEN_SIZE) + t_q*HIDDEN_SIZE + h*HEAD_DIM + jj
                                sc: float16 = D_SCALE
                                q_vec[jj] = q_d[q_idx] * sc
                            fifo_in_Q[r].put(q_vec)

        # ---- fill: read K/V from DRAM (sequential, single AXI port) and stream
        #      one full sweep of all KV tiles into the handoff FIFOs.  Order
        #      (tile, kf, c) matches the replay's first-query-tile consumption
        #      order so the replay can forward directly while it buffers. ----
        @df.kernel(mapping=[1], args=[k_mem, v_mem])
        def load_kv_fill(k_d: float16[QKV_ELEMS], v_d: float16[QKV_ELEMS]):
            for b in range(BATCH_SIZE):
                for h in range(NUM_HEADS):
                    for tile_f in range(NUM_KV_TILES):
                        for kf in range(BS):
                            for c in range(NCOLS):
                                t = tile_f * KV_TILE + c * BS + kf
                                base = b * (CONTEXT_LENGTH * HIDDEN_SIZE) + t * HIDDEN_SIZE + h * HEAD_DIM
                                k_vec: float16[HEAD_DIM] = 0
                                v_vec: float16[HEAD_DIM] = 0
                                for jjf in range(HEAD_DIM):
                                    k_vec[jjf] = k_d[base + jjf]
                                    v_vec[jjf] = v_d[base + jjf]
                                fifo_fill_K.put(k_vec)
                                fifo_fill_V.put(v_vec)

        # ---- replay: first query tile is fed straight from the fill stream
        #      (overlapping the DRAM fill) while it is captured into k_buf/v_buf;
        #      all remaining query tiles replay from the on-chip buffer. ----
        @df.kernel(mapping=[1], args=[])
        def load_kv_replay():
            k_buf: float16[NUM_KV_TILES, NCOLS, BS, HEAD_DIM]
            v_buf: float16[NUM_KV_TILES, NCOLS, BS, HEAD_DIM]
            for b in range(BATCH_SIZE):
                for h in range(NUM_HEADS):
                    for tile_r in range(NUM_KV_TILES):
                        for kr in range(BS):
                            with allo.meta_for(NCOLS) as c:
                                k_vec: float16[HEAD_DIM] = fifo_fill_K.get()
                                v_vec: float16[HEAD_DIM] = fifo_fill_V.get()
                                for jjr in range(HEAD_DIM):
                                    k_buf[tile_r, c, kr, jjr] = k_vec[jjr]
                                    v_buf[tile_r, c, kr, jjr] = v_vec[jjr]
                                fifo_in_K[c].put(k_vec)
                                fifo_in_V[c].put(v_vec)

                    for tr in range(NROWS, CONTEXT_LENGTH, NROWS):
                        for tile_rr in range(NUM_KV_TILES):
                            for krr in range(BS):
                                with allo.meta_for(NCOLS) as c2:
                                    k2_vec: float16[HEAD_DIM] = 0
                                    v2_vec: float16[HEAD_DIM] = 0
                                    for jjr2 in range(HEAD_DIM):
                                        k2_vec[jjr2] = k_buf[tile_rr, c2, krr, jjr2]
                                        v2_vec[jjr2] = v_buf[tile_rr, c2, krr, jjr2]
                                    fifo_in_K[c2].put(k2_vec)
                                    fifo_in_V[c2].put(v2_vec)

        @df.kernel(mapping=[P0, P1], args=[])
        def pe():
            i, j = df.get_pid()

            x_buf: float16[BS] = 0
            v_loc: float16[BS, HEAD_DIM] = 0
            o_acc: float16[HEAD_DIM] = 0

            # ---- corners ----
            with allo.meta_if(i in {0, P0 - 1} and j in {0, P1 - 1}):
                pass

            # ---- top border: push K/V down, BS per tile ----
            with allo.meta_elif(i == 0):
                for b in range(BATCH_SIZE):
                    for h in range(NUM_HEADS):
                        for tr in range(0, CONTEXT_LENGTH, NROWS):
                            for tc in range(0, CONTEXT_LENGTH, KV_TILE):
                                for k in range(BS):
                                    k_vec: float16[HEAD_DIM] = fifo_in_K[j - 1].get()
                                    v_vec: float16[HEAD_DIM] = fifo_in_V[j - 1].get()
                                    fifo_K[i + 1, j].put(k_vec)
                                    fifo_V[i + 1, j].put(v_vec)

            # ---- left border: inject Q + FRESH state for every KV tile (no feedback) ----
            with allo.meta_elif(j == 0):
                for b in range(BATCH_SIZE):
                    for h in range(NUM_HEADS):
                        for tr in range(0, CONTEXT_LENGTH, NROWS):
                            q_local: float16[HEAD_DIM] = fifo_in_Q[i - 1].get()
                            for kt in range(NUM_KV_TILES):
                                m_init: float16 = -1e30
                                d_init: float16 = 0.0
                                o_init: float16[HEAD_DIM] = 0
                                fifo_Q[i, j + 1].put(q_local)
                                fifo_m[i, j + 1].put(m_init)
                                fifo_d[i, j + 1].put(d_init)
                                fifo_o[i, j + 1].put(o_init)

            # ---- bottom border: drain K/V ----
            with allo.meta_elif(i == P0 - 1):
                for b in range(BATCH_SIZE):
                    for h in range(NUM_HEADS):
                        for tr in range(0, CONTEXT_LENGTH, NROWS):
                            for kt in range(NUM_KV_TILES):
                                for k in range(BS):
                                    _k: float16[HEAD_DIM] = fifo_K[i, j].get()
                                    _v: float16[HEAD_DIM] = fifo_V[i, j].get()

            # ---- right border: emit the per-tile partial to combine ----
            with allo.meta_elif(j == P1 - 1):
                for b in range(BATCH_SIZE):
                    for h in range(NUM_HEADS):
                        for tr in range(0, CONTEXT_LENGTH, NROWS):
                            for kt_strip in range(NUM_KV_TILES // COMBINE_LANES):
                                with allo.meta_for(COMBINE_LANES) as lane:
                                    _q: float16[HEAD_DIM] = fifo_Q[i, j].get()
                                    m_t: float16 = fifo_m[i, j].get()
                                    d_t: float16 = fifo_d[i, j].get()
                                    acc_t: float16[HEAD_DIM] = fifo_o[i, j].get()

                                    fifo_part_m[i - 1, lane].put(m_t)
                                    fifo_part_d[i - 1, lane].put(d_t)
                                    fifo_part_o[i - 1, lane].put(acc_t)

            # ---- compute PEs: fold BS keys into the (per-tile) running state ----
            with allo.meta_else():
                for b in range(BATCH_SIZE):
                    for h in range(NUM_HEADS):
                        for qt in range(NUM_Q_TILES):
                            for kt in range(NUM_KV_TILES):
                                q_vec: float16[HEAD_DIM] = fifo_Q[i, j].get()
                                m_in: float16 = fifo_m[i, j].get()
                                d_in: float16 = fifo_d[i, j].get()
                                acc_in: float16[HEAD_DIM] = fifo_o[i, j].get()
                                m_t: float16 = m_in

                                for k in range(BS):
                                    k_vec: float16[HEAD_DIM] = fifo_K[i, j].get()
                                    v_vec: float16[HEAD_DIM] = fifo_V[i, j].get()
                                    fifo_K[i + 1, j].put(k_vec)
                                    fifo_V[i + 1, j].put(v_vec)

                                    x: float16 = 0.0
                                    for dd_dot in range(HEAD_DIM):
                                        x += q_vec[dd_dot] * k_vec[dd_dot]
                                    x_buf[k] = x
                                    for dd_cp in range(HEAD_DIM):
                                        v_loc[k, dd_cp] = v_vec[dd_cp]
                                    if x > m_t:
                                        m_t = x

                                corr: float16 = allo.exp(m_in - m_t)
                                d_acc: float16 = d_in * corr
                                for dd_rs in range(HEAD_DIM):
                                    o_acc[dd_rs] = acc_in[dd_rs] * corr

                                for k2 in range(BS):
                                    p: float16 = allo.exp(x_buf[k2] - m_t)
                                    d_acc += p
                                    for dd_acc in range(HEAD_DIM):
                                        o_acc[dd_acc] += p * v_loc[k2, dd_acc]

                                fifo_Q[i, j+1].put(q_vec)
                                fifo_m[i, j+1].put(m_t)
                                fifo_d[i, j+1].put(d_acc)
                                fifo_o[i, j+1].put(o_acc)

        @df.kernel(mapping=[NROWS], args=[])
        def combine():
            r = df.get_pid()
            m_ln: float16[COMBINE_LANES] = 0
            d_ln: float16[COMBINE_LANES] = 0
            acc_ln: float16[COMBINE_LANES, HEAD_DIM] = 0
            o_f: float16[HEAD_DIM] = 0
            for b in range(BATCH_SIZE):
                for h in range(NUM_HEADS):
                    for tr in range(0, CONTEXT_LENGTH, NROWS):
                        with allo.meta_for(COMBINE_LANES) as g:
                            m_t: float16 = fifo_part_m[r, g].get()
                            d_t: float16 = fifo_part_d[r, g].get()
                            acc_t: float16[HEAD_DIM] = fifo_part_o[r, g].get()
                            m_ln[g] = m_t
                            d_ln[g] = d_t
                            for dd_init in range(HEAD_DIM):
                                acc_ln[g, dd_init] = acc_t[dd_init]
                        for kt in range(1, NUM_KV_TILES // COMBINE_LANES):
                            with allo.meta_for(COMBINE_LANES) as g:
                                m_t: float16 = fifo_part_m[r, g].get()
                                d_t: float16 = fifo_part_d[r, g].get()
                                acc_t: float16[HEAD_DIM] = fifo_part_o[r, g].get()
                                m_old: float16 = m_ln[g]
                                m_new: float16 = m_old
                                if m_t > m_old:
                                    m_new = m_t
                                sa: float16 = allo.exp(m_old - m_new)
                                st: float16 = allo.exp(m_t   - m_new)
                                d_ln[g] = sa * d_ln[g] + st * d_t
                                for dd_a in range(HEAD_DIM):
                                    acc_ln[g, dd_a] = sa * acc_ln[g, dd_a] + st * acc_t[dd_a]
                                m_ln[g] = m_new
                        # ---- balanced binary-tree log-sum-exp merge over COMBINE_LANES=4 lanes ----
                        # (tree is hardcoded to COMBINE_LANES; a 4-lane tree is 2 levels: 4->2->1)
                        # level 0: 4 -> 2  (pairs 0+2, 1+3)
                        for gp0 in range(2):
                            m_a0: float16 = m_ln[gp0]
                            m_b0: float16 = m_ln[gp0 + 2]
                            m_m0: float16 = m_a0
                            if m_b0 > m_a0:
                                m_m0 = m_b0
                            sa0: float16 = allo.exp(m_a0 - m_m0)
                            sb0: float16 = allo.exp(m_b0 - m_m0)
                            d_ln[gp0] = sa0 * d_ln[gp0] + sb0 * d_ln[gp0 + 2]
                            with allo.meta_for(HEAD_DIM) as dpr0:
                                acc_ln[gp0, dpr0] = sa0 * acc_ln[gp0, dpr0] + sb0 * acc_ln[gp0 + 2, dpr0]
                            m_ln[gp0] = m_m0
                        # level 1: 2 -> 1  (pair 0+1)
                        for gp1 in range(1):
                            m_a1: float16 = m_ln[gp1]
                            m_b1: float16 = m_ln[gp1 + 1]
                            m_m1: float16 = m_a1
                            if m_b1 > m_a1:
                                m_m1 = m_b1
                            sa1: float16 = allo.exp(m_a1 - m_m1)
                            sb1: float16 = allo.exp(m_b1 - m_m1)
                            d_ln[gp1] = sa1 * d_ln[gp1] + sb1 * d_ln[gp1 + 1]
                            with allo.meta_for(HEAD_DIM) as dpr1:
                                acc_ln[gp1, dpr1] = sa1 * acc_ln[gp1, dpr1] + sb1 * acc_ln[gp1 + 1, dpr1]
                            m_ln[gp1] = m_m1
                        d_f: float16 = d_ln[0]
                        inv: float16 = 1.0 / d_f
                        with allo.meta_for(HEAD_DIM) as dn:
                            o_f[dn] = acc_ln[0, dn] * inv
                        fifo_out[r].put(o_f)


        # ---- store: single write port; drain the NROWS row results into output_mem ----
        @df.kernel(mapping=[1], args=[output_mem])
        def store(global_mem: float16[OUT_ELEMS]):
            for b in range(BATCH_SIZE):
                for h in range(NUM_HEADS):
                    for tr in range(0, CONTEXT_LENGTH, NROWS):
                        with allo.meta_for(NROWS) as rr:
                            out_vec: float16[HEAD_DIM] = fifo_out[rr].get()
                            for jj in range(HEAD_DIM):
                                idx = (
                                    (b * CONTEXT_LENGTH + (tr + rr)) * NUM_HEADS + h
                                ) * HEAD_DIM + jj
                                global_mem[idx] = out_vec[jj]

    return top


def get_scheduled_systolic(
    BATCH_SIZE, CONTEXT_LENGTH, HIDDEN_SIZE, NUM_HEADS, NROWS, NCOLS, BS
):
    HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
    P0 = NROWS + 2
    P1 = NCOLS + 2
    COMBINE_LANES = 4

    s = df.customize(
        get_kv_tiled_flash_attention(
            BATCH_SIZE, CONTEXT_LENGTH, HIDDEN_SIZE, NUM_HEADS, NROWS, NCOLS, BS
        )
    )

    def pipe_all(band, base, maxn=40):
        for n in range(maxn):
            nm = base if n == 0 else f"{base}_{n}"
            try: s.pipeline(getattr(band, nm)); print("pipe", nm)
            except Exception: pass

    def unroll_all(band, base, maxn=40):
        for n in range(maxn):
            nm = base if n == 0 else f"{base}_{n}"
            try: s.unroll(getattr(band, nm)); print("unroll", nm)
            except Exception: pass

    # ---------------- load_kv_fill: sequential DRAM read -> handoff FIFO -------------
    try:
        root = s.get_loops("load_kv_fill_0").S_b_0
        print("load_kv_fill attrs:", [n for n in dir(root) if not n.startswith("_")])
        pipe_all(root, "jjf")    # fill-path DRAM-read / vec build
        pipe_all(root, "c")      # innermost column push (sequential)
    except Exception as e:
        print(f"[load_kv_fill] sched FAILED: {e}")

    # ---------------- load_kv_replay: buffer + replay, both paths pipelined ----------
    try:
        s.partition("load_kv_replay_0:k_buf", dim=2)
        s.partition("load_kv_replay_0:v_buf", dim=2)
        s.partition("load_kv_replay_0:k_buf", dim=4)
        s.partition("load_kv_replay_0:v_buf", dim=4)
    except Exception as e:
        print(f"[load_kv_replay] partition FAILED: {e}")
    try:
        root = s.get_loops("load_kv_replay_0").S_b_0
        print("load_kv_replay attrs:", [n for n in dir(root) if not n.startswith("_")])
        pipe_all(root, "kr")      # first-tile capture+forward push
        pipe_all(root, "krr")     # replay-from-buffer push
        unroll_all(root, "jjr")   # NCOLS copies, first-tile capture
        unroll_all(root, "jjr2")  # NCOLS copies, replay-from-buffer
    except Exception as e:
        print(f"[load_kv_replay] sched FAILED: {e}")

    # ---------------- load_q: concurrent Q read ----------------
    root = s.get_loops("load_q_0").S_b_0
    pipe_all(root, "tr")
    unroll_all(root, "jj")

    # ---------------- pe array ----------------
    for pi in range(1, P0 - 1):
        for pj in range(1, P1 - 1):
            pname = f"pe_{pi}_{pj}"
            try: lp = s.get_loops(pname).S_b_0
            except Exception as e:
                print(f"[{pname}] get_loops FAILED: {e}"); continue
            for arr, d in [("x_buf", 1), ("v_loc", 0), ("o_acc", 1)]:
                try: s.partition(f"{pname}:{arr}", dim=d)
                except Exception as e: print(f"[{pname}] part {arr}: {e}")
            try: s.pipeline(lp.kt)         # auto-unrolls the nested BS `k` loop
            except Exception as e: print(f"[{pname}] pipe kt: {e}")
            for ln in ["dd_dot", "dd_cp", "dd_rs", "dd_acc", "k2"]:
                try: s.unroll(getattr(lp, ln))
                except Exception: pass

    def unroll_partial(band, base, factor, maxn=40):
        for n in range(maxn):
            nm = base if n == 0 else f"{base}_{n}"
            try: s.unroll(getattr(band, nm), factor=factor); print("unroll_p", nm, factor)
            except Exception: pass

    # ---------------- combine: folded init + tree-reduced merge ----------------
    for r in range(NROWS):
        c = f"combine_{r}"
        s.partition(f"{c}:m_ln", dim=1)
        s.partition(f"{c}:d_ln", dim=1)
        s.partition(f"{c}:acc_ln", dim=0)
        s.partition(f"{c}:acc_ln", dim=1)
        root = s.get_loops(c).S_b_0
        if r == 0:
            print("combine attrs:", [n for n in dir(root) if not n.startswith("_")])
        try: s.pipeline(root.kt)
        except Exception as e: print(f"[{c}] pipe kt: {e}")
        for base in ["dd_init", "gp0", "gp1", "gp2"]:
            unroll_all(root, base)
        for base in ["dd_a", "dpr0", "dpr1", "dpr2", "dn"]:
            unroll_partial(root, base, factor=HEAD_DIM)

    return s

def run_test_with_params(BATCH_SIZE, CONTEXT_LENGTH, HIDDEN_SIZE, NUM_HEADS, NROWS, NCOLS, BS, mode="csyn"):
    assert HIDDEN_SIZE % NUM_HEADS == 0
    assert CONTEXT_LENGTH % NROWS == 0
    assert CONTEXT_LENGTH % (BS * NCOLS) == 0

    print("=" * 60)
    print(
        f"split-K FlashAttn: BATCH={BATCH_SIZE}, SEQ_LEN={CONTEXT_LENGTH}, "
        f"HIDDEN={HIDDEN_SIZE}, HEADS={NUM_HEADS}, NROWS={NROWS}, NCOLS={NCOLS}, BS={BS}"
    )
    print("=" * 60)

    HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
    D_SQRT = HEAD_DIM**0.5
    OUT_ELEMS = BATCH_SIZE * CONTEXT_LENGTH * NUM_HEADS * HEAD_DIM

    A = np.random.rand(BATCH_SIZE * CONTEXT_LENGTH * 3 * HIDDEN_SIZE).astype(np.float16)
    A5 = A.reshape((BATCH_SIZE, CONTEXT_LENGTH, 3, NUM_HEADS, HEAD_DIM))
    Q = np.ascontiguousarray(A5[:, :, 0, :, :]).reshape(-1).astype(np.float16)   # [B,T,H,Dh]
    K = np.ascontiguousarray(A5[:, :, 1, :, :]).reshape(-1).astype(np.float16)
    V = np.ascontiguousarray(A5[:, :, 2, :, :]).reshape(-1).astype(np.float16)
    B_out = np.zeros(OUT_ELEMS, dtype=np.float16)

    # ---- golden reference (unchanged math) ----
    Q_np = A5[:, :, 0, :, :].transpose((0, 2, 1, 3))
    K_np = A5[:, :, 1, :, :].transpose((0, 2, 1, 3))
    V_np = A5[:, :, 2, :, :].transpose((0, 2, 1, 3))
    scores = np.matmul(Q_np, K_np.transpose((0, 1, 3, 2))) * (1.0 / D_SQRT)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    attn = np.exp(scores)
    attn = attn / np.sum(attn, axis=-1, keepdims=True)
    out_np = np.matmul(attn, V_np)
    B_golden = out_np.transpose((0, 2, 1, 3)).flatten()

    # ---- optional HLS C-synthesis ----
    if hls.is_available("vitis_hls"):
        print("Running Vitis HLS C-synthesis...")
        s = get_scheduled_systolic(
            BATCH_SIZE, CONTEXT_LENGTH, HIDDEN_SIZE, NUM_HEADS, NROWS, NCOLS, BS
        )
        #with tempfile.TemporaryDirectory() as sim_dir:
        hls_mod = s.build(target="vitis_hls", mode=mode, project="/scratch/ry375/restructure_head32_seq1024.prj")
        if mode != "csyn":
            hls_mod(Q, K, V, B_out)
            np.testing.assert_allclose(B_out, B_golden, rtol=0.02, atol=1e-2)
            print(f"✅ {mode} finished.")
        else:
            hls_mod()
            print("✅ csyn finished - read csynth.rpt")
    else:
        print("⚠️ Vitis HLS not available, skipping C synthesis.")


def test_splitk_tiny():
    # toy config: 4x4 grid, emulable in seconds — validate logic here first
    run_test_with_params(
        BATCH_SIZE=4, CONTEXT_LENGTH=4096, HIDDEN_SIZE=256, NUM_HEADS=4,
        NROWS=8, NCOLS=8, BS=8,
    )


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "128"

    # ── the ONE knob the kernel-coder flips each round ──────────────────────
    RUN = "hw"   # "perf": full csyn | "gate": small hw_emu | "func": tiny end-to-end hw_emu assert | "emu": full hw_emu | "hw": full bitstream
    # ────────────────────────────────────────────────────────────────────────

    REAL = dict(BATCH_SIZE=1, CONTEXT_LENGTH=1024, HIDDEN_SIZE=32, NUM_HEADS=1,
                NROWS=4, NCOLS=4, BS=4)
    GATE = dict(BATCH_SIZE=1, CONTEXT_LENGTH=128,  HIDDEN_SIZE=32, NUM_HEADS=4,
                NROWS=4, NCOLS=4, BS=2)      # small; keeps NUM_KV_TILES == COMBINE_LANES == 16
    FUNC = dict(BATCH_SIZE=1, CONTEXT_LENGTH=64,   HIDDEN_SIZE=32, NUM_HEADS=1,
                NROWS=4, NCOLS=4, BS=2)      # tiny: NUM_KV_TILES=8, COMBINE_LANES=4 -> 2 strips + full tree

    cfg, mode = {"gate": (GATE, "hw_emu"),   # small  hw_emu — structural deadlock
                 "func": (FUNC, "hw_emu"),   # tiny   hw_emu — end-to-end numerical assert
                 "emu":  (REAL, "hw_emu"),   # full   hw_emu — capacity deadlock
                 "hw":   (REAL, "hw"),       # full   hw     — place/route/timing
                 "perf": (REAL, "csyn")}[RUN]

    run_test_with_params(**cfg, mode=mode)

    del os.environ["OMP_NUM_THREADS"]