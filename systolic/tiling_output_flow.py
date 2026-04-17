# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
 
"""
Systolic Array Flash Attention — Loop-back Free
 Key architecture:
  - The tc loop is *absorbed into each compute PE's inner k-loop*
  - Each PE column j is statically assigned to exactly one tc block
      column 1 → K/V[ 0.. 3]
      column 2 → K/V[ 4.. 7]
      column 3 → K/V[ 8..11]
      column 4 → K/V[12..15]
  - State (m, d, o) flows strictly left → right with no feedback path
  - Constraint: NUM_TC == BLOCK_T (both equal 4 here)
 
Algorithm (same one-pass FlashAttention, per query row k):
 
  for i <- 1, N do
      x_i  <- Q[k,:] * K^T[:,i]
      m_i  <- max(m_{i-1}, x_i)
      d'_i <- d'_{i-1} * exp(m_{i-1} - m_i) + exp(x_i - m_i)
      o'_i <- o'_{i-1} * d'_{i-1} * exp(m_{i-1} - m_i) / d'_i
             + exp(x_i - m_i) / d'_i * V[i,:]
  end
  O[k,:] <- o'_N
 
Systolic array mapping (4x4 compute core + 1-wide border):
 
       j=0      j=1           j=2           j=3           j=4           j=5
  i=0 [corner] [K tc0,V tc0 ↓] [K tc1,V tc1 ↓] [K tc2,V tc2 ↓] [K tc3,V tc3 ↓] [corner]
  i=1 [Q0>]   [PE(0,0) k=0..3] [PE(0,1) k=0..3] [PE(0,2) k=0..3] [PE(0,3) k=0..3] [→O0]
  i=2 [Q1>]   [PE(1,0) k=0..3] [PE(1,1) k=0..3] [PE(1,2) k=0..3] [PE(1,3) k=0..3] [→O1]
  i=3 [Q2>]   [PE(2,0) k=0..3] [PE(2,1) k=0..3] [PE(2,2) k=0..3] [PE(2,3) k=0..3] [→O2]
  i=4 [Q3>]   [PE(3,0) k=0..3] [PE(3,1) k=0..3] [PE(3,2) k=0..3] [PE(3,3) k=0..3] [→O3]
  i=5 [corner] [drainK/V]      [drainK/V]        [drainK/V]        [drainK/V]        [corner]
 
Data flow (all unidirectional, no loop-back):
  Q[k,:]   : left  → right (horizontal), unchanged per tr
  K[j tc,:]: top   → bottom (vertical),  column j holds tc block j-1
  V[j tc,:]: top   → bottom (vertical),  column j holds tc block j-1
  state    : left  → right (horizontal), accumulated over k=0..BLOCK_T-1 inside PE
"""

import os
import tempfile
import scipy.special

import allo
import numpy as np
import pytest
from allo.ir.types import float32, Stream
import allo.dataflow as df
import allo.backend.hls as hls

BATCH_SIZE     = 4
NUM_HEADS      = 8
CONTEXT_LENGTH = 16
HIDDEN_SIZE    = 32
BLOCK_T        = 4          # PE-array dimension AND tc-block size
NUM_TC         = CONTEXT_LENGTH // BLOCK_T   # must equal BLOCK_T

assert NUM_TC == BLOCK_T, "This design requires NUM_TC == BLOCK_T"

HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
P0 = BLOCK_T + 2   # 6 rows  (1 border top + 4 compute + 1 border bottom)
P1 = BLOCK_T + 2   # 6 cols  (1 border left + 4 compute + 1 border right)
 
D_SQRT  = float(HEAD_DIM ** 0.5)
D       = 1.0 / D_SQRT
THREE_H = 3 * HIDDEN_SIZE
IN_ELEMS  = BATCH_SIZE * CONTEXT_LENGTH * THREE_H
OUT_ELEMS = BATCH_SIZE * CONTEXT_LENGTH * NUM_HEADS * HEAD_DIM
 
 
@df.region()
def top(
    input_mem:  float32[IN_ELEMS],
    output_mem: float32[OUT_ELEMS],
):
    # ── Intra-array streams ────────────────────────────────────────────────
    fifo_Q: Stream[float32[HEAD_DIM], 512][P0, P1]   # Q flows left → right
    fifo_K: Stream[float32[HEAD_DIM], 512][P0, P1]   # K flows top  → bottom
    fifo_V: Stream[float32[HEAD_DIM], 512][P0, P1]   # V flows top  → bottom
 
    # Flash Attention state (strictly left → right, no feedback)
    fifo_m: Stream[float32,          512][P0, P1]    # running max
    fifo_d: Stream[float32,          512][P0, P1]    # running normaliser
    fifo_o: Stream[float32[HEAD_DIM],512][P0, P1]    # running output
 
    # ── Boundary I/O streams ──────────────────────────────────────────────
    # One FIFO per row for Q (left boundary)
    fifo_in_Q: Stream[float32[HEAD_DIM], 512][P0 - 2]
    # One FIFO per column for K/V; column j+1 gets tc block j
    fifo_in_K: Stream[float32[HEAD_DIM], 512][P1 - 2]  
    fifo_in_V: Stream[float32[HEAD_DIM], 512][P1 - 2]
    # One FIFO per row for output (right boundary)
    fifo_out:  Stream[float32[HEAD_DIM], 512][P0 - 2]
 
 
    # ── Load kernel ───────────────────────────────────────────────────────
    @df.kernel(mapping=[1], args=[input_mem])
    def load(input_d: float32[IN_ELEMS]):
        """
        For each (batch, head, query-tile tr):
          1. Push Q[tr+0..tr+3] into fifo_in_Q[0..3].
          2. Push K and V for every tc block into the corresponding column
             FIFO: tc block tc_b → fifo_in_K[tc_b] / fifo_in_V[tc_b].
 
        Note: K and V are re-sent for each tr block because each query tile
        must attend over all CONTEXT_LENGTH key positions.
        """
        for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):
            for tr in range(0, CONTEXT_LENGTH, BLOCK_T):
 
                # ── Q (4 vectors, one per query row) ──────────────────────
                for i in range(BLOCK_T):
                    q_vec: float32[HEAD_DIM] = 0
                    t_q = tr + i
                    for jj in range(HEAD_DIM):
                        q_idx = (b * (CONTEXT_LENGTH * THREE_H)
                                 + t_q * THREE_H
                                 + 0 * HIDDEN_SIZE
                                 + h * HEAD_DIM + jj)
                        q_vec[jj] = input_d[q_idx]
                    if i == 0:
                        fifo_in_Q[0].put(q_vec)
                    elif i == 1:
                        fifo_in_Q[1].put(q_vec)
                    elif i == 2:
                        fifo_in_Q[2].put(q_vec)
                    elif i == 3:
                        fifo_in_Q[3].put(q_vec)
 
                # ── K, V (BLOCK_T vectors per tc block, 4 tc blocks) ──────
                # tc_b == 0 → fifo_in_K[0]  (top row of PE column 1)
                # tc_b == 1 → fifo_in_K[1]  (top row of PE column 2)
                # ...
                for tc_b in range(NUM_TC):
                    for i in range(BLOCK_T):
                        k_vec: float32[HEAD_DIM] = 0
                        v_vec: float32[HEAD_DIM] = 0
                        t_k = tc_b * BLOCK_T + i
                        for jj in range(HEAD_DIM):
                            k_idx = (b * (CONTEXT_LENGTH * THREE_H)
                                     + t_k * THREE_H
                                     + 1 * HIDDEN_SIZE
                                     + h * HEAD_DIM + jj)
                            v_idx = (b * (CONTEXT_LENGTH * THREE_H)
                                     + t_k * THREE_H
                                     + 2 * HIDDEN_SIZE
                                     + h * HEAD_DIM + jj)
                            k_vec[jj] = input_d[k_idx]
                            v_vec[jj] = input_d[v_idx]
                        if tc_b == 0:
                            fifo_in_K[0].put(k_vec)
                            fifo_in_V[0].put(v_vec)
                        elif tc_b == 1:
                            fifo_in_K[1].put(k_vec)
                            fifo_in_V[1].put(v_vec)
                        elif tc_b == 2:
                            fifo_in_K[2].put(k_vec)
                            fifo_in_V[2].put(v_vec)
                        elif tc_b == 3:
                            fifo_in_K[3].put(k_vec)
                            fifo_in_V[3].put(v_vec)
 
    # ── PE array kernel ───────────────────────────────────────────────────
    @df.kernel(mapping=[P0, P1], args=[])
    def pe():
        i, j = df.get_pid()
 
        # ── Corners: no-op ────────────────────────────────────────────────
        with allo.meta_if(i in {0, P0 - 1} and j in {0, P1 - 1}):
            pass
 
        # ── Top row (i=0, j=1..BLOCK_T): relay K/V for tc block j-1 ↓ ───
        # Receives BLOCK_T K/V vectors per (b, h, tr) from fifo_in_K[j-1]
        # and forwards them down to compute row 1.
        with allo.meta_elif(i == 0):
            for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):
                for tr in range(0, CONTEXT_LENGTH, BLOCK_T):
                    for k in range(BLOCK_T):
                        kv: float32[HEAD_DIM] = fifo_in_K[j - 1].get()
                        vv: float32[HEAD_DIM] = fifo_in_V[j - 1].get()
                        fifo_K[i + 1, j].put(kv)
                        fifo_V[i + 1, j].put(vv)
 
        # ── Left column (i=1..BLOCK_T, j=0): inject Q + initialise state ─
        # State is always initialised fresh: no feedback from right column.
        with allo.meta_elif(j == 0):
            for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):
                for tr in range(0, CONTEXT_LENGTH, BLOCK_T):
                    q_vec: float32[HEAD_DIM] = fifo_in_Q[i - 1].get()
                    fifo_Q[i, j + 1].put(q_vec)
                    m_init: float32 = -1e30
                    d_init: float32 = 0.0
                    o_init: float32[HEAD_DIM] = 0
                    fifo_m[i, j + 1].put(m_init)
                    fifo_d[i, j + 1].put(d_init)
                    fifo_o[i, j + 1].put(o_init)
 
        # ── Bottom row (i=P0-1, j=1..BLOCK_T): drain K/V ─────────────────
        with allo.meta_elif(i == P0 - 1):
            for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):
                for tr in range(0, CONTEXT_LENGTH, BLOCK_T):
                    for k in range(BLOCK_T):
                        _k: float32[HEAD_DIM] = fifo_K[i, j].get()
                        _v: float32[HEAD_DIM] = fifo_V[i, j].get()
 
        # ── Right column (i=1..BLOCK_T, j=P1-1): collect output ──────────
        # Drains Q and final state; writes o_final to fifo_out.
        with allo.meta_elif(j == P1 - 1):
            for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):
                for tr in range(0, CONTEXT_LENGTH, BLOCK_T):
                    _q: float32[HEAD_DIM] = fifo_Q[i, j].get()
                    _m: float32           = fifo_m[i, j].get()
                    _d: float32           = fifo_d[i, j].get()
                    o_final: float32[HEAD_DIM] = fifo_o[i, j].get()
                    fifo_out[i - 1].put(o_final)
 
        # ── Compute PEs (i=1..BLOCK_T, j=1..BLOCK_T) ─────────────────────
        # Each PE:
        #   1. Receives Q and state (m, d, o) from its left neighbour.
        #   2. Processes BLOCK_T K/V vectors streamed from above (the tc
        #      block assigned to this column), updating state after each.
        #   3. Forwards each K/V vector down immediately so the row below
        #      can start its own accumulation (systolic pipeline).
        #   4. Forwards Q and final state to the right neighbour.
        # The inner k-loop replaces the old outer tc-loop + loop-back path.
        with allo.meta_else():
            for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):
                for tr in range(0, CONTEXT_LENGTH, BLOCK_T):
 
                    # ── Receive Q and incoming state from left ─────────────
                    q_vec: float32[HEAD_DIM]  = fifo_Q[i, j].get()
                    m0:    float32            = fifo_m[i, j].get()
                    d0:    float32            = fifo_d[i, j].get()
                    o0:    float32[HEAD_DIM]  = fifo_o[i, j].get()
 
                    # ── k = 0 ─────────────────────────────────────────────
                    k0: float32[HEAD_DIM] = fifo_K[i, j].get()
                    v0: float32[HEAD_DIM] = fifo_V[i, j].get()
                    x0: float32 = 0.0
                    for dd in range(HEAD_DIM):
                        x0 += q_vec[dd] * k0[dd]
                    x0 = x0 * D
                    m1: float32 = m0
                    if x0 > m0:
                        m1 = x0
                    ep0: float32 = allo.exp(m0 - m1)
                    ex0: float32 = allo.exp(x0 - m1)
                    d1:  float32 = d0 * ep0 + ex0
                    al0: float32 = d0 * ep0 / d1
                    be0: float32 = ex0 / d1
                    o1: float32[HEAD_DIM] = 0
                    for dd in range(HEAD_DIM):
                        o1[dd] = o0[dd] * al0 + be0 * v0[dd]
                    fifo_K[i + 1, j].put(k0)
                    fifo_V[i + 1, j].put(v0)
 
                    # ── k = 1 ─────────────────────────────────────────────
                    k1: float32[HEAD_DIM] = fifo_K[i, j].get()
                    v1: float32[HEAD_DIM] = fifo_V[i, j].get()
                    x1: float32 = 0.0
                    for dd in range(HEAD_DIM):
                        x1 += q_vec[dd] * k1[dd]
                    x1 = x1 * D
                    m2: float32 = m1
                    if x1 > m1:
                        m2 = x1
                    ep1: float32 = allo.exp(m1 - m2)
                    ex1: float32 = allo.exp(x1 - m2)
                    d2:  float32 = d1 * ep1 + ex1
                    al1: float32 = d1 * ep1 / d2
                    be1: float32 = ex1 / d2
                    o2: float32[HEAD_DIM] = 0
                    for dd in range(HEAD_DIM):
                        o2[dd] = o1[dd] * al1 + be1 * v1[dd]
                    fifo_K[i + 1, j].put(k1)
                    fifo_V[i + 1, j].put(v1)
 
                    # ── k = 2 ─────────────────────────────────────────────
                    k2: float32[HEAD_DIM] = fifo_K[i, j].get()
                    v2: float32[HEAD_DIM] = fifo_V[i, j].get()
                    x2: float32 = 0.0
                    for dd in range(HEAD_DIM):
                        x2 += q_vec[dd] * k2[dd]
                    x2 = x2 * D
                    m3: float32 = m2
                    if x2 > m2:
                        m3 = x2
                    ep2: float32 = allo.exp(m2 - m3)
                    ex2: float32 = allo.exp(x2 - m3)
                    d3:  float32 = d2 * ep2 + ex2
                    al2: float32 = d2 * ep2 / d3
                    be2: float32 = ex2 / d3
                    o3: float32[HEAD_DIM] = 0
                    for dd in range(HEAD_DIM):
                        o3[dd] = o2[dd] * al2 + be2 * v2[dd]
                    fifo_K[i + 1, j].put(k2)
                    fifo_V[i + 1, j].put(v2)
 
                    # ── k = 3 ─────────────────────────────────────────────
                    k3: float32[HEAD_DIM] = fifo_K[i, j].get()
                    v3: float32[HEAD_DIM] = fifo_V[i, j].get()
                    x3: float32 = 0.0
                    for dd in range(HEAD_DIM):
                        x3 += q_vec[dd] * k3[dd]
                    x3 = x3 * D
                    m4: float32 = m3
                    if x3 > m3:
                        m4 = x3
                    ep3: float32 = allo.exp(m3 - m4)
                    ex3: float32 = allo.exp(x3 - m4)
                    d4:  float32 = d3 * ep3 + ex3
                    al3: float32 = d3 * ep3 / d4
                    be3: float32 = ex3 / d4
                    o4: float32[HEAD_DIM] = 0
                    for dd in range(HEAD_DIM):
                        o4[dd] = o3[dd] * al3 + be3 * v3[dd]
                    fifo_K[i + 1, j].put(k3)
                    fifo_V[i + 1, j].put(v3)

 
                    # ── Forward Q and final accumulated state right ────────
                    fifo_Q[i, j + 1].put(q_vec)
                    fifo_m[i, j + 1].put(m4)
                    fifo_d[i, j + 1].put(d4)
                    fifo_o[i, j + 1].put(o4)
 
 
    # ── Store kernel ──────────────────────────────────────────────────────
    @df.kernel(mapping=[1], args=[output_mem])
    def store(global_mem: float32[OUT_ELEMS]):
        for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):
            for tr in range(0, CONTEXT_LENGTH, BLOCK_T):
                for i in range(BLOCK_T):
                    if i == 0:
                        o0: float32[HEAD_DIM] = fifo_out[0].get()
                        for jj in range(HEAD_DIM):
                            idx = (((b * CONTEXT_LENGTH + (tr + i)) * NUM_HEADS + h) * HEAD_DIM + jj)
                            global_mem[idx] = o0[jj]
                    elif i == 1:
                        o1: float32[HEAD_DIM] = fifo_out[1].get()
                        for jj in range(HEAD_DIM):
                            idx = (((b * CONTEXT_LENGTH + (tr + i)) * NUM_HEADS + h) * HEAD_DIM + jj)
                            global_mem[idx] = o1[jj]
                    elif i == 2:
                        o2: float32[HEAD_DIM] = fifo_out[2].get()
                        for jj in range(HEAD_DIM):
                            idx = (((b * CONTEXT_LENGTH + (tr + i)) * NUM_HEADS + h) * HEAD_DIM + jj)
                            global_mem[idx] = o2[jj]
                    elif i == 3:
                        o3: float32[HEAD_DIM] = fifo_out[3].get()
                        for jj in range(HEAD_DIM):
                            idx = (((b * CONTEXT_LENGTH + (tr + i)) * NUM_HEADS + h) * HEAD_DIM + jj)
                            global_mem[idx] = o3[jj]
 
 
# ── Test ──────────────────────────────────────────────────────────────────────
def test_tiled_systolic():
    A = np.random.rand(IN_ELEMS).astype(np.float32)
    B = np.zeros(OUT_ELEMS, dtype=np.float32)
 
    if hls.is_available("vitis_hls"):
        scratch_dir = "/scratch/ry375"
        os.makedirs(scratch_dir, exist_ok=True)
 
        with tempfile.TemporaryDirectory(dir=scratch_dir) as tmpdir:
            mod_csyn = df.build(
                top,
                target="vitis_hls",
                mode="csyn",
                project=tmpdir,
                wrap_io=True,
            )
            mod_csyn()
 
        with tempfile.TemporaryDirectory(dir=scratch_dir) as tmpdir:
            mod_hw = df.build(
                top,
                target="vitis_hls",
                mode="hw",
                project=tmpdir,
                wrap_io=True,
            )
            mod_hw(A, B)
 
    # ── Reference (NumPy) ─────────────────────────────────────────────────
    A_reshaped  = A.reshape((BATCH_SIZE, CONTEXT_LENGTH, 3, NUM_HEADS, HEAD_DIM))
    Q_np        = A_reshaped[:, :, 0, :, :].transpose((0, 2, 1, 3))
    K_np        = A_reshaped[:, :, 1, :, :].transpose((0, 2, 1, 3))
    V_np        = A_reshaped[:, :, 2, :, :].transpose((0, 2, 1, 3))
 
    scores      = np.matmul(Q_np, K_np.transpose((0, 1, 3, 2))) * D
    attn_w      = scipy.special.softmax(scores, axis=-1)
    out_np      = np.matmul(attn_w, V_np)
    ref         = out_np.transpose((0, 2, 1, 3)).flatten()
 
    np.testing.assert_allclose(B, ref, rtol=0.02, atol=1e-4)
    print("✅ Passed!")
 
 
if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "128"
    test_tiled_systolic()
    del os.environ["OMP_NUM_THREADS"]
