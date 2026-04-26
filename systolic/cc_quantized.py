# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import scipy.special

import allo
import numpy as np
from allo.ir.types import float16, Int, Stream
import allo.dataflow as df
import allo.backend.hls as hls

int8  = Int(8)
int32 = Int(32)

BATCH_SIZE     = 4
NUM_HEADS      = 4
CONTEXT_LENGTH = 144
HIDDEN_SIZE    = 32
BLOCK_T        = 12
NUM_TC         = CONTEXT_LENGTH // BLOCK_T

assert NUM_TC == BLOCK_T, "This design requires NUM_TC == BLOCK_T"

HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
P0 = BLOCK_T + 2
P1 = BLOCK_T + 2

D_SQRT  = float(HEAD_DIM ** 0.5)
D       = 1.0 / D_SQRT
THREE_H = 3 * HIDDEN_SIZE
IN_ELEMS  = BATCH_SIZE * CONTEXT_LENGTH * THREE_H
OUT_ELEMS = BATCH_SIZE * CONTEXT_LENGTH * NUM_HEADS * HEAD_DIM


@df.region()
def top(
    input_mem:  float16[IN_ELEMS],
    output_mem: float16[OUT_ELEMS],
):
    fifo_Q:  Stream[int8[HEAD_DIM],    256][P0, P1]
    fifo_K:  Stream[int8[HEAD_DIM],    256][P0, P1]
    fifo_V:  Stream[float16[HEAD_DIM], 256][P0, P1]
    fifo_dQ: Stream[float16,            32][P0, P1]
    fifo_dK: Stream[float16,            32][P0, P1]
    fifo_m:  Stream[float16,            32][P0, P1]
    fifo_d:  Stream[float16,            32][P0, P1]
    fifo_o:  Stream[float16[HEAD_DIM], 256][P0, P1]

    fifo_in_Q:  Stream[int8[HEAD_DIM],    256][P0 - 2]
    fifo_in_dQ: Stream[float16,            32][P0 - 2]
    fifo_in_K:  Stream[int8[HEAD_DIM],    256][P1 - 2]
    fifo_in_dK: Stream[float16,            32][P1 - 2]
    fifo_in_V:  Stream[float16[HEAD_DIM], 256][P1 - 2]
    fifo_out:   Stream[float16[HEAD_DIM], 256][P0 - 2]


    @df.kernel(mapping=[1], args=[input_mem])
    def load(input_d: float16[IN_ELEMS]):

        for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):

            # ── Pre-pass: compute mean(K) ──────────────────────────────
            k_mean: float16[HEAD_DIM] = 0
            for t in range(CONTEXT_LENGTH):
                for jj in range(HEAD_DIM):
                    k_idx = (b * (CONTEXT_LENGTH * THREE_H)
                             + t * THREE_H
                             + 1 * HIDDEN_SIZE
                             + h * HEAD_DIM + jj)
                    k_mean[jj] += input_d[k_idx]
            for jj in range(HEAD_DIM):
                # annotation drives implicit float16/int division
                inv_ctx: float16 = 1.0 / CONTEXT_LENGTH
                k_mean[jj] = k_mean[jj] * inv_ctx

            for tr in range(0, CONTEXT_LENGTH, BLOCK_T):

                # ── Q: fold 1/√d, per-vector INT8 quantization ────────
                for i in range(BLOCK_T):
                    q_scaled:  float16[HEAD_DIM] = 0
                    q_abs_max: float16 = 1e-8
                    t_q = tr + i
                    for jj in range(HEAD_DIM):
                        q_idx = (b * (CONTEXT_LENGTH * THREE_H)
                                 + t_q * THREE_H
                                 + 0 * HIDDEN_SIZE
                                 + h * HEAD_DIM + jj)
                        d_f16: float16 = D          # typed literal
                        val: float16 = input_d[q_idx] * d_f16
                        q_scaled[jj] = val
                        neg_val: float16 = -val
                        abs_val: float16 = val if val >= 0.0 else neg_val
                        if abs_val > q_abs_max:
                            q_abs_max = abs_val

                    inv127: float16 = 1.0 / 127.0
                    delta_Q: float16 = q_abs_max * inv127

                    # ── implicit float16→int8 via typed target ─────────
                    q_int8: int8[HEAD_DIM] = 0
                    for jj in range(HEAD_DIM):
                        # Allo sees target is int8, inserts cast from float16
                        q_int8[jj] = q_scaled[jj] / delta_Q

                    if i == 0:
                        fifo_in_Q[0].put(q_int8)
                        fifo_in_dQ[0].put(delta_Q)
                    elif i == 1:
                        fifo_in_Q[1].put(q_int8)
                        fifo_in_dQ[1].put(delta_Q)
                    elif i == 2:
                        fifo_in_Q[2].put(q_int8)
                        fifo_in_dQ[2].put(delta_Q)
                    elif i == 3:
                        fifo_in_Q[3].put(q_int8)
                        fifo_in_dQ[3].put(delta_Q)
                    elif i == 4:
                        fifo_in_Q[4].put(q_int8)
                        fifo_in_dQ[4].put(delta_Q)
                    elif i == 5:
                        fifo_in_Q[5].put(q_int8)
                        fifo_in_dQ[5].put(delta_Q)
                    elif i == 6:
                        fifo_in_Q[6].put(q_int8)
                        fifo_in_dQ[6].put(delta_Q)
                    elif i == 7:
                        fifo_in_Q[7].put(q_int8)
                        fifo_in_dQ[7].put(delta_Q)
                    elif i == 8:
                        fifo_in_Q[8].put(q_int8)
                        fifo_in_dQ[8].put(delta_Q)
                    elif i == 9:
                        fifo_in_Q[9].put(q_int8)
                        fifo_in_dQ[9].put(delta_Q)
                    elif i == 10:
                        fifo_in_Q[10].put(q_int8)
                        fifo_in_dQ[10].put(delta_Q)
                    elif i == 11:
                        fifo_in_Q[11].put(q_int8)
                        fifo_in_dQ[11].put(delta_Q)

                # ── K: smooth + per-block INT8 quantization ───────────
                for tc_b in range(NUM_TC):

                    k_block:   float16[BLOCK_T, HEAD_DIM] = 0
                    k_abs_max: float16 = 1e-8

                    for i in range(BLOCK_T):
                        t_k = tc_b * BLOCK_T + i
                        for jj in range(HEAD_DIM):
                            k_idx = (b * (CONTEXT_LENGTH * THREE_H)
                                     + t_k * THREE_H
                                     + 1 * HIDDEN_SIZE
                                     + h * HEAD_DIM + jj)
                            val: float16 = input_d[k_idx] - k_mean[jj]
                            k_block[i, jj] = val
                            neg_val: float16 = -val
                            abs_val: float16 = val if val >= 0.0 else neg_val
                            if abs_val > k_abs_max:
                                k_abs_max = abs_val

                    inv127k: float16 = 1.0 / 127.0
                    delta_K: float16 = k_abs_max * inv127k

                    if tc_b == 0:
                        fifo_in_dK[0].put(delta_K)
                    elif tc_b == 1:
                        fifo_in_dK[1].put(delta_K)
                    elif tc_b == 2:
                        fifo_in_dK[2].put(delta_K)
                    elif tc_b == 3:
                        fifo_in_dK[3].put(delta_K)
                    elif tc_b == 4:
                        fifo_in_dK[4].put(delta_K)
                    elif tc_b == 5:
                        fifo_in_dK[5].put(delta_K)
                    elif tc_b == 6:
                        fifo_in_dK[6].put(delta_K)
                    elif tc_b == 7:
                        fifo_in_dK[7].put(delta_K)
                    elif tc_b == 8:
                        fifo_in_dK[8].put(delta_K)
                    elif tc_b == 9:
                        fifo_in_dK[9].put(delta_K)
                    elif tc_b == 10:
                        fifo_in_dK[10].put(delta_K)
                    elif tc_b == 11:
                        fifo_in_dK[11].put(delta_K)

                    for i in range(BLOCK_T):
                        # ── implicit float16→int8 via typed target ─────
                        k_int8: int8[HEAD_DIM] = 0
                        for jj in range(HEAD_DIM):
                            k_int8[jj] = k_block[i, jj] / delta_K

                        v_vec: float16[HEAD_DIM] = 0
                        t_k = tc_b * BLOCK_T + i
                        for jj in range(HEAD_DIM):
                            v_idx = (b * (CONTEXT_LENGTH * THREE_H)
                                     + t_k * THREE_H
                                     + 2 * HIDDEN_SIZE
                                     + h * HEAD_DIM + jj)
                            v_vec[jj] = input_d[v_idx]

                        if tc_b == 0:
                            fifo_in_K[0].put(k_int8)
                            fifo_in_V[0].put(v_vec)
                        elif tc_b == 1:
                            fifo_in_K[1].put(k_int8)
                            fifo_in_V[1].put(v_vec)
                        elif tc_b == 2:
                            fifo_in_K[2].put(k_int8)
                            fifo_in_V[2].put(v_vec)
                        elif tc_b == 3:
                            fifo_in_K[3].put(k_int8)
                            fifo_in_V[3].put(v_vec)
                        elif tc_b == 4:
                            fifo_in_K[4].put(k_int8)
                            fifo_in_V[4].put(v_vec)
                        elif tc_b == 5:
                            fifo_in_K[5].put(k_int8)
                            fifo_in_V[5].put(v_vec)
                        elif tc_b == 6:
                            fifo_in_K[6].put(k_int8)
                            fifo_in_V[6].put(v_vec)
                        elif tc_b == 7:
                            fifo_in_K[7].put(k_int8)
                            fifo_in_V[7].put(v_vec)
                        elif tc_b == 8:
                            fifo_in_K[8].put(k_int8)
                            fifo_in_V[8].put(v_vec)
                        elif tc_b == 9:
                            fifo_in_K[9].put(k_int8)
                            fifo_in_V[9].put(v_vec)
                        elif tc_b == 10:
                            fifo_in_K[10].put(k_int8)
                            fifo_in_V[10].put(v_vec)
                        elif tc_b == 11:
                            fifo_in_K[11].put(k_int8)
                            fifo_in_V[11].put(v_vec)

    @df.kernel(mapping=[P0, P1], args=[])
    def pe():
        i, j = df.get_pid()

        with allo.meta_if(i in {0, P0 - 1} and j in {0, P1 - 1}):
            pass

        with allo.meta_elif(i == 0):
            for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):
                for tr in range(0, CONTEXT_LENGTH, BLOCK_T):
                    dK: float16 = fifo_in_dK[j - 1].get()
                    fifo_dK[i + 1, j].put(dK)
                    for k in range(BLOCK_T):
                        kv: int8[HEAD_DIM]    = fifo_in_K[j - 1].get()
                        vv: float16[HEAD_DIM] = fifo_in_V[j - 1].get()
                        fifo_K[i + 1, j].put(kv)
                        fifo_V[i + 1, j].put(vv)

        with allo.meta_elif(j == 0):
            for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):
                for tr in range(0, CONTEXT_LENGTH, BLOCK_T):
                    q_int8: int8[HEAD_DIM] = fifo_in_Q[i - 1].get()
                    dQ:     float16        = fifo_in_dQ[i - 1].get()
                    fifo_Q[i,  j + 1].put(q_int8)
                    fifo_dQ[i, j + 1].put(dQ)
                    m_init: float16           = -1e4    # float16 safe large neg
                    d_init: float16           = 0.0
                    o_init: float16[HEAD_DIM] = 0
                    fifo_m[i, j + 1].put(m_init)
                    fifo_d[i, j + 1].put(d_init)
                    fifo_o[i, j + 1].put(o_init)

        with allo.meta_elif(i == P0 - 1):
            for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):
                for tr in range(0, CONTEXT_LENGTH, BLOCK_T):
                    _dk: float16          = fifo_dK[i, j].get()
                    for k in range(BLOCK_T):
                        _k: int8[HEAD_DIM]    = fifo_K[i, j].get()
                        _v: float16[HEAD_DIM] = fifo_V[i, j].get()

        with allo.meta_elif(j == P1 - 1):
            for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):
                for tr in range(0, CONTEXT_LENGTH, BLOCK_T):
                    _q:  int8[HEAD_DIM]        = fifo_Q[i,  j].get()
                    _dq: float16               = fifo_dQ[i, j].get()
                    _m:  float16               = fifo_m[i,  j].get()
                    _d:  float16               = fifo_d[i,  j].get()
                    o_final: float16[HEAD_DIM] = fifo_o[i,  j].get()
                    fifo_out[i - 1].put(o_final)

        with allo.meta_else():
            for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):
                for tr in range(0, CONTEXT_LENGTH, BLOCK_T):

                    q_int8: int8[HEAD_DIM]    = fifo_Q[i,  j].get()
                    dQ:     float16           = fifo_dQ[i, j].get()
                    m_cur:  float16           = fifo_m[i,  j].get()
                    d_cur:  float16           = fifo_d[i,  j].get()
                    o_cur:  float16[HEAD_DIM] = fifo_o[i,  j].get()

                    dK: float16 = fifo_dK[i, j].get()
                    fifo_dK[i + 1, j].put(dK)

                    for k in range(BLOCK_T):
                        k_int8: int8[HEAD_DIM]    = fifo_K[i, j].get()
                        vv:     float16[HEAD_DIM] = fifo_V[i, j].get()

                        # ── INT8 × INT8 → INT32 accumulator ───────────
                        # Declare int32 intermediates — Allo auto-widens int8
                        x_int32: int32 = 0
                        for dd in range(HEAD_DIM):
                            qi: int32 = q_int8[dd]   # int8 → int32 via annotation
                            ki: int32 = k_int8[dd]   # int8 → int32 via annotation
                            x_int32 += qi * ki

                        # ── Dequantize INT32 → float16 via annotation ──
                        x_f16: float16 = x_int32     # int32 → float16 via annotation
                        x: float16 = x_f16 * dQ * dK

                        # ── Online softmax — all float16 ───────────────
                        m_new: float16 = m_cur
                        if x > m_cur:
                            m_new = x
                        ep: float16 = allo.exp(m_cur - m_new)
                        ex: float16 = allo.exp(x - m_new)
                        d_new: float16 = d_cur * ep + ex
                        al: float16    = d_cur * ep / d_new
                        be: float16    = ex / d_new

                        # ── P·V: float16 × float16 ─────────────────────
                        o_new: float16[HEAD_DIM] = 0
                        for dd in range(HEAD_DIM):
                            o_new[dd] = o_cur[dd] * al + be * vv[dd]

                        fifo_K[i + 1, j].put(k_int8)
                        fifo_V[i + 1, j].put(vv)

                        m_cur = m_new
                        d_cur = d_new
                        for dd in range(HEAD_DIM):
                            o_cur[dd] = o_new[dd]

                    fifo_Q[i,  j + 1].put(q_int8)
                    fifo_dQ[i, j + 1].put(dQ)
                    fifo_m[i,  j + 1].put(m_cur)
                    fifo_d[i,  j + 1].put(d_cur)
                    fifo_o[i,  j + 1].put(o_cur)


    @df.kernel(mapping=[1], args=[output_mem])
    def store(global_mem: float16[OUT_ELEMS]):
        for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):
            for tr in range(0, CONTEXT_LENGTH, BLOCK_T):
                for i in range(BLOCK_T):
                    if i == 0:
                        o0: float16[HEAD_DIM] = fifo_out[0].get()
                        for jj in range(HEAD_DIM):
                            idx = (((b * CONTEXT_LENGTH + (tr + i)) * NUM_HEADS + h) * HEAD_DIM + jj)
                            global_mem[idx] = o0[jj]
                    elif i == 1:
                        o1: float16[HEAD_DIM] = fifo_out[1].get()
                        for jj in range(HEAD_DIM):
                            idx = (((b * CONTEXT_LENGTH + (tr + i)) * NUM_HEADS + h) * HEAD_DIM + jj)
                            global_mem[idx] = o1[jj]
                    elif i == 2:
                        o2: float16[HEAD_DIM] = fifo_out[2].get()
                        for jj in range(HEAD_DIM):
                            idx = (((b * CONTEXT_LENGTH + (tr + i)) * NUM_HEADS + h) * HEAD_DIM + jj)
                            global_mem[idx] = o2[jj]
                    elif i == 3:
                        o3: float16[HEAD_DIM] = fifo_out[3].get()
                        for jj in range(HEAD_DIM):
                            idx = (((b * CONTEXT_LENGTH + (tr + i)) * NUM_HEADS + h) * HEAD_DIM + jj)
                            global_mem[idx] = o3[jj]
                    elif i == 4:
                        o4: float16[HEAD_DIM] = fifo_out[4].get()
                        for jj in range(HEAD_DIM):
                            idx = (((b * CONTEXT_LENGTH + (tr + i)) * NUM_HEADS + h) * HEAD_DIM + jj)
                            global_mem[idx] = o4[jj]
                    elif i == 5:
                        o5: float16[HEAD_DIM] = fifo_out[5].get()
                        for jj in range(HEAD_DIM):
                            idx = (((b * CONTEXT_LENGTH + (tr + i)) * NUM_HEADS + h) * HEAD_DIM + jj)
                            global_mem[idx] = o5[jj]
                    elif i == 6:
                        o6: float16[HEAD_DIM] = fifo_out[6].get()
                        for jj in range(HEAD_DIM):
                            idx = (((b * CONTEXT_LENGTH + (tr + i)) * NUM_HEADS + h) * HEAD_DIM + jj)
                            global_mem[idx] = o6[jj]
                    elif i == 7:
                        o7: float16[HEAD_DIM] = fifo_out[7].get()
                        for jj in range(HEAD_DIM):
                            idx = (((b * CONTEXT_LENGTH + (tr + i)) * NUM_HEADS + h) * HEAD_DIM + jj)
                            global_mem[idx] = o7[jj]
                    elif i == 8:
                        o8: float16[HEAD_DIM] = fifo_out[8].get()
                        for jj in range(HEAD_DIM):
                            idx = (((b * CONTEXT_LENGTH + (tr + i)) * NUM_HEADS + h) * HEAD_DIM + jj)
                            global_mem[idx] = o8[jj]
                    elif i == 9:
                        o9: float16[HEAD_DIM] = fifo_out[9].get()
                        for jj in range(HEAD_DIM):
                            idx = (((b * CONTEXT_LENGTH + (tr + i)) * NUM_HEADS + h) * HEAD_DIM + jj)
                            global_mem[idx] = o9[jj]
                    elif i == 10:
                        o10: float16[HEAD_DIM] = fifo_out[10].get()
                        for jj in range(HEAD_DIM):
                            idx = (((b * CONTEXT_LENGTH + (tr + i)) * NUM_HEADS + h) * HEAD_DIM + jj)
                            global_mem[idx] = o10[jj]
                    elif i == 11:
                        o11: float16[HEAD_DIM] = fifo_out[11].get()
                        for jj in range(HEAD_DIM):
                            idx = (((b * CONTEXT_LENGTH + (tr + i)) * NUM_HEADS + h) * HEAD_DIM + jj)
                            global_mem[idx] = o11[jj]


# ── Test ──────────────────────────────────────────────────────────────────────
def test_tiled_systolic():
    A = np.random.rand(IN_ELEMS).astype(np.float16)
    B = np.zeros(OUT_ELEMS, dtype=np.float16)

    if hls.is_available("vitis_hls"):
        scratch_dir = "/scratch/ry375"
        os.makedirs(scratch_dir, exist_ok=True)

        with tempfile.TemporaryDirectory(dir=scratch_dir) as tmpdir:
            mod_csyn = df.build(
                top, target="vitis_hls", mode="csyn",
                project=tmpdir, wrap_io=True,
            )
            mod_csyn()

        with tempfile.TemporaryDirectory(dir=scratch_dir) as tmpdir:
            mod_hw = df.build(
                top, target="vitis_hls", mode="hw",
                project=tmpdir, wrap_io=True,
            )
            mod_hw(A, B)

    A_reshaped = A.reshape((BATCH_SIZE, CONTEXT_LENGTH, 3, NUM_HEADS, HEAD_DIM))
    Q_np = A_reshaped[:, :, 0, :, :].transpose((0, 2, 1, 3))
    K_np = A_reshaped[:, :, 1, :, :].transpose((0, 2, 1, 3))
    V_np = A_reshaped[:, :, 2, :, :].transpose((0, 2, 1, 3))

    scores = np.matmul(Q_np, K_np.transpose((0, 1, 3, 2))) * D
    attn_w = scipy.special.softmax(scores.astype(np.float32), axis=-1).astype(np.float16)
    out_np = np.matmul(attn_w, V_np)
    ref    = out_np.transpose((0, 2, 1, 3)).flatten()

    np.testing.assert_allclose(B, ref, rtol=0.05, atol=1e-2)
    print("✅ Passed!")


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "128"
    test_tiled_systolic()
    del os.environ["OMP_NUM_THREADS"]