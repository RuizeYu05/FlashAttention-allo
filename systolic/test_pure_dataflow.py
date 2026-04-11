#Full implementation of systolic flashattention
#Matrix multiplication is weight stationary

import tempfile

import pytest
import allo
import scipy.special
from allo.ir.types import int8, Stream, UInt, float32, int32, index
from allo.utils import get_np_struct_type
import allo.dataflow as df
import allo.backend.hls as hls
import allo.dsl as dsl
import numpy as np

BATCH_SIZE = 4
CONTEXT_LENGTH = 16
HIDDEN_SIZE = 64
NUM_HEADS = 16
BLOCK_T = 4

HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
P0 = HEAD_DIM
P1 = BLOCK_T
D_SQRT = float(HEAD_DIM ** 0.5)
D = 1.0 / float(D_SQRT)
THREE_H = 3 * HIDDEN_SIZE
IN_ELEMS = BATCH_SIZE * CONTEXT_LENGTH * THREE_H
OUT_ELEMS = BATCH_SIZE * CONTEXT_LENGTH * NUM_HEADS * HEAD_DIM


@df.region()
def top(
    input_mem: float32[IN_ELEMS],
    output_mem: float32[OUT_ELEMS],
):

    L_Q: Stream[float32, 1024][P1]
    L_K: Stream[float32, 1024][P0]
    L_M: Stream[float32, 1024][P1]
    L_S: Stream[float32, 1024][P1]
    L_V: Stream[float32, 1024][P0]
    L_acc: Stream[float32, 1024][P1]
    L_out: Stream[float32, 1024][P1]


    fifo_Q: Stream[float32, 1024][P0, P1]
    fifo_K: Stream[float32, 1024][P0, P1]
    fifo_S: Stream[float32, 1024][P0, P1]
    fifo_SD: Stream[float32, 1024][P0, P1]

    sync_token: Stream[int32, 16][1]

    @df.kernel(mapping=[1], args=[input_mem])
    def load_tile(input_d: float32[IN_ELEMS]):
        for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):
            for tr in range(0, CONTEXT_LENGTH, BLOCK_T):
                for t, d in allo.grid(BLOCK_T, HEAD_DIM):
                    t_global = tr + t
                    global_c = h * HEAD_DIM + d
                    idx = b * (CONTEXT_LENGTH * THREE_H) + t_global * THREE_H + global_c
                    val = input_d[idx]
                    if t == 0: L_Q[0].put(val)
                    elif t == 1: L_Q[1].put(val)
                    elif t == 2: L_Q[2].put(val)
                    elif t == 3: L_Q[3].put(val)
                for tc in range(0, CONTEXT_LENGTH, BLOCK_T):

                    for t, d in allo.grid(BLOCK_T, HEAD_DIM):
                        t_global = tc + t
                        global_c = 1 * HIDDEN_SIZE + h * HEAD_DIM + d
                        idx = b * (CONTEXT_LENGTH * THREE_H) + t_global * THREE_H + global_c
                        val = input_d[idx]
                        if d == 0: L_K[0].put(val)
                        elif d == 1: L_K[1].put(val)
                        elif d == 2: L_K[2].put(val)
                        elif d == 3: L_K[3].put(val)

                    
                    local_V_buf: float32[BLOCK_T, HEAD_DIM]
                    for t, d in allo.grid(BLOCK_T, HEAD_DIM):
                        t_global = tc + t
                        global_c = 2 * HIDDEN_SIZE + h * HEAD_DIM + d
                        idx = b * (CONTEXT_LENGTH * THREE_H) + t_global * THREE_H + global_c
                        local_V_buf[t, d] = input_d[idx]
                        
                    for t_rev in range(BLOCK_T):
                        for d in range(HEAD_DIM):
                            t = BLOCK_T - 1 - t_rev
                            val = local_V_buf[t, d]
                            if t == 3: L_V[0].put(val)
                            elif t == 2: L_V[1].put(val)
                            elif t == 1: L_V[2].put(val)
                            elif t == 0: L_V[3].put(val)


    @df.kernel(mapping=[P0, P1], args=[])
    def gemm():
        i, j = df.get_pid()

        with allo.meta_if(j == 0 and i == P0-1):
            q: float32
            v: float32
            alpha: float32
            beta: float32
            new_m: float32
            pre_ss: float32

            pre_m: float32

            pre_s: float32

            for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):
                for tr in range(0, CONTEXT_LENGTH, BLOCK_T):
                    pre_ss = 0.0
                    pre_m = -1e30
                    #load Q matrix into systolic array, since it's weight stationary.
                    for m in range(BLOCK_T):
                        q = L_Q[j].get()
                        if m != BLOCK_T-1:
                            fifo_Q[i-1, j].put(q)
                    for tc in range(0, CONTEXT_LENGTH, BLOCK_T):
                        #input K matrix and compute its multiplication with Q, the element will stream through systolic array
                        s: float32 = 0
                        for k in range(BLOCK_T):
                            a: float32 = L_K[i].get()
                            s = a * q
                            fifo_K[i, j+1].put(a)
                            fifo_S[i-1, j].put(s)

                        s = fifo_SD[i,j].get()
                        max = fifo_SD[i,j].get()
                        #minus row max to avoid overflow
                        s = allo.exp(s-max)

                        if max > pre_m:
                            new_m = max
                        else:
                            new_m = pre_m

                        if tc == 0:
                            alpha = 0.0
                        else:
                            alpha = allo.exp(pre_m - new_m)

                        pre_m = new_m

                        beta = allo.exp(max - new_m)

                        L_acc[j].put(alpha)
                        L_acc[j].put(beta)

                        #get the sum of its row and output it
                        ss: float32 = 0.0
                        ss = s + fifo_SD[i,j].get()
                        ss = pre_ss * alpha + ss * beta

                        pre_ss = ss
                        #input V matrix and compute its multiplication with P, the element will stream through systolic array
                        for g in range(BLOCK_T):
                            v = L_V[i].get()
                            fifo_K[i,j+1].put(v)
                            ss = fifo_SD[i,j].get()
                            ss = ss + s * v
                            L_acc[j].put(ss)


                    L_acc[j].put(pre_ss)
                    


                
        with allo.meta_elif(i == P0-1 and j == P1-1):
            q: float32
            v: float32
            alpha: float32
            beta: float32
            new_m: float32
            pre_ss: float32

            pre_m: float32

            pre_s: float32

            for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):
                for tr in range(0, CONTEXT_LENGTH, BLOCK_T):
                    pre_ss = 0.0
                    pre_m = -1e30
                    #load Q matrix into systolic array, since it's weight stationary.
                    for m in range(BLOCK_T):
                        q = L_Q[j].get()
                        if m != BLOCK_T-1:
                            fifo_Q[i-1, j].put(q)
                    for tc in range(0, CONTEXT_LENGTH, BLOCK_T):
                        #input K matrix and compute its multiplication with Q, the element will stream through systolic array
                        s: float32 = 0
                        a: float32 = 0
                        for n in range(BLOCK_T):
                            a = fifo_K[i, j].get()
                            s = a * q
                            fifo_S[i-1, j].put(s)
                        
                        s = fifo_SD[i,j].get()
                        max = fifo_SD[i,j].get()
                        #minus row max to avoid overflow
                        s = allo.exp(s-max)

                        if max > pre_m:
                            new_m = max
                        else:
                            new_m = pre_m

                        if tc == 0:
                            alpha = 0.0
                        else:
                            alpha = allo.exp(pre_m - new_m)

                        pre_m = new_m
                        beta = allo.exp(max - new_m)

                        L_acc[j].put(alpha)
                        L_acc[j].put(beta)

                        #get the sum of its row and output it
                        ss: float32 = 0.0
                        ss = s + fifo_SD[i,j].get()
                        ss = pre_ss * alpha + ss * beta

                        pre_ss = ss
                        #input V matrix and compute its multiplication with P, the element will stream through systolic array
                        for g in range(BLOCK_T):
                            v = fifo_K[i,j].get()
                            ss = fifo_SD[i,j].get()
                            ss = ss + s * v
                            L_acc[j].put(ss)

                    L_acc[j].put(pre_ss)



        #bottom two    
        with allo.meta_elif(i == P0-1):
            q: float32
            v: float32
            alpha: float32
            beta: float32
            new_m: float32
            pre_ss: float32

            pre_m: float32

            pre_s: float32

            for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):
                for tr in range(0, CONTEXT_LENGTH, BLOCK_T):
                    pre_ss = 0.0
                    pre_m = -1e30
                    #load Q matrix into systolic array, since it's weight stationary.
                    for m in range(BLOCK_T):
                        q = L_Q[j].get()
                        if m != BLOCK_T-1:
                            fifo_Q[i-1, j].put(q)

                    for tc in range(0, CONTEXT_LENGTH, BLOCK_T):
                        
                        s: float32 = 0
                        a: float32 = 0
                        #input K matrix and compute its multiplication with Q, the element will stream through systolic array
                        for n in range(BLOCK_T):
                            a = fifo_K[i, j].get()
                            s = a * q
                            fifo_K[i,j+1].put(a)
                            fifo_S[i-1, j].put(s)

                        s = fifo_SD[i,j].get()
                        max = fifo_SD[i,j].get()
                        #minus row max to avoid overflow
                        s = allo.exp(s-max)

                        if max > pre_m:
                            new_m = max
                        else:
                            new_m = pre_m

                        if tc == 0:
                            alpha = 0.0
                        else:
                            alpha = allo.exp(pre_m - new_m)

                        pre_m = new_m
                        beta = allo.exp(max - new_m)

                        L_acc[j].put(alpha)
                        L_acc[j].put(beta)

                        #get the sum of its row and output it
                        ss: float32 = 0.0
                        ss = s + fifo_SD[i,j].get()
                        ss = pre_ss * alpha + ss * beta

                        pre_ss = ss
                        #input V matrix and compute its multiplication with P, the element will stream through systolic array
                        for g in range(BLOCK_T):
                            v = fifo_K[i, j].get()
                            fifo_K[i,j+1].put(v)
                            ss = fifo_SD[i,j].get()
                            ss = ss + s * v

                            L_acc[j].put(ss)


                    L_acc[j].put(pre_ss)


        with allo.meta_elif(j == 0 and i == 0):
            q: float32 = 0
            s: float32 = 0
            v: float32
            ss: float32 = 0

            for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):
                for tr in range(0, CONTEXT_LENGTH, BLOCK_T):
                    #load Q matrix into systolic array, since it's weight stationary.
                    q = fifo_Q[i,j].get()
                    for tc in range(0, CONTEXT_LENGTH, BLOCK_T):
                        max: float32 = -1e30
                        #input K matrix and compute its multiplication with Q, the element will stream through systolic array
                        for m in range(BLOCK_T):
                            a = L_K[i].get()
                            fifo_K[i,j+1].put(a)
                            s = fifo_S[i,j].get()
                            s = s + a * q
                            s = s * D
                            #the top row will act as a comparator and compute the row's max value on the fly
                            if s > max:
                                max = s
                            if BLOCK_T-i-m-1>0:
                                fifo_SD[i+1,j].put(s)
                        #minus row max to avoid overflow
                        fifo_SD[i+1,j].put(max)
                        s = allo.exp(s-max)

                        fifo_SD[i+1,j].put(s)
                        #input V matrix and compute its multiplication with P, the element will stream through systolic array
                        for g in range(BLOCK_T):
                            v = L_V[i].get()
                            fifo_K[i,j+1].put(v)
                            ss = v * s
                            fifo_SD[i+1,j].put(ss)


        with allo.meta_elif(j == P1-1 and i == 0):
            q: float32 = 0
            s: float32 = 0
            v: float32

            for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):
                for tr in range(0, CONTEXT_LENGTH, BLOCK_T):
                    #load Q matrix into systolic array, since it's weight stationary.
                    q = fifo_Q[i,j].get()
                    for tc in range(0, CONTEXT_LENGTH, BLOCK_T):
                        max: float32 = -1e30
                        #input K matrix and compute its multiplication with Q, the element will stream through systolic array
                        for m in range(BLOCK_T):
                            a = fifo_K[i,j].get()
                            s = fifo_S[i,j].get()
                            s = s + a * q
                            s = s*D
                            #the top row will act as a comparator and compute the row's max value on the fly
                            if s > max:
                                max = s
                            if BLOCK_T-i-m-1>0:
                                fifo_SD[i+1,j].put(s) #keep one s
                        #minus row max to avoid overflow
                        fifo_SD[i+1,j].put(max)
                        s = allo.exp(s-max)
                        
                        fifo_SD[i+1,j].put(s)
                        #input V matrix and compute its multiplication with P, the element will stream through systolic array
                        for g in range(BLOCK_T):
                            v = fifo_K[i,j].get()
                            ss = v * s
                            fifo_SD[i+1,j].put(ss)
                
            

        #upside two
        with allo.meta_elif(i == 0):
            q: float32 = 0
            s: float32 = 0
            v: float32

            for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):
                for tr in range(0, CONTEXT_LENGTH, BLOCK_T):
                    #load Q matrix into systolic array, since it's weight stationary.
                    q = fifo_Q[i,j].get()
                    for tc in range(0, CONTEXT_LENGTH, BLOCK_T):
                        max: float32 = -1e30
                        #input K matrix and compute its multiplication with Q, the element will stream through systolic array
                        for m in range(BLOCK_T):
                            a = fifo_K[i,j].get()
                            fifo_K[i,j+1].put(a)
                            s = fifo_S[i,j].get()
                            s = s + a * q
                            s = s*D
                            #the top row will act as a comparator and compute the row's max value on the fly
                            if s > max:
                                max = s
                            if BLOCK_T-i-m-1>0:
                                fifo_SD[i+1,j].put(s)
                        #minus row max to avoid overflow
                        fifo_SD[i+1,j].put(max)
                        s = allo.exp(s-max)

                        fifo_SD[i+1,j].put(s)
                        #input V matrix and compute its multiplication with P, the element will stream through systolic array
                        for g in range(BLOCK_T):
                            v = fifo_K[i,j].get()
                            fifo_K[i,j+1].put(v)
                            ss = v * s
                            fifo_SD[i+1,j].put(ss)

        #left side 2
        with allo.meta_elif(j == 0):
            q: float32 = 0
            s: float32 = 0
            v: float32

            for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):
                for tr in range(0, CONTEXT_LENGTH, BLOCK_T):
                    #load Q matrix into systolic array, since it's weight stationary.
                    for m in range(i+1):
                        q = fifo_Q[i,j].get()
                        if i-m>0:
                            fifo_Q[i-1, j].put(q)
                    for tc in range(0, CONTEXT_LENGTH, BLOCK_T):
                        #input K matrix and compute its multiplication with Q, the element will stream through systolic array
                        for n in range(BLOCK_T):
                            a = L_K[i].get()
                            fifo_K[i,j+1].put(a)
                            s = fifo_S[i,j].get()

                            s = s+q*a
                            fifo_S[i-1, j].put(s)

                        for l in range(BLOCK_T):
                            if BLOCK_T-i-l>0:
                                s = fifo_SD[i,j].get()
                            if BLOCK_T-i-l-1>0:
                                fifo_SD[i+1, j].put(s)
                                
                        max = fifo_SD[i,j].get()
                        #minus row max to avoid overflow
                        fifo_SD[i+1,j].put(max)
                        s = allo.exp(s-max)

                        ss: float32 = 0.0
                        ss = s + fifo_SD[i,j].get()
                        fifo_SD[i+1,j].put(ss)
                        #input V matrix and compute its multiplication with P, the element will stream through systolic array
                        for g in range(BLOCK_T):
                            v = L_V[i].get()
                            fifo_K[i,j+1].put(v)
                            ss = fifo_SD[i,j].get()
                            ss = ss + v * s
                            fifo_SD[i+1,j].put(ss)


        #right side two
        with allo.meta_elif(j == P1-1):
            q: float32 = 0
            s: float32 = 0
            v: float32

            for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):
                for tr in range(0, CONTEXT_LENGTH, BLOCK_T):
                    #load Q matrix into systolic array, since it's weight stationary.
                    for m in range(i+1):
                        q = fifo_Q[i,j].get()
                        if i-m>0:
                            fifo_Q[i-1, j].put(q)
                    for tc in range(0, CONTEXT_LENGTH, BLOCK_T):
                        #input K matrix and compute its multiplication with Q, the element will stream through systolic array
                        for n in range(BLOCK_T):
                            a = fifo_K[i,j].get()
                            s = fifo_S[i,j].get()

                            s = s+q*a
                            fifo_S[i-1, j].put(s)

                        for l in range(BLOCK_T):
                            if BLOCK_T-i-l>0:
                                s = fifo_SD[i,j].get()
                            if BLOCK_T-i-l-1>0:
                                fifo_SD[i+1, j].put(s)
                        max = fifo_SD[i,j].get()
                        #minus row max to avoid overflow
                        fifo_SD[i+1,j].put(max)
                        s = allo.exp(s-max)

                        ss: float32 = 0.0
                        ss = s + fifo_SD[i,j].get()
                        fifo_SD[i+1,j].put(ss)
                        #input V matrix and compute its multiplication with P, the element will stream through systolic array
                        for g in range(BLOCK_T):
                            v = fifo_K[i,j].get()
                            ss = fifo_SD[i,j].get()
                            ss = ss + v * s
                            fifo_SD[i+1,j].put(ss)


        #middle four
        with allo.meta_else():
            q: float32 = 0
            s: float32 = 0
            v: float32

            for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):
                for tr in range(0, CONTEXT_LENGTH, BLOCK_T):
                    #load Q matrix into systolic array, since it's weight stationary.
                    for m in range(i+1):
                        q = fifo_Q[i,j].get()
                        if i-m>0:
                            fifo_Q[i-1, j].put(q)
                    for tc in range(0, CONTEXT_LENGTH, BLOCK_T):
                        #input K matrix and compute its multiplication with Q, the element will stream through systolic array
                        for n in range(BLOCK_T):
                            a = fifo_K[i,j].get()
                            fifo_K[i,j+1].put(a)
                            s = fifo_S[i,j].get()

                            s = s+q*a
                            fifo_S[i-1, j].put(s)

                        for l in range(BLOCK_T):
                            if BLOCK_T-i-l>0:
                                s = fifo_SD[i,j].get()
                            if BLOCK_T-i-l-1>0:
                                fifo_SD[i+1, j].put(s)

                        max = fifo_SD[i,j].get()
                        #minus row max to avoid overflow
                        fifo_SD[i+1,j].put(max)
                        s = allo.exp(s-max)

                        ss: float32 = 0.0
                        ss = s + fifo_SD[i,j].get()
                        fifo_SD[i+1,j].put(ss)
                        #input V matrix and compute its multiplication with P, the element will stream through systolic array
                        for g in range(BLOCK_T):
                            v = fifo_K[i,j].get()
                            fifo_K[i,j+1].put(v)
                            ss = fifo_SD[i,j].get()
                            ss = ss + v * s
                            fifo_SD[i+1,j].put(ss)
    
    @df.kernel(mapping=[P1], args=[])
    def accum():

        d = df.get_pid()
        for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):
            for tr in range(0, CONTEXT_LENGTH, BLOCK_T):
                o0: float32 = 0.0
                o1: float32 = 0.0
                o2: float32 = 0.0
                o3: float32 = 0.0

                for tc in range(0, CONTEXT_LENGTH, BLOCK_T):
                    alpha0: float32 = L_acc[d].get()
                    beta0: float32 = L_acc[d].get()
                    o0 = o0 * alpha0 + L_acc[d].get() * beta0
                    o1 = o1 * alpha0 + L_acc[d].get() * beta0
                    o2 = o2 * alpha0 + L_acc[d].get() * beta0
                    o3 = o3 * alpha0 + L_acc[d].get() * beta0

                row_sum0: float32 = L_acc[d].get()
                L_out[d].put(o0/row_sum0)
                L_out[d].put(o1/row_sum0)
                L_out[d].put(o2/row_sum0)
                L_out[d].put(o3/row_sum0)




    @df.kernel(mapping=[1], args=[output_mem])
    def store_tile(global_mem: float32[OUT_ELEMS]):
        for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):
            for tr in range(0, CONTEXT_LENGTH, BLOCK_T):
                for t, d in allo.grid(BLOCK_T, HEAD_DIM):
                    t_global = tr + t
                    idx = (((b*CONTEXT_LENGTH + t_global) * NUM_HEADS + h) * HEAD_DIM + d)
                    val: float32
                    if t == 0: val = L_out[0].get()
                    elif t == 1: val = L_out[1].get()
                    elif t == 2: val = L_out[2].get()
                    elif t == 3: val = L_out[3].get()
                    global_mem[idx] = val



def test_tiled_systolic():

    A = np.random.rand(IN_ELEMS).astype(np.float32)
    B = np.zeros((OUT_ELEMS), dtype=np.float32)

    if hls.is_available("vitis_hls"):

        with tempfile.TemporaryDirectory() as tmpdir:
            mod_csyn = df.build(
                top,
                target="vitis_hls", 
                mode="csyn", 
                project=tmpdir,
                wrap_io=True
            )
            mod_csyn()

        with tempfile.TemporaryDirectory() as tmpdir:
            mod_hw = df.build(
                top,
                target="vitis_hls", 
                mode="hw_emu", 
                project=tmpdir,
                wrap_io=True
            )
            mod_hw(A, B)

    A_reshaped = A.reshape((BATCH_SIZE, CONTEXT_LENGTH, 3, NUM_HEADS, HEAD_DIM))

    Q_np = A_reshaped[:, :, 0, :, :]
    K_np = A_reshaped[:, :, 1, :, :]
    V_np = A_reshaped[:, :, 2, :, :]

    Q_np = Q_np.transpose((0, 2, 1, 3))
    K_np = K_np.transpose((0, 2, 1, 3))
    V_np = V_np.transpose((0, 2, 1, 3))

    scores = np.matmul(Q_np, K_np.transpose((0, 1, 3, 2)))
    scores = scores * (1.0 / D_SQRT)
    attn_weights = scipy.special.softmax(scores, axis=-1)
    out_np = np.matmul(attn_weights, V_np)
    out_np_reshaped = out_np.transpose((0, 2, 1, 3)).flatten()

    np.testing.assert_allclose(B, out_np_reshaped, rtol=1e-4, atol=1e-4)
    print("Passed")


if __name__ == "__main__":
    test_tiled_systolic()