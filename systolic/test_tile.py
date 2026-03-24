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

BLOCK_T, HEAD_DIM= 4, 4
P0, P1 = HEAD_DIM, BLOCK_T
D = 1.0 / float(HEAD_DIM ** 0.5)

@df.region()
def compute_block_attention(
    Q: float32[BLOCK_T, HEAD_DIM],
    Max_in: float32[BLOCK_T],
    S_in: float32[BLOCK_T, HEAD_DIM],
    Sum_in: float32[BLOCK_T],
    K_m: float32[BLOCK_T, HEAD_DIM],
    V: float32[BLOCK_T, HEAD_DIM],
    S_out: float32[BLOCK_T, HEAD_DIM],
    Sum_out: float32[BLOCK_T],
    Max_out: float32[BLOCK_T],
    is_first_block: int32[1],
):
    L_Q: Stream[float32, 1024][P1]
    L_K: Stream[float32, 1024][P0]
    L_S: Stream[float32, 1024][P1]


    fifo_Q: Stream[float32, 1024][P0, P1]
    fifo_K: Stream[float32, 1024][P0, P1]
    fifo_S: Stream[float32, 1024][P0, P1]
    fifo_SD: Stream[float32, 1024][P0, P1]

#load Q matrix
    @df.kernel(mapping=[1], args=[Q, Max_in, S_in, Sum_in])
    def offchip_loadQ(local_Q: float32[BLOCK_T, HEAD_DIM], local_Max_in: float32[BLOCK_T], local_S_in: float32[BLOCK_T, HEAD_DIM], local_Sum_in: float32[BLOCK_T]):
        
        for i in range(BLOCK_T):
            val = local_Sum_in[i]
            if i == 0: L_Q[0].put(val)
            elif i == 1: L_Q[1].put(val)
            elif i == 2: L_Q[2].put(val)
            elif i == 3: L_Q[3].put(val)

        for i in range(BLOCK_T):
            val = local_Max_in[i]
            if i == 0: L_Q[0].put(val)
            elif i == 1: L_Q[1].put(val)
            elif i == 2: L_Q[2].put(val)
            elif i == 3: L_Q[3].put(val)

        for i, j in dsl.grid(BLOCK_T, HEAD_DIM):
            val = local_Q[i, j]
            if i == 0: L_Q[0].put(val)
            elif i == 1: L_Q[1].put(val)
            elif i == 2: L_Q[2].put(val)
            elif i == 3: L_Q[3].put(val)

        for i, j in dsl.grid(BLOCK_T, HEAD_DIM):
            val = local_S_in[i, j]
            if i == 0: L_Q[0].put(val)
            elif i == 1: L_Q[1].put(val)
            elif i == 2: L_Q[2].put(val)
            elif i == 3: L_Q[3].put(val)


#first load K and then V matrix
    @df.kernel(mapping=[1], args=[K_m, V])
    def offchip_loadK(local_K: float32[BLOCK_T, HEAD_DIM], local_V: float32[BLOCK_T, HEAD_DIM]):
        for i, j in dsl.grid(BLOCK_T, HEAD_DIM):
            val = local_K[i, j]
            if j == 0: L_K[0].put(val)
            elif j == 1: L_K[1].put(val)
            elif j == 2: L_K[2].put(val)
            elif j == 3: L_K[3].put(val)

        for m, n in dsl.grid(BLOCK_T, HEAD_DIM):
            val = local_V[m, n]
            if m == 0: L_K[3].put(val)
            elif m == 1: L_K[2].put(val)
            elif m == 2: L_K[1].put(val)
            elif m == 3: L_K[0].put(val)



#store the rowsum and output O matrix
    @df.kernel(mapping=[1], args=[S_out, Sum_out, Max_out])
    def offchip_storeS(local_S_out: float32[BLOCK_T, HEAD_DIM], local_Sum_out: float32[BLOCK_T], local_Max_out: float32[BLOCK_T]):
        
        for i in range(BLOCK_T):
            if i == 0: local_Max_out[0] = L_S[0].get()
            elif i == 1: local_Max_out[1] = L_S[1].get()
            elif i == 2: local_Max_out[2] = L_S[2].get()
            elif i == 3: local_Max_out[3] = L_S[3].get()

        for i in range(BLOCK_T):
            if i == 0: local_Sum_out[0] = L_S[0].get()
            elif i == 1: local_Sum_out[1] = L_S[1].get()
            elif i == 2: local_Sum_out[2] = L_S[2].get()
            elif i == 3: local_Sum_out[3] = L_S[3].get()

        for m, n in dsl.grid(BLOCK_T, HEAD_DIM):
            if m == 0: local_S_out[m, n] = L_S[0].get()
            elif m == 1: local_S_out[m, n] = L_S[1].get()
            elif m == 2: local_S_out[m, n] = L_S[2].get()
            elif m == 3: local_S_out[m, n] = L_S[3].get()



    @df.kernel(mapping=[P0, P1], args=[is_first_block])
    def gemm(is_first: int32[1]):
        i, j = df.get_pid()

        with allo.meta_if(j == 0 and i == P0-1):
            q: float32
            v: float32
            alpha: float32
            beta: float32
            new_m: float32
            pre_ss: float32 = L_Q[j].get()

            pre_m: float32 = L_Q[j].get()

            pre_s: float32
            #load Q matrix into systolic array, since it's weight stationary.
            for m in range(BLOCK_T):
                q = L_Q[j].get()
                if m != BLOCK_T-1:
                    fifo_Q[i-1, j].put(q)
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

            L_S[j].put(new_m)

            if is_first[0] == 1:
                alpha = 0.0
            else:
                alpha = allo.exp(pre_m - new_m)
            beta = allo.exp(max - new_m)

            #get the sum of its row and output it
            ss: float32 = 0.0
            ss = s + fifo_SD[i,j].get()
            ss = pre_ss * alpha + ss * beta

            L_S[j].put(ss)
            #input V matrix and compute its multiplication with P, the element will stream through systolic array
            for g in range(BLOCK_T):
                v = L_K[i].get()
                fifo_K[i,j+1].put(v)
                ss = fifo_SD[i,j].get()
                ss = ss + s * v
                pre_s = L_Q[j].get()
                ss = pre_s * alpha + ss * beta

                L_S[j].put(ss)


                
        with allo.meta_elif(i == P0-1 and j == P1-1):
            q: float32
            alpha: float32
            beta: float32
            new_m: float32
            pre_ss: float32 = L_Q[j].get()

            pre_m: float32 = L_Q[j].get()

            pre_s: float32
            #load Q matrix into systolic array, since it's weight stationary.
            for m in range(BLOCK_T):
                q = L_Q[j].get()
                if m != BLOCK_T-1:
                    fifo_Q[i-1, j].put(q)
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

            L_S[j].put(new_m)

            if is_first[0] == 1:
                alpha = 0.0
            else:
                alpha = allo.exp(pre_m - new_m)
            beta = allo.exp(max - new_m)

            #get the sum of its row and output it
            ss: float32 = 0.0
            ss = s + fifo_SD[i,j].get()
            ss = pre_ss * alpha + ss * beta

            L_S[j].put(ss)  
            #input V matrix and compute its multiplication with P, the element will stream through systolic array
            for g in range(BLOCK_T):
                v = fifo_K[i,j].get()
                ss = fifo_SD[i,j].get()
                ss = ss + s * v
                pre_s = L_Q[j].get()
                ss = pre_s * alpha + ss * beta

                L_S[j].put(ss)


        #bottom two    
        with allo.meta_elif(i == P0-1):
            q: float32
            alpha: float32
            beta: float32
            new_m: float32
            pre_ss: float32 = L_Q[j].get()

            pre_m: float32 = L_Q[j].get()

            pre_s: float32
            #load Q matrix into systolic array, since it's weight stationary.
            for m in range(BLOCK_T):
                q = L_Q[j].get()
                if m != BLOCK_T-1:
                    fifo_Q[i-1, j].put(q)
            
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

            L_S[j].put(new_m)

            if is_first[0] == 1:
                alpha = 0.0
            else:
                alpha = allo.exp(pre_m - new_m)
            beta = allo.exp(max - new_m)

            #get the sum of its row and output it
            ss: float32 = 0.0
            ss = s + fifo_SD[i,j].get()
            ss = pre_ss * alpha + ss * beta

            L_S[j].put(ss)
            #input V matrix and compute its multiplication with P, the element will stream through systolic array
            for g in range(BLOCK_T):
                v = fifo_K[i, j].get()
                fifo_K[i,j+1].put(v)
                ss = fifo_SD[i,j].get()
                ss = ss + s * v
                pre_s = L_Q[j].get()
                ss = pre_s * alpha + ss * beta

                L_S[j].put(ss)


        with allo.meta_elif(j == 0 and i == 0):
            q: float32 = 0
            s: float32 = 0
            v: float32
            ss: float32 = 0
            max: float32 = -128
            #load Q matrix into systolic array, since it's weight stationary.
            q = fifo_Q[i,j].get()
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
                v = L_K[i].get()
                fifo_K[i,j+1].put(v)
                ss = v * s
                fifo_SD[i+1,j].put(ss)


        with allo.meta_elif(j == P1-1 and i == 0):
            q: float32 = 0
            s: float32 = 0
            max: float32 = -128
            #load Q matrix into systolic array, since it's weight stationary.
            q = fifo_Q[i,j].get()
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
            max: float32 = -128
            #load Q matrix into systolic array, since it's weight stationary.
            q = fifo_Q[i,j].get()
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

            #load Q matrix into systolic array, since it's weight stationary.
            for m in range(i+1):
                q = fifo_Q[i,j].get()
                if i-m>0:
                    fifo_Q[i-1, j].put(q)
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
                v = L_K[i].get()
                fifo_K[i,j+1].put(v)
                ss = fifo_SD[i,j].get()
                ss = ss + v * s
                fifo_SD[i+1,j].put(ss)


        #right side two
        with allo.meta_elif(j == P1-1):
            q: float32 = 0
            s: float32 = 0

            #load Q matrix into systolic array, since it's weight stationary.
            for m in range(i+1):
                q = fifo_Q[i,j].get()
                if i-m>0:
                    fifo_Q[i-1, j].put(q)
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

            #load Q matrix into systolic array, since it's weight stationary.
            for m in range(i+1):
                q = fifo_Q[i,j].get()
                if i-m>0:
                    fifo_Q[i-1, j].put(q)
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


BATCH_SIZE: int = 4
CONTEXT_LENGTH: int = 16
HIDDEN_SIZE: int = 64
NUM_HEADS: int = 16
BLOCK_T: int = 4

HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
D_SQRT = float(HEAD_DIM ** 0.5)
THREE_H = 3 * HIDDEN_SIZE
IN_ELEMS = BATCH_SIZE * CONTEXT_LENGTH * THREE_H
OUT_ELEMS = BATCH_SIZE * CONTEXT_LENGTH * NUM_HEADS * HEAD_DIM

def load_tile(global_mem: float32[IN_ELEMS], local_tile: float32[BLOCK_T, HEAD_DIM], b: index, h: index, t_start: index, type_offset: int32):
    for t, d in allo.grid(BLOCK_T, HEAD_DIM):
        t_global: index = t_start + t
        global_c: index = type_offset * HIDDEN_SIZE + h * HEAD_DIM + d
        idx: index = b * (CONTEXT_LENGTH * THREE_H) + t_global * THREE_H + global_c
        local_tile[t, d] = global_mem[idx]

def store_tile(global_mem: float32[OUT_ELEMS], local_tile: float32[BLOCK_T, HEAD_DIM], b: index, h: index, t_start: index):
    for t, d in allo.grid(BLOCK_T, HEAD_DIM):
        t_global: index = t_start + t
        idx: index = (((b * CONTEXT_LENGTH + t_global) * NUM_HEADS + h) * HEAD_DIM + d)
        global_mem[idx] = local_tile[t, d]

def compute_engine(input_mem: float32[IN_ELEMS], output_mem: float32[OUT_ELEMS]):
    Q_sram: float32[BLOCK_T, HEAD_DIM]
    K_sram: float32[BLOCK_T, HEAD_DIM]
    V_sram: float32[BLOCK_T, HEAD_DIM]

    S_in: float32[BLOCK_T, HEAD_DIM]
    m_in: float32[BLOCK_T]
    l_in: float32[BLOCK_T]

    S_out: float32[BLOCK_T, HEAD_DIM]
    m_out: float32[BLOCK_T]
    l_out: float32[BLOCK_T]

    is_first_arr: int32[1]

    for b, h in allo.grid(BATCH_SIZE, NUM_HEADS):
        for tr in range(0, CONTEXT_LENGTH, BLOCK_T):
            load_tile(input_mem, Q_sram, b, h, tr, 0)
            for m in range(BLOCK_T):
                m_in[m] = -1e30
                l_in[m] = 0.0
                for n in range(HEAD_DIM):
                    S_in[m, n] = 0.0
                    
            for tc in range(0, CONTEXT_LENGTH, BLOCK_T):
                load_tile(input_mem, K_sram, b, h, tc, 1)
                load_tile(input_mem, V_sram, b, h, tc, 2)

                if tc == 0:
                    is_first_arr[0] = 1
                else:
                    is_first_arr[0] = 0
                compute_block_attention(
                    Q_sram, m_in, S_in, l_in,
                    K_sram, V_sram,
                    S_out, l_out, m_out, is_first_arr
                )

                for m in range(BLOCK_T):
                    m_in[m] = m_out[m]
                    l_in[m] = l_out[m]
                    for n in range(HEAD_DIM):
                        S_in[m,n] = S_out[m,n]
                
            for i in range(BLOCK_T):
                inv_l: float32 = 1.0 / (l_out[i] + 1e-9)
                for d in range(HEAD_DIM):
                    S_out[i, d] = S_out[i, d] * inv_l

            store_tile(output_mem, S_out, b, h, tr)

A = np.random.rand(IN_ELEMS).astype(np.float32)
B = np.zeros((OUT_ELEMS), dtype=np.float32)

s1 = allo.customize(load_tile)
s1.pipeline("d")
s1.pipeline("t")

s2 = allo.customize(store_tile)
s2.pipeline("d")
s2.pipeline("t")

s = allo.customize(compute_engine)

s.compose([s1, s2])

if hls.is_available("vitis_hls"):
    mod_csyn = s.build(
        target="vitis_hls", 
        mode="csyn", 
        project="flash_attention_csyn",
        wrap_io=True
    )
    mod_csyn()

    mod_hw = s.build(
        target="vitis_hls", 
        mode="hw_emu", 
        project="flash_attention_hw_emu",
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