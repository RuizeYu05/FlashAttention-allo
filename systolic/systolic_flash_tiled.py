#Full implementation of systolic flashattention
#Matrix multiplication is weight stationary

import tempfile

import pytest
import allo
from allo.ir.types import int8, Stream, UInt, float32
from allo.utils import get_np_struct_type
import allo.dataflow as df
import allo.backend.hls as hls
import allo.dsl as dsl
import numpy as np

BLOCK_T, HEAD_DIM= 16, 16
P0, P1 = HEAD_DIM, BLOCK_T
D = 1.0 / float(HEAD_DIM ** 0.5)

@df.region()
def top(
    Q: float32[BLOCK_T, HEAD_DIM],
    K_m: float32[BLOCK_T, HEAD_DIM],
    V: float32[BLOCK_T, HEAD_DIM],
    S: float32[BLOCK_T, HEAD_DIM],
    Sum: float32[BLOCK_T],
    Max: float32[BLOCK_T],
    is_first_block: bool,
):
    L_Q: Stream[float32, 1024][P1]
    L_K: Stream[float32, 1024][P0]
    L_M: Stream[float32, 1024][P1]
    L_S: Stream[float32, 1024][P1]


    fifo_Q: Stream[float32, 1024][P0, P1]
    fifo_K: Stream[float32, 1024][P0, P1]
    fifo_S: Stream[float32, 1024][P0, P1]
    fifo_SD: Stream[float32, 1024][P0, P1]

#load Q matrix
    @df.kernel(mapping=[1], args=[Q, Max, S])
    def offchip_loadQ(local_Q: float32[BLOCK_T, HEAD_DIM], local_Max: float32[BLOCK_T], local_S: float32[BLOCK_T, HEAD_DIM]):

        for i in range(BLOCK_T):
            val = local_Max[i]
            if i == 0: L_Q[0].put(val)
            elif i == 1: L_Q[1].put(val)
            elif i == 2: L_Q[2].put(val)
            elif i == 3: L_Q[3].put(val)
            elif i == 4: L_Q[4].put(val)
            elif i == 5: L_Q[5].put(val)
            elif i == 6: L_Q[6].put(val)
            elif i == 7: L_Q[7].put(val)
            elif i == 8: L_Q[8].put(val)
            elif i == 9: L_Q[9].put(val)
            elif i == 10: L_Q[10].put(val)
            elif i == 11: L_Q[11].put(val)
            elif i == 12: L_Q[12].put(val)
            elif i == 13: L_Q[13].put(val)
            elif i == 14: L_Q[14].put(val)
            elif i == 15: L_Q[15].put(val)

        for i, j in dsl.grid(BLOCK_T, HEAD_DIM):
            val = local_S[i, j]
            if i == 0: L_Q[0].put(val)
            elif i == 1: L_Q[1].put(val)
            elif i == 2: L_Q[2].put(val)
            elif i == 3: L_Q[3].put(val)
            elif i == 4: L_Q[4].put(val)
            elif i == 5: L_Q[5].put(val)
            elif i == 6: L_Q[6].put(val)
            elif i == 7: L_Q[7].put(val)
            elif i == 8: L_Q[8].put(val)
            elif i == 9: L_Q[9].put(val)
            elif i == 10: L_Q[10].put(val)
            elif i == 11: L_Q[11].put(val)
            elif i == 12: L_Q[12].put(val)
            elif i == 13: L_Q[13].put(val)
            elif i == 14: L_Q[14].put(val)
            elif i == 15: L_Q[15].put(val)


        for i, j in dsl.grid(BLOCK_T, HEAD_DIM):
            val = local_Q[i, j]
            if i == 0: L_Q[0].put(val)
            elif i == 1: L_Q[1].put(val)
            elif i == 2: L_Q[2].put(val)
            elif i == 3: L_Q[3].put(val)
            elif i == 4: L_Q[4].put(val)
            elif i == 5: L_Q[5].put(val)
            elif i == 6: L_Q[6].put(val)
            elif i == 7: L_Q[7].put(val)
            elif i == 8: L_Q[8].put(val)
            elif i == 9: L_Q[9].put(val)
            elif i == 10: L_Q[10].put(val)
            elif i == 11: L_Q[11].put(val)
            elif i == 12: L_Q[12].put(val)
            elif i == 13: L_Q[13].put(val)
            elif i == 14: L_Q[14].put(val)
            elif i == 15: L_Q[15].put(val)

#first load K and then V matrix
    @df.kernel(mapping=[1], args=[K_m, V])
    def offchip_loadK(local_K: float32[BLOCK_T, HEAD_DIM], local_V: float32[BLOCK_T, HEAD_DIM]):
        for i, j in dsl.grid(BLOCK_T, HEAD_DIM):
            val = local_K[i, j]
            if j == 0: L_K[0].put(val)
            elif j == 1: L_K[1].put(val)
            elif j == 2: L_K[2].put(val)
            elif j == 3: L_K[3].put(val)
            elif j == 4: L_K[4].put(val)
            elif j == 5: L_K[5].put(val)
            elif j == 6: L_K[6].put(val)
            elif j == 7: L_K[7].put(val)
            elif j == 8: L_K[8].put(val)
            elif j == 9: L_K[9].put(val)
            elif j == 10: L_K[10].put(val)
            elif j == 11: L_K[11].put(val)
            elif j == 12: L_K[12].put(val)
            elif j == 13: L_K[13].put(val)
            elif j == 14: L_K[14].put(val)
            elif j == 15: L_K[15].put(val)

        for m, n in dsl.grid(BLOCK_T, HEAD_DIM):
            val = local_V[m, n]
            if m == 0: L_K[15].put(val)
            elif m == 1: L_K[14].put(val)
            elif m == 2: L_K[13].put(val)
            elif m == 3: L_K[12].put(val)
            elif m == 4: L_K[11].put(val)
            elif m == 5: L_K[10].put(val)
            elif m == 6: L_K[9].put(val)
            elif m == 7: L_K[8].put(val)
            elif m == 8: L_K[7].put(val)
            elif m == 9: L_K[6].put(val)
            elif m == 10: L_K[5].put(val)
            elif m == 11: L_K[4].put(val)
            elif m == 12: L_K[3].put(val)
            elif m == 13: L_K[2].put(val)
            elif m == 14: L_K[1].put(val)
            elif m == 15: L_K[0].put(val)



#store the rowsum and output O matrix
    @df.kernel(mapping=[1], args=[S, Sum, Max])
    def offchip_storeS(local_S: float32[BLOCK_T, HEAD_DIM], local_Sum: float32[BLOCK_T], local_max: float32[BLOCK_T]):

        for i in range(BLOCK_T):
            if i == 0: local_Sum[0] = L_S[0].get()
            elif i == 1: local_Sum[1] = L_S[1].get()
            elif i == 2: local_Sum[2] = L_S[2].get()
            elif i == 3: local_Sum[3] = L_S[3].get()
            elif i == 4: local_Sum[4] = L_S[4].get()
            elif i == 5: local_Sum[5] = L_S[5].get()
            elif i == 6: local_Sum[6] = L_S[6].get()
            elif i == 7: local_Sum[7] = L_S[7].get()
            elif i == 8: local_Sum[8] = L_S[8].get()
            elif i == 9: local_Sum[9] = L_S[9].get()
            elif i == 10: local_Sum[10] = L_S[10].get()
            elif i == 11: local_Sum[11] = L_S[11].get()
            elif i == 12: local_Sum[12] = L_S[12].get()
            elif i == 13: local_Sum[13] = L_S[13].get()
            elif i == 14: local_Sum[14] = L_S[14].get()
            elif i == 15: local_Sum[15] = L_S[15].get()

        for m, n in dsl.grid(BLOCK_T, HEAD_DIM):
            if m == 0: local_S[m, n] = L_S[0].get()
            elif m == 1: local_S[m, n] = L_S[1].get()
            elif m == 2: local_S[m, n] = L_S[2].get()
            elif m == 3: local_S[m, n] = L_S[3].get()
            elif m == 4: local_S[m, n] = L_S[4].get()
            elif m == 5: local_S[m, n] = L_S[5].get()
            elif m == 6: local_S[m, n] = L_S[6].get()
            elif m == 7: local_S[m, n] = L_S[7].get()
            elif m == 8: local_S[m, n] = L_S[8].get()
            elif m == 9: local_S[m, n] = L_S[9].get()
            elif m == 10: local_S[m, n] = L_S[10].get()
            elif m == 11: local_S[m, n] = L_S[11].get()
            elif m == 12: local_S[m, n] = L_S[12].get()
            elif m == 13: local_S[m, n] = L_S[13].get()
            elif m == 14: local_S[m, n] = L_S[14].get()
            elif m == 15: local_S[m, n] = L_S[15].get()

        


    @df.kernel(mapping=[P0, P1], args=[is_first_block])
    def gemm(is_first: bool):
        i, j = df.get_pid()

        with allo.meta_if(j == 0 and i == BLOCK_T-1):
            q: float32
            v: float32

            pre_m: float32 = L_M[j].get()

            pre_s: float32 = fifo_SD[i, j].get()
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
            s = s - max
            s = allo.exp(s*D)
            #get the sum of its row and output it
            ss: float32 = 0.0
            ss = s + fifo_SD[i,j].get()

            L_S[j].put(ss)
            #input V matrix and compute its multiplication with P, the element will stream through systolic array
            for g in range(BLOCK_T):
                v = L_K[i].get()
                fifo_K[i,j+1].put(v)
                ss = fifo_SD[i,j].get()
                ss = ss + s * v

                L_S[j].put(ss)


                
        with allo.meta_elif(i == BLOCK_T-1 and j == HEAD_DIM-1):
            q: float32

            pre_m: float32 = L_M[j].get()

            pre_s: float32 = fifo_SD[i, j].get()
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
            s = s - max
            s = allo.exp(s*D)

            if 
            #get the sum of its row and output it
            ss: float32 = 0.0
            ss = s + fifo_SD[i,j].get()

            L_S[j].put(ss)  
            #input V matrix and compute its multiplication with P, the element will stream through systolic array
            for g in range(BLOCK_T):
                v = fifo_K[i,j].get()
                ss = fifo_SD[i,j].get()
                ss = ss + s * v

                L_S[j].put(ss)


        #bottom two    
        with allo.meta_elif(i == BLOCK_T-1):
            q: float32

            pre_m: float32 = L_M[j].get()

            pre_s: float32 = fifo_SD[i, j].get()
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
            s = s - max
            s = allo.exp(s*D)
            #get the sum of its row and output it
            ss: float32 = 0.0
            ss = s + fifo_SD[i,j].get()

            L_S[j].put(ss)
            #input V matrix and compute its multiplication with P, the element will stream through systolic array
            for g in range(BLOCK_T):
                v = fifo_K[i, j].get()
                fifo_K[i,j+1].put(v)
                ss = fifo_SD[i,j].get()
                ss = ss + s * v

                L_S[j].put(ss)


        with allo.meta_elif(j == 0 and i == 0):
            q: float32 = 0
            s: float32 = 0
            v: float32
            ss: float32 = 0
            max: float32 = -128
            for m in range(BLOCK_T):
                pre_s = L_Q[j].get()
                if m != BLOCK_T-1:
                    fifo_SD[i+1, j].put(pre_s)
            #load Q matrix into systolic array, since it's weight stationary.
            q = fifo_Q[i,j].get()
            #input K matrix and compute its multiplication with Q, the element will stream through systolic array
            for m in range(BLOCK_T):
                a = L_K[i].get()
                fifo_K[i,j+1].put(a)
                s = fifo_S[i,j].get()
                s = s + a * q
                #the top row will act as a comparator and compute the row's max value on the fly
                if s > max:
                    max = s
                if BLOCK_T-i-m-1>0:
                    fifo_SD[i+1,j].put(s)
            #minus row max to avoid overflow
            s = s - max
            fifo_SD[i+1,j].put(max)
            s = allo.exp(s*D)

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

            for m in range(BLOCK_T):
                pre_s = L_Q[j].get()
                if m != BLOCK_T-1:
                    fifo_SD[i+1, j].put(pre_s)
            #load Q matrix into systolic array, since it's weight stationary.
            q = fifo_Q[i,j].get()
            #input K matrix and compute its multiplication with Q, the element will stream through systolic array
            for m in range(BLOCK_T):
                a = fifo_K[i,j].get()
                s = fifo_S[i,j].get()
                s = s + a * q
                #the top row will act as a comparator and compute the row's max value on the fly
                if s > max:
                    max = s
                if BLOCK_T-i-m-1>0:
                    fifo_SD[i+1,j].put(s) #keep one s
            #minus row max to avoid overflow
            s = s-max
            fifo_SD[i+1,j].put(max)
            s = allo.exp(s*D)
            
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
            pre_s: float32 = 0

            for m in range(BLOCK_T):
                pre_s = L_Q[j].get()
                if m != BLOCK_T-1:
                    fifo_SD[i+1, j].put(pre_s)
            #load Q matrix into systolic array, since it's weight stationary.
            q = fifo_Q[i,j].get()
            #input K matrix and compute its multiplication with Q, the element will stream through systolic array
            for m in range(BLOCK_T):
                a = fifo_K[i,j].get()
                fifo_K[i,j+1].put(a)
                s = fifo_S[i,j].get()
                s = s + a * q
                #the top row will act as a comparator and compute the row's max value on the fly
                if s > max:
                    max = s
                if BLOCK_T-i-m-1>0:
                    fifo_SD[i+1,j].put(s)
            #minus row max to avoid overflow
            s = s - max
            fifo_SD[i+1,j].put(max)
            s = allo.exp(s*D)

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
            pre_s: float32 = 0
            for m in range(BLOCK_T - i):
                pre_s = fifo_SD[i,j].get()
                if(m != BLOCK_T-i-1):
                    fifo_SD[i+1,j].put(pre_s)

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
            s = s - max
            fifo_SD[i+1,j].put(max)
            s = allo.exp(s*D)

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

            pre_s: float32 = 0
            for m in range(BLOCK_T - i):
                pre_s = fifo_SD[i,j].get()
                if(m != BLOCK_T-i-1):
                    fifo_SD[i+1,j].put(pre_s)

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
            s = s - max
            fifo_SD[i+1,j].put(max)
            s = allo.exp(s*D)

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

            pre_s: float32 = 0
            for m in range(BLOCK_T - i):
                pre_s = fifo_SD[i,j].get()
                if(m != BLOCK_T-i-1):
                    fifo_SD[i+1,j].put(pre_s)
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
            s = s - max
            fifo_SD[i+1,j].put(max)
            s = allo.exp(s*D)

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


def test_large_scale_gemm():

    A = np.random.uniform(-1.0, 1.0, (M, K)).astype(np.float32)
    B = np.random.uniform(-1.0, 1.0, (K, N)).astype(np.float32)
    V = np.random.uniform(-1.0, 1.0, (N, K)).astype(np.float32)
    A_hw = A.flatten()
    B_hw = B.flatten()
    O_hw = np.zeros(M*N, dtype=np.float32)
    Sum_hw = np.zeros(M, dtype=np.float32)
    V_hw = V.flatten()


    S_golden = np.dot(A, B.T)
    row_max = np.max(S_golden, axis=1, keepdims=True)
    P_golden = np.exp((S_golden - row_max) * D)

    Sum_golden = np.sum(P_golden, axis=1)

    O_golden = np.dot(P_golden, V)

    O_hw_emu = np.zeros(M*K, dtype=np.float32)
    Sum_hw_emu = np.zeros(M, dtype=np.float32)

    #sim_mod = df.build(top, target="simulator")
    #sim_mod(A_hw, B_hw, V_hw, O_hw_emu, Sum_hw_emu)
    #O_emu_out = O_hw_emu.reshape((M,K))
    #np.testing.assert_allclose(Sum_hw_emu, Sum_golden, atol=1e-5)
    #np.testing.assert_allclose(O_emu_out, O_golden, atol=1e-4)
    #print("Dataflow Simulator Passed!")


    if hls.is_available("vitis_hls"):

        modc = df.build(
            top,
            target="vitis_hls",
            mode="csyn",
            project=systolic_csyn,
            wrap_io=False,
        )
        modc()

        modhw = df.build(
            top,
            target="vitis_hls",
            mode="hw_emu",
            project=systolic_hw_emu,
            wrap_io=False,
        )
        O_hw_emu = np.zeros(M*N, dtype=np.float32)
        Sum_hw_emu = np.zeros(M, dtype=np.float32)
        modhw(A_hw, B_hw, V_hw, O_hw_emu, Sum_hw_emu)
        O_emu_out = O_hw_emu.reshape((M,K))
        np.testing.assert_allclose(Sum_hw_emu, Sum_golden, atol=1e-5)
        print("Sum is CORRECT!!!")

        np.testing.assert_allclose(O_emu_out, O_golden, atol=1e-4)
        print("Output O is CORRECT!!!")

        print("hw_emu tests passed!!!")

if __name__ == "__main__":
    test_large_scale_gemm()