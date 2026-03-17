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

M, K, N = 4, 4, 4
P0, P1 = K, N
D = 1.0 / float(4 ** 0.5)

@df.region()
def top(
    Q: float32[M*K],
    K_m: float32[N*K],
    V: float32[N*K],
    S: float32[M*K],
    Sum: float32[M],
):
    L_Q: Stream[float32, 1024][P1]
    L_K: Stream[float32, 1024][P0]
    L_S: Stream[float32, 1024][P1]


    fifo_Q: Stream[float32, 1024][P0, P1]
    fifo_K: Stream[float32, 1024][P0, P1]
    fifo_S: Stream[float32, 1024][P0, P1]
    fifo_SD: Stream[float32, 1024][P0, P1]

#load Q matrix
    @df.kernel(mapping=[1], args=[Q])
    def offchip_loadQ(local_Q: float32[M*K]):
        for i, j in dsl.grid(M, K):
            val = local_Q[i * K + j]
            if i == 0:
                L_Q[0].put(val)
            elif i == 1:
                L_Q[1].put(val)
            elif i == 2:
                L_Q[2].put(val)
            elif i == 3:
                L_Q[3].put(val)

#first load K and then V matrix
    @df.kernel(mapping=[1], args=[K_m, V])
    def offchip_loadK(local_K: float32[K*N], local_V: float32[N*K]):
        for i, j in dsl.grid(K, N):
            val = local_K[i*N+j]
            if j == 0:
                L_K[0].put(val)
            elif j == 1:
                L_K[1].put(val)
            elif j == 2:
                L_K[2].put(val)
            elif j == 3:
                L_K[3].put(val)

        for m, n in dsl.grid(N, K):
            val = local_V[m*K+n]
            if m == 0:
                L_K[3].put(val)
            elif m == 1:
                L_K[2].put(val)
            elif m == 2:
                L_K[1].put(val)
            elif m == 3:
                L_K[0].put(val)



#store the rowsum and output O matrix
    @df.kernel(mapping=[1], args=[S, Sum])
    def offchip_storeS(local_S: float32[M*N], local_Sum: float32[M]):

        for i in range(M):
            if i == 0:
                local_Sum[i] = L_S[0].get()
            elif i == 1:
                local_Sum[i] = L_S[1].get()
            elif i == 2:
                local_Sum[i] = L_S[2].get()
            elif i == 3:
                local_Sum[i] = L_S[3].get()

        for m, n in dsl.grid(M, K):
            if m == 0:
                local_S[m*K+n]=L_S[0].get()
            elif m == 1:
                local_S[m*K+n]=L_S[1].get()
            elif m == 2:
                local_S[m*K+n]=L_S[2].get()
            elif m == 3:
                local_S[m*K+n]=L_S[3].get()

        


    @df.kernel(mapping=[P0, P1], args=[])
    def gemm():
        i, j = df.get_pid()

        with allo.meta_if(j == 0 and i == K-1):
            q: float32
            v: float32
            #load Q matrix into systolic array, since it's weight stationary.
            for m in range(K):
                q = L_Q[j].get()
                if m != K-1:
                    fifo_Q[i-1, j].put(q)
            #input K matrix and compute its multiplication with Q, the element will stream through systolic array
            s: float32 = 0
            for k in range(K):
                a: float32 = L_K[i].get()
                s = a * q
                fifo_K[i, j+1].put(a)
                fifo_S[i-1, j].put(s)

            s = fifo_SD[i,j].get()
            max = fifo_SD[i,j].get()
            #minus row max to avoid overflow
            s = s - max
            #s = allo.exp(s*D)
            #get the sum of its row and output it
            ss: float32 = 0.0
            ss = s + fifo_SD[i,j].get()

            L_S[j].put(ss)
            #input V matrix and compute its multiplication with P, the element will stream through systolic array
            for g in range(K):
                v = L_K[i].get()
                fifo_K[i,j+1].put(v)
                ss = fifo_SD[i,j].get()
                ss = ss + s * v

                L_S[j].put(ss)


                
        with allo.meta_elif(i == K-1 and j == N-1):
            q: float32
            #load Q matrix into systolic array, since it's weight stationary.
            for m in range(K):
                q = L_Q[j].get()
                if m != K-1:
                    fifo_Q[i-1, j].put(q)
            #input K matrix and compute its multiplication with Q, the element will stream through systolic array
            s: float32 = 0
            a: float32 = 0
            for n in range(K):
                a = fifo_K[i, j].get()
                s = a * q
                fifo_S[i-1, j].put(s)
            
            s = fifo_SD[i,j].get()
            max = fifo_SD[i,j].get()
            #minus row max to avoid overflow
            s = s - max
            #s = allo.exp(s*D)
            #get the sum of its row and output it
            ss: float32 = 0.0
            ss = s + fifo_SD[i,j].get()

            L_S[j].put(ss)  
            #input V matrix and compute its multiplication with P, the element will stream through systolic array
            for g in range(K):
                v = fifo_K[i,j].get()
                ss = fifo_SD[i,j].get()
                ss = ss + s * v

                L_S[j].put(ss)


        #bottom two    
        with allo.meta_elif(i == K-1):
            q: float32
            #load Q matrix into systolic array, since it's weight stationary.
            for m in range(K):
                q = L_Q[j].get()
                if m != K-1:
                    fifo_Q[i-1, j].put(q)
            
            s: float32 = 0
            a: float32 = 0
            #input K matrix and compute its multiplication with Q, the element will stream through systolic array
            for n in range(K):
                a = fifo_K[i, j].get()
                s = a * q
                fifo_K[i,j+1].put(a)
                fifo_S[i-1, j].put(s)

            s = fifo_SD[i,j].get()
            max = fifo_SD[i,j].get()
            #minus row max to avoid overflow
            s = s - max
            #s = allo.exp(s*D)
            #get the sum of its row and output it
            ss: float32 = 0.0
            ss = s + fifo_SD[i,j].get()

            L_S[j].put(ss)
            #input V matrix and compute its multiplication with P, the element will stream through systolic array
            for g in range(K):
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
            #load Q matrix into systolic array, since it's weight stationary.
            q = fifo_Q[i,j].get()
            #input K matrix and compute its multiplication with Q, the element will stream through systolic array
            for m in range(K):
                a = L_K[i].get()
                fifo_K[i,j+1].put(a)
                s = fifo_S[i,j].get()
                s = s + a * q
                #the top row will act as a comparator and compute the row's max value on the fly
                if s > max:
                    max = s
                if K-i-m-1>0:
                    fifo_SD[i+1,j].put(s)
            #minus row max to avoid overflow
            s = s - max
            fifo_SD[i+1,j].put(max)
            #s = allo.exp(s*D)

            fifo_SD[i+1,j].put(s)
            #input V matrix and compute its multiplication with P, the element will stream through systolic array
            for g in range(K):
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
            for m in range(K):
                a = fifo_K[i,j].get()
                s = fifo_S[i,j].get()
                s = s + a * q
                #the top row will act as a comparator and compute the row's max value on the fly
                if s > max:
                    max = s
                if K-i-m-1>0:
                    fifo_SD[i+1,j].put(s) #keep one s
            #minus row max to avoid overflow
            s = s-max
            fifo_SD[i+1,j].put(max)
            #s = allo.exp(s*D)
            
            fifo_SD[i+1,j].put(s)
            #input V matrix and compute its multiplication with P, the element will stream through systolic array
            for g in range(K):
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
            for m in range(K):
                a = fifo_K[i,j].get()
                fifo_K[i,j+1].put(a)
                s = fifo_S[i,j].get()
                s = s + a * q
                #the top row will act as a comparator and compute the row's max value on the fly
                if s > max:
                    max = s
                if K-i-m-1>0:
                    fifo_SD[i+1,j].put(s)
            #minus row max to avoid overflow
            s = s - max
            fifo_SD[i+1,j].put(max)
            #s = allo.exp(s*D)

            fifo_SD[i+1,j].put(s)
            #input V matrix and compute its multiplication with P, the element will stream through systolic array
            for g in range(K):
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
            for n in range(K):
                a = L_K[i].get()
                fifo_K[i,j+1].put(a)
                s = fifo_S[i,j].get()

                s = s+q*a
                fifo_S[i-1, j].put(s)

            for l in range(K):
                if K-i-l>0:
                    s = fifo_SD[i,j].get()
                if K-i-l-1>0:
                    fifo_SD[i+1, j].put(s)
                    
            max = fifo_SD[i,j].get()
            #minus row max to avoid overflow
            s = s - max
            fifo_SD[i+1,j].put(max)
            #s = allo.exp(s*D)

            ss: float32 = 0.0
            ss = s + fifo_SD[i,j].get()
            fifo_SD[i+1,j].put(ss)
            #input V matrix and compute its multiplication with P, the element will stream through systolic array
            for g in range(K):
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
            for n in range(K):
                a = fifo_K[i,j].get()
                s = fifo_S[i,j].get()

                s = s+q*a
                fifo_S[i-1, j].put(s)

            for l in range(K):
                if K-i-l>0:
                    s = fifo_SD[i,j].get()
                if K-i-l-1>0:
                    fifo_SD[i+1, j].put(s)
            max = fifo_SD[i,j].get()
            #minus row max to avoid overflow
            s = s - max
            fifo_SD[i+1,j].put(max)
            #s = allo.exp(s*D)

            ss: float32 = 0.0
            ss = s + fifo_SD[i,j].get()
            fifo_SD[i+1,j].put(ss)
            #input V matrix and compute its multiplication with P, the element will stream through systolic array
            for g in range(K):
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
            for n in range(K):
                a = fifo_K[i,j].get()
                fifo_K[i,j+1].put(a)
                s = fifo_S[i,j].get()

                s = s+q*a
                fifo_S[i-1, j].put(s)

            for l in range(K):
                if K-i-l>0:
                    s = fifo_SD[i,j].get()
                if K-i-l-1>0:
                    fifo_SD[i+1, j].put(s)

            max = fifo_SD[i,j].get()
            #minus row max to avoid overflow
            s = s - max
            fifo_SD[i+1,j].put(max)
            #s = allo.exp(s*D)

            ss: float32 = 0.0
            ss = s + fifo_SD[i,j].get()
            fifo_SD[i+1,j].put(ss)
            #input V matrix and compute its multiplication with P, the element will stream through systolic array
            for g in range(K):
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
    #P_golden = np.exp((S_golden - row_max) * D)
    P_golden = S_golden - row_max

    Sum_golden = np.sum(P_golden, axis=1)

    O_golden = np.dot(P_golden, V)

    O_hw_emu = np.zeros(M*K, dtype=np.float32)
    Sum_hw_emu = np.zeros(M, dtype=np.float32)

    sim_mod = df.build(top, target="simulator")
    sim_mod(A_hw, B_hw, V_hw, O_hw_emu, Sum_hw_emu)
    O_emu_out = O_hw_emu.reshape((M,K))
    np.testing.assert_allclose(Sum_hw_emu, Sum_golden, atol=1e-5)
    np.testing.assert_allclose(O_emu_out, O_golden, atol=1e-4)
    print("Dataflow Simulator Passed!")


    if hls.is_available("vitis_hls"):

        with tempfile.TemporaryDirectory() as tmpdir:
            modc = df.build(
                top,
                target="vitis_hls",
                mode="csyn",
                project=tmpdir,
                wrap_io=False,
            )
            modc()

        with tempfile.TemporaryDirectory() as tmpdir:
            modhw = df.build(
                top,
                target="vitis_hls",
                mode="hw_emu",
                project=tmpdir,
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