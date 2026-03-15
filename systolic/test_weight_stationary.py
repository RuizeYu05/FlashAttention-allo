import tempfile

import pytest
import allo
from allo.ir.types import int8, Stream, UInt
from allo.utils import get_np_struct_type
import allo.dataflow as df
import allo.backend.hls as hls
import allo.dsl as dsl
import numpy as np

M, K, N = 4, 4, 4
P0, P1 = K, N

@df.region()
def top(
    Q: int8[M*K],
    K_m: int8[K*N],
    S: int8[M*N],
):
    L_Q: Stream[int8, 64][P1]
    L_K: Stream[int8, 64][P0]
    L_S: Stream[int8, 64][P1]


    fifo_Q: Stream[int8, 64][P0, P1]
    fifo_K: Stream[int8, 64][P0, P1]
    fifo_S: Stream[int8, 64][P0, P1]


    @df.kernel(mapping=[1], args=[Q])
    def offchip_loadQ(local_Q: int8[M*K]):
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
            

    @df.kernel(mapping=[1], args=[K_m])
    def offchip_loadK(local_K: int8[K*N]):
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

    @df.kernel(mapping=[1], args=[S])
    def offchip_storeS(local_S: int8[M*N]):
        for i, j in dsl.grid(M, N):
            if i == 0:
                local_S[i*N+j]=L_S[0].get()
            elif i == 1:
                local_S[i*N+j]=L_S[1].get()
            elif i == 2:
                local_S[i*N+j]=L_S[2].get()
            elif i == 3:
                local_S[i*N+j]=L_S[3].get()


    @df.kernel(mapping=[P0, P1], args=[])
    def gemm():
        i, j = df.get_pid()

        with allo.meta_if(j == 0 and i == K-1):
            q: int8
            for m in range(K):
                q = L_Q[j].get()
                if m != K-1:
                    fifo_Q[i-1, j].put(q)

            s: int8 = 0
            for k in range(K):
                a: UInt(8) = L_K[i].get()
                s = a * q
                fifo_K[i, j+1].put(a)
                fifo_S[i-1, j].put(s)

        with allo.meta_elif(i == K-1 and j == N-1):
            q: int8
            for m in range(K):
                q = L_Q[j].get()
                if m != K-1:
                    fifo_Q[i-1, j].put(q)

            s: int8 = 0
            a: int8 = 0
            for n in range(K):
                a = fifo_K[i, j].get()
                s = a * q
                fifo_S[i-1, j].put(s)
                
        with allo.meta_elif(i == K-1):
            q: int8
            for m in range(K):
                q = L_Q[j].get()
                if m != K-1:
                    fifo_Q[i-1, j].put(q)
            
            s: int8 = 0
            a: int8 = 0
            for n in range(K):
                a = fifo_K[i, j].get()
                s = a * q
                fifo_K[i,j+1].put(a)
                fifo_S[i-1, j].put(s)

        with allo.meta_elif(j == 0 and i == 0):
            q: int8 = 0
            s: int8 = 0
            q = fifo_Q[i,j].get()
            for m in range(K):
                a = L_K[i].get()
                fifo_K[i,j+1].put(a)
                s = fifo_S[i,j].get()
                s = s + a * q
                L_S[j].put(s)

        with allo.meta_elif(j == P1-1 and i == 0):
            q: int8 = 0
            s: int8 = 0
            q = fifo_Q[i,j].get()
            for m in range(K):
                a = fifo_K[i,j].get()
                s = fifo_S[i,j].get()
                s = s + a * q
                L_S[j].put(s)

        with allo.meta_elif(i == 0):
            q: int8 = 0
            s: int8 = 0
            q = fifo_Q[i,j].get()
            for m in range(K):
                a = fifo_K[i,j].get()
                fifo_K[i,j+1].put(a)
                s = fifo_S[i,j].get()
                s = s + a * q
                L_S[j].put(s)

        with allo.meta_elif(j == 0):
            q: int8 = 0
            s: int8 = 0
            for m in range(i+1):
                q = fifo_Q[i,j].get()
                if i-m>0:
                    fifo_Q[i-1, j].put(q)
            
            for n in range(K):
                a = L_K[i].get()
                fifo_K[i,j+1].put(a)
                s = fifo_S[i,j].get()

                s = s+q*a
                fifo_S[i-1, j].put(s)

        with allo.meta_elif(j == P1-1):
            q: int8 = 0
            s: int8 = 0
            for m in range(i+1):
                q = fifo_Q[i,j].get()
                if i-m>0:
                    fifo_Q[i-1, j].put(q)

            for n in range(K):
                a = fifo_K[i,j].get()
                s = fifo_S[i,j].get()

                s = s+q*a
                fifo_S[i-1, j].put(s)

        with allo.meta_else():
            q: int8 = 0
            s: int8 = 0
            for m in range(i+1):
                q = fifo_Q[i,j].get()
                if i-m>0:
                    fifo_Q[i-1, j].put(q)

            for n in range(K):
                a = fifo_K[i,j].get()
                fifo_K[i,j+1].put(a)
                s = fifo_S[i,j].get()

                s = s+q*a
                fifo_S[i-1, j].put(s)

def test_large_scale_gemm():

    A = np.random.randint(-2, 2, (M,K), dtype=np.int8)
    B = np.random.randint(-2, 2, (K,N), dtype=np.int8)
    A_hw = A.flatten()
    B_hw = B.flatten()
    C_hw = np.zeros(M*N, dtype=np.int8)

    sim_mod = df.build(top, target="simulator")
    sim_mod(A_hw, B_hw, C_hw)
    C_out = C_hw.reshape((M,N))
    np.testing.assert_allclose(C_out, np.dot(A, B.T), atol=1e-5)
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
            C_hw_emu = np.zeros(M*N, dtype=np.int8)
            modhw(A_hw, B_hw, C_hw_emu)
            C_emu_out = C_hw_emu.reshape((M,N))
            np.testing.assert_allclose(C_emu_out, np.dot(A, B.T), atol=1e-5)
            print("hw_emu test passed")

if __name__ == "__main__":
    test_large_scale_gemm()


