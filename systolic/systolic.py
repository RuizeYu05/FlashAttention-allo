import tempfile
import pytest
import allo
from allo.ir.types import int8, int32, Stream, UInt, ConstExpr
from allo.utils import get_np_struct_type
import allo.dataflow as df
import allo.backend.hls as hls
import allo.dsl as dsl
import numpy as np


@df.region()
def systolicAttention[
    Rt, Ct, M, N, K
](
    Q_Packed: "UInt(32)[M]",
    K_Packed: "UInt(32)[N]",
    S_Packed_out: "int8[M*N]",
    Max_out: "int8[M]",

):
    P0: ConstExpr[int32] = Rt + 2
    P1: ConstExpr[int32] = Ct + 2

    L3_Q: Stream[UInt(32), 1024]
    L3_K: Stream[UInt(32), 1024]
    L3_S: Stream[int8, 1024]
    L3_Max: Stream[int8, 1024]

    L2_Q: Stream[UInt(32), 1024][P0 - 1]
    L2_K: Stream[UInt(32), 1024][P1 - 1]

    L1_C: Stream[int8, 1024][Ct]
    L2_C: Stream[int8, 1024][P1 - 1]
    L2_S: Stream[int8, 1024][P1 - 1]


    fifo_Q: Stream[int8, 1024][Rt, Ct]
    fifo_K: Stream[int8, 1024][Rt, Ct]
    fifo_S: Stream[int8, 1024][Rt, Ct]

    @df.kernel(mapping=[1], args=[K_Packed])
    def offchip_loadK(K_Packed_in: "UInt(32)[N]"):
        for n in range(N):
            L3_K.put(K_Packed_in[n])

    @df.kernel(mapping=[1], args=[Q_Packed])
    def offchip_loadQ(Q_Packed_in: "UInt(32)[M]"):
        for m in range(M):
            L3_Q.put(Q_Packed_in[m])

    @df.kernel(mapping=[P0, P1])
    def gemm():
        i, j = df.get_pid()

        with allo.meta_if(i == 0 and j == 0):
            for n in range(N):
                L2_K[1].put(L3_K.get())

        with allo.meta_elif(i == P0 - 1 and j == 0):
            for m in range(M):
                L2_Q[1].put(L3_Q.get())

        with allo.meta_elif(i == 0 and j == P1 -1):
            for n in range(N):
                for c in range(Ct):
                    L3_S.put(L2_S[Ct - 1].get())
            for c in range(Ct):
                L3_Max.put(L2_C[Ct - 1].get())

        with allo.meta_elif(i == P0-1 and j == P1 - 1):
            pass

        with allo.meta_elif(j == 0):
            for n in range(N):
                val: UInt(32) = L2_K[i].get()
                with allo.meta_if(i < Rt):
                    L2_K[i+1].put(val)
                feat: int8 = val[8*(i-1):8*i]
                fifo_K[i-1, 0].put(feat)

        with allo.meta_elif(i == P0 - 1):
            for m in range(M):
                val: UInt(32) = L2_Q[j].get()
                with allo.meta_if(j < Ct):
                    L2_Q[j+1].put(val)

                if m == j-1:
                    for r in range(Rt):
                        fifo_Q[Rt-1, j-1].put(val[8*r:8*(r+1)])


        with allo.meta_elif(i == 0):
            local_max: int8 = 2
            for n in range(N):

                local_val: int8 = L1_C[j-1].get()

                with allo.meta_if(j != 1):
                    for ind in range(j - 1):
                        L2_S[j-1].put(L2_S[j-2].get())
                L2_S[j-1].put(local_val)

            with allo.meta_if(j != 1):
                for ind in range(j - 1):
                    L2_C[j-1].put(L2_C[j-2].get())
            L2_C[j-1].put(local_max)

        with allo.meta_elif(j == P1 - 1):
            pass

        with allo.meta_else():
            b_reg: int8 = 0
            for k in range(i):
                tmp_b: int8 = fifo_Q[i-1, j-1].get()
                if k == i-1:
                    b_reg = tmp_b
                else:
                    with allo.meta_if(i>1):
                        fifo_Q[i-2, j-1].put(tmp_b)

            for n in range(N):
                a: int8 = fifo_K[i-1, j-1].get()

                s_in: int8 = 0
                with allo.meta_if(i == Rt):
                    s_in = 0
                with allo.meta_else():
                    s_in = fifo_S[i-1, j-1].get()
                
                s_out: int8 = 1

                with allo.meta_if(j < Ct):
                    fifo_K[i-1, j].put(a)
                with allo.meta_if(i > 1):
                    fifo_S[i-2, j-1].put(s_out)
                with allo.meta_else():
                    L1_C[j-1].put(s_out)


    @df.kernel(mapping=[1], args=[S_Packed_out, Max_out])
    def offchip_store(
        S_Packed_i: "int8[M*N]",
        Max_out_i: "int8[M]",
    ):
        for n in range(N):
            for m in range(M):
                S_Packed_i[n*M+m] = L3_S.get()
        for m in range(M):
            Max_out_i[m] = L3_Max.get()

Rt, Ct = 4, 4
M, N, K = 4, 4, 4

@df.region()
def top(
    Q: "UInt(32)[M]",
    K: "UInt(32)[N]",
    S: "int8[M*N]",
    Max_o: "int8[M]",
):
    @df.kernel(mapping=[1], args=[Q, K, S, Max_o])
    def wrapper(
        Q_Packed: "UInt(32)[M]",
        K_Packed: "UInt(32)[N]",
        S_Packed: "int8[M*N]",
        Max: "int8[M]",
    ):
        systolicAttention[Rt,Ct,M,N,K](Q_Packed, K_Packed, S_Packed, Max)

def test_systolic_attention_4x4():
    def serialize_to_uint32(matrix):
        return matrix.flatten().view(np.uint32)

    def deserialize_S(S_ser):
        matrix_S = np.zeros((M, N), dtype=np.int8)
        idx = 0
        for n in range(N):
            for m in range(M):
                matrix_S[m, n] = S_ser[idx]
                idx += 1
        return matrix_S

    Q_np = np.random.randint(-2, 3, (M, K), dtype=np.int8)
    K_np = np.random.randint(-2, 3, (N, K), dtype=np.int8)
    
    S_golden = np.dot(Q_np, K_np.T)
    Max_golden = np.max(S_golden, axis=1)
    
    Q_packed = serialize_to_uint32(Q_np)
    K_packed = serialize_to_uint32(K_np)
    S_out = np.zeros(M * N, dtype=np.int8)
    Max_out = np.zeros(M, dtype=np.int8)
    
    sim_mod = df.build(top, target="simulator")
    print("Start 4x4 Dataflow Simulator")
    sim_mod(Q_packed, K_packed, S_out, Max_out)
    
    S_sim = deserialize_S(S_out)
    
    print("S Matrix Output:\n", S_sim)
    print("Golden S Matrix:\n", S_golden)
    np.testing.assert_allclose(S_sim, S_golden)
    
    print("\nRow-Max Output:\n", Max_out)
    print("Golden Row-Max:\n", Max_golden)
    np.testing.assert_allclose(Max_out, Max_golden)
    
    print("\n=> 4x4 Dataflow Simulator Passed Successfully!")

if __name__ == "__main__":
    test_systolic_attention_4x4()