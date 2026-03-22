#Full implementation of systolic flashattention
#Matrix multiplication is weight stationary

import tempfile

import pytest
import allo
from allo.ir.types import int8, Stream, UInt, float32, int32
from allo.utils import get_np_struct_type
import allo.dataflow as df
import allo.backend.hls as hls
import allo.dsl as dsl
import numpy as np

BLOCK_T, HEAD_DIM= 4, 4
P0, P1 = HEAD_DIM, BLOCK_T
D = 1.0 / float(HEAD_DIM ** 0.5)

@df.region()
def top(
    Q: float32[BLOCK_T, HEAD_DIM],
    K_m: float32[BLOCK_T, HEAD_DIM],
    V: float32[BLOCK_T, HEAD_DIM],
    is_first_block: int32[1],
    Sum: float32[BLOCK_T],
    Max: float32[BLOCK_T],
    S: float32[BLOCK_T, HEAD_DIM],
):
    L_Q: Stream[float32, 1024][P1]
    L_K: Stream[float32, 1024][P0]
    L_S: Stream[float32, 1024][P1]


    fifo_Q: Stream[float32, 1024][P0, P1]
    fifo_K: Stream[float32, 1024][P0, P1]
    fifo_S: Stream[float32, 1024][P0, P1]
    fifo_SD: Stream[float32, 1024][P0, P1]

#load Q matrix
    @df.kernel(mapping=[1], args=[Q, Max, S, Sum])
    def offchip_loadQ(local_Q: float32[BLOCK_T, HEAD_DIM], local_Max: float32[BLOCK_T], local_S: float32[BLOCK_T, HEAD_DIM], local_Sum: float32[BLOCK_T]):

        for i in range(BLOCK_T):
            val = local_Sum[i]
            if i == 0: L_Q[0].put(val)
            elif i == 1: L_Q[1].put(val)
            elif i == 2: L_Q[2].put(val)
            elif i == 3: L_Q[3].put(val)

        for i in range(BLOCK_T):
            val = local_Max[i]
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
            val = local_S[i, j]
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
    @df.kernel(mapping=[1], args=[S, Sum, Max])
    def offchip_storeS(local_S: float32[BLOCK_T, HEAD_DIM], local_Sum: float32[BLOCK_T], local_Max: float32[BLOCK_T]):
        
        for i in range(BLOCK_T):
            if i == 0: local_Max[0] = L_S[0].get()
            elif i == 1: local_Max[1] = L_S[1].get()
            elif i == 2: local_Max[2] = L_S[2].get()
            elif i == 3: local_Max[3] = L_S[3].get()

        for i in range(BLOCK_T):
            if i == 0: local_Sum[0] = L_S[0].get()
            elif i == 1: local_Sum[1] = L_S[1].get()
            elif i == 2: local_Sum[2] = L_S[2].get()
            elif i == 3: local_Sum[3] = L_S[3].get()

        for m, n in dsl.grid(BLOCK_T, HEAD_DIM):
            if m == 0: local_S[m, n] = L_S[0].get()
            elif m == 1: local_S[m, n] = L_S[1].get()
            elif m == 2: local_S[m, n] = L_S[2].get()
            elif m == 3: local_S[m, n] = L_S[3].get()



    @df.kernel(mapping=[P0, P1], args=[is_first_block])
    def gemm(is_first: int32[1]):
        i, j = df.get_pid()

        with allo.meta_if(j == 0 and i == BLOCK_T-1):
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
            s = s - max
            s = allo.exp(s*D)

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


                
        with allo.meta_elif(i == BLOCK_T-1 and j == HEAD_DIM-1):
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
            s = s - max
            s = allo.exp(s*D)

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
        with allo.meta_elif(i == BLOCK_T-1):
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
            s = s - max
            s = allo.exp(s*D)

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

    SEQ_LEN = 8

    Q_full = np.random.uniform(-1.0, 1.0, (SEQ_LEN, HEAD_DIM)).astype(np.float32)
    K_full = np.random.uniform(-1.0, 1.0, (SEQ_LEN, HEAD_DIM)).astype(np.float32)
    V_full = np.random.uniform(-1.0, 1.0, (SEQ_LEN, HEAD_DIM)).astype(np.float32)

    S_full = np.dot(Q_full, K_full.T)
    row_max = np.max(S_full, axis=1, keepdims=True)
    P_full = np.exp((S_full - row_max)*D)
    row_sum = np.sum(P_full, axis=1, keepdims=True)
    O_golden = np.dot(P_full, V_full) / row_sum

    Q_tile = Q_full[0:BLOCK_T, :]
    O_hw_state = np.zeros((BLOCK_T, HEAD_DIM), dtype=np.float32)
    Sum_hw_state = np.zeros(BLOCK_T, dtype=np.float32)
    Max_hw_state = np.full(BLOCK_T, -1e30, dtype=np.float32)

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
            project="systolic_csyn",
            wrap_io=False,
        )
        modc()

        modhw = df.build(
            top,
            target="vitis_hls",
            mode="hw_emu",
            project="systolic_hw_emu",
            wrap_io=False,
        )

        for tc in range(0, SEQ_LEN, BLOCK_T):
            print(f"   => 正在向脉动阵列喂入 K, V 的第 {tc//BLOCK_T + 1} 个 Tile...")
            is_first_val = 1 if tc == 0 else 0
            is_first_block = np.array([is_first_val], dtype=np.int32)

            # 切片当前需要的 K 和 V
            K_tile = K_full[tc:tc+BLOCK_T, :]
            V_tile = V_full[tc:tc+BLOCK_T, :]

            # 执行硬件，状态会在 O_hw_state, Sum_hw_state, Max_hw_state 里就地更新
            modhw(
                Q_tile.flatten(),
                K_tile.flatten(),
                V_tile.flatten(),
                is_first_block,
                O_hw_state.flatten(),
                Sum_hw_state.flatten(),
                Max_hw_state.flatten(),
            )

        print("✅ 硬件计算完毕，正在对比结果...")

        # 6. 因为我们只喂了 Q 的第一个 Tile，所以只需和 Golden 的前 BLOCK_T 行比对
        # FlashAttention 别忘了最后要除以 Sum (如果硬件里没做的话)
        O_hw_final = O_hw_state / Sum_hw_state.reshape(BLOCK_T, 1)

        try:
            np.testing.assert_allclose(O_hw_final, O_golden[0:BLOCK_T, :], atol=1e-3)
            print("🎉 恭喜！！！4x4 脉动阵列 Tiled FlashAttention 测试完美通过！")
        except AssertionError as e:
            print("❌ 对比失败！请检查 hw_emu 的数据流逻辑或 FIFO 是否死锁。")
            print(e)

if __name__ == "__main__":
    test_large_scale_gemm()