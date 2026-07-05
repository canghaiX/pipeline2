# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


cpp_fused___rshift____to_copy_add_bitwise_and_copy__embedding_native_layer_norm_0 = async_compile.cpp_pybinding(['const int32_t*', 'const float*', 'const float*', 'const int64_t*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*', 'int32_t*', 'const int64_t'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const int32_t* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const int64_t* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       int32_t* out_ptr5,
                       const int64_t ks0)
{
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(ks0); x0+=static_cast<int64_t>(1L))
            {
                {
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    static WelfordHelper<float, 4096> scalar_welford_helper0(static_cast<int64_t>(1024L));
                    static WelfordHelper<at::vec::Vectorized<float>, 4096> welford_helper0(static_cast<int64_t>(64L));
                    static WelfordHelper<at::vec::Vectorized<float>, 4096> masked_welford_helper0(static_cast<int64_t>(0L));
                    for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(1024L); x1+=static_cast<int64_t>(16L))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(1024L)))
                            {
                                auto tmp0 = out_ptr5[static_cast<int64_t>(x0)];
                                auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(16));
                                auto tmp27 = in_ptr3[static_cast<int64_t>(x0)];
                                auto tmp1 = static_cast<int32_t>(-1073741825);
                                auto tmp2 = decltype(tmp0)(tmp0 & tmp1);
                                auto tmp3 = c10::convert<int64_t>(tmp2);
                                auto tmp4 = 250048L;
                                auto tmp5 = c10::convert<int64_t>(tmp4);
                                auto tmp6 = int64_t(tmp3 + tmp5);
                                auto tmp7 = tmp3 < 0;
                                auto tmp8 = tmp7 ? tmp6 : tmp3;
                                auto tmp9 = tmp8;
                                auto tmp10 = c10::convert<int64_t>(tmp9);
                                TORCH_CHECK((0 <= tmp10) & (tmp10 < 250048L), "index out of bounds: 0 <= tmp10 < 250048L");
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 1024L*tmp8), static_cast<int64_t>(16));
                                auto tmp13 = static_cast<int32_t>(1073741824);
                                auto tmp14 = decltype(tmp0)(tmp0 & tmp13);
                                auto tmp15 = static_cast<int32_t>(30);
                                auto tmp16 =
                                [&]()
                                {
                                    constexpr decltype(tmp15) max_shift = sizeof(int32_t) * CHAR_BIT - std::is_signed_v<int32_t>;
                                    if ((static_cast<std::make_signed_t<int32_t>>(tmp15) < 0) || (tmp15 >= max_shift))
                                    {
                                        return decltype(tmp14)(tmp14 >> max_shift);
                                    }
                                    return decltype(tmp14)(tmp14 >> tmp15);
                                }
                                ()
                                ;
                                auto tmp17 = 1L;
                                auto tmp18 = c10::convert<int64_t>(tmp17);
                                auto tmp19 = int64_t(tmp16 + tmp18);
                                auto tmp20 = tmp16 < 0;
                                auto tmp21 = tmp20 ? tmp19 : tmp16;
                                auto tmp22 = tmp21;
                                auto tmp23 = c10::convert<int64_t>(tmp22);
                                TORCH_CHECK((0 <= tmp23) & (tmp23 < 1L), "index out of bounds: 0 <= tmp23 < 1L");
                                auto tmp26 = tmp12 + tmp25;
                                auto tmp28 = static_cast<int64_t>(1);
                                auto tmp29 = int64_t(tmp27 + tmp28);
                                auto tmp30 = int64_t(tmp29 + tmp28);
                                auto tmp31 = 8194L;
                                auto tmp32 = c10::convert<int64_t>(tmp31);
                                auto tmp33 = int64_t(tmp30 + tmp32);
                                auto tmp34 = tmp30 < 0;
                                auto tmp35 = tmp34 ? tmp33 : tmp30;
                                auto tmp36 = tmp35;
                                auto tmp37 = c10::convert<int64_t>(tmp36);
                                TORCH_CHECK((0 <= tmp37) & (tmp37 < 8194L), "index out of bounds: 0 <= tmp37 < 8194L");
                                auto tmp39 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x1 + 1024L*tmp35), static_cast<int64_t>(16));
                                auto tmp40 = tmp26 + tmp39;
                                tmp40.store(out_ptr0 + static_cast<int64_t>(x1 + 1024L*x0));
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp40, &welford_helper0);
                            }
                        }
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, &scalar_welford_helper0);
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, &welford_helper0);
                    masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, &masked_welford_helper0);
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(ks0); x0+=static_cast<int64_t>(1L))
            {
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(1024L); x1+=static_cast<int64_t>(16L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(1024L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<int64_t>(x1 + 1024L*x0), static_cast<int64_t>(16));
                            auto tmp1 = out_ptr1[static_cast<int64_t>(x0)];
                            auto tmp4 = out_ptr2[static_cast<int64_t>(x0)];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x1), static_cast<int64_t>(16));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<int64_t>(x1), static_cast<int64_t>(16));
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 - tmp2;
                            auto tmp5 = static_cast<float>(1024.0);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = float(tmp6 + tmp7);
                            auto tmp9 = 1 / std::sqrt(tmp8);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp3 * tmp10;
                            auto tmp13 = tmp11 * tmp12;
                            auto tmp15 = tmp13 + tmp14;
                            tmp15.store(out_ptr3 + static_cast<int64_t>(x1 + 1024L*x0));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(ks0); x0+=static_cast<int64_t>(16L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(16L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(16L))))))
                    {
                        auto tmp0 = at::vec::Vectorized<int32_t>::loadu(out_ptr5 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
                        auto tmp1 = static_cast<int32_t>(-1073741825);
                        auto tmp2 = at::vec::Vectorized<int32_t>(tmp1);
                        auto tmp3 = tmp0 & tmp2;
                        tmp3.store(out_ptr5 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
                    }
                    if(C10_UNLIKELY(x0 >= static_cast<int64_t>(16L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(16L)))) && x0 < static_cast<int64_t>(ks0)))
                    {
                        auto tmp0 = at::vec::Vectorized<int32_t>::loadu(out_ptr5 + static_cast<int64_t>(x0), static_cast<int64_t>(ks0 + (-16L)*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(16L)))));
                        auto tmp1 = static_cast<int32_t>(-1073741825);
                        auto tmp2 = at::vec::Vectorized<int32_t>(tmp1);
                        auto tmp3 = tmp0 & tmp2;
                        tmp3.store(out_ptr5 + static_cast<int64_t>(x0), static_cast<int64_t>(ks0 + (-16L)*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(16L)))));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_1 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'const int64_t'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       const int64_t ks0)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(ks0); x0+=static_cast<int64_t>(1L))
            {
                {
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    static WelfordHelper<float, 4096> scalar_welford_helper0(static_cast<int64_t>(1024L));
                    static WelfordHelper<at::vec::Vectorized<float>, 4096> welford_helper0(static_cast<int64_t>(64L));
                    static WelfordHelper<at::vec::Vectorized<float>, 4096> masked_welford_helper0(static_cast<int64_t>(0L));
                    for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(1024L); x1+=static_cast<int64_t>(16L))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(1024L)))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 1024L*x0), static_cast<int64_t>(16));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 1024L*x0), static_cast<int64_t>(16));
                                auto tmp2 = tmp0 + tmp1;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2, &welford_helper0);
                            }
                        }
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, &scalar_welford_helper0);
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, &welford_helper0);
                    masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, &masked_welford_helper0);
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(1024L); x1+=static_cast<int64_t>(16L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(1024L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 1024L*x0), static_cast<int64_t>(16));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 1024L*x0), static_cast<int64_t>(16));
                            auto tmp3 = out_ptr0[static_cast<int64_t>(x0)];
                            auto tmp6 = out_ptr1[static_cast<int64_t>(x0)];
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(16));
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(16));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 - tmp4;
                            auto tmp7 = static_cast<float>(1024.0);
                            auto tmp8 = tmp6 / tmp7;
                            auto tmp9 = static_cast<float>(1e-05);
                            auto tmp10 = float(tmp8 + tmp9);
                            auto tmp11 = 1 / std::sqrt(tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp5 * tmp12;
                            auto tmp15 = tmp13 * tmp14;
                            auto tmp17 = tmp15 + tmp16;
                            tmp17.store(in_out_ptr0 + static_cast<int64_t>(x1 + 1024L*x0));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_onednn_mm_2 = async_compile.cpp_pybinding(['const float*', 'float*', 'const int64_t'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       float* out_ptr0,
                       const int64_t ks0)
{
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(4096L*ks0); x0+=static_cast<int64_t>(16L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(4096L*ks0)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.7071067811865476);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp0 * tmp5;
                        auto tmp7 = tmp6.erf();
                        auto tmp8 = static_cast<float>(1.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 + tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        tmp11.store(out_ptr0 + static_cast<int64_t>(x0));
                    }
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1 = args
        args.clear()
        s72 = arg0_1
        s80 = arg3_1
        buf0 = empty_strided_cpu((s72, 1024), (1024, 1), torch.float32)
        buf1 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf2 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf5 = empty_strided_cpu((s72, 1024), (1024, 1), torch.float32)
        buf4 = empty_strided_cpu((s72, 3072), (3072, 1), torch.float32)
        cpp_fused___rshift____to_copy_add_bitwise_and_copy__embedding_native_layer_norm_0(arg1_1, arg2_1, arg6_1, arg4_1, arg5_1, arg7_1, arg8_1, buf0, buf1, buf2, buf5, arg1_1, s72)
        del arg1_1
        del arg2_1
        del arg4_1
        del arg5_1
        del arg6_1
        del arg7_1
        del arg8_1
        # Topologically Sorted Source Nodes: [onednn_mm], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf4, buf5, arg9_1, arg10_1)
        del arg10_1
        del arg9_1
        buf8 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
        buf9 = torch.ops.vllm.unified_kv_cache_update.default(reinterpret_tensor(buf4, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf4, (s72, 16, 64), (3072, 64, 1), 2048), arg11_1)
        buf10 = buf9
        del buf9
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep, unified_attention_with_output, output_2], Original ATen: [aten.split_with_sizes, aten.view]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf4, (s72, 16, 64), (3072, 64, 1), 0), reinterpret_tensor(buf4, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf4, (s72, 16, 64), (3072, 64, 1), 2048), reinterpret_tensor(buf8, (s72, 16, 64), (1024, 64, 1), 0), arg11_1, None, None, buf10)
        del arg11_1
        del buf10
        del buf4
        buf13 = empty_strided_cpu((s72, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_2, unified_attention_with_output, onednn_mm_1], Original ATen: [aten.view]
        torch.ops._C.onednn_mm.default(buf13, buf8, arg12_1, arg13_1)
        del arg12_1
        del arg13_1
        del buf8
        buf16 = buf2; del buf2  # reuse
        buf17 = buf1; del buf1  # reuse
        buf20 = buf5; del buf5  # reuse
        buf19 = empty_strided_cpu((s72, 4096), (4096, 1), torch.float32)
        cpp_fused_add_native_layer_norm_1(buf20, buf13, arg14_1, arg15_1, buf16, buf17, s72)
        del arg14_1
        del arg15_1
        del buf16
        del buf17
        # Topologically Sorted Source Nodes: [onednn_mm_2], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf19, buf20, arg16_1, arg17_1)
        del arg16_1
        del arg17_1
        buf23 = buf13; del buf13  # reuse
        buf24 = empty_strided_cpu((s72, 4096), (4096, 1), torch.float32)
        cpp_fused_gelu_onednn_mm_2(buf19, buf24, s72)
        del buf19
        # Topologically Sorted Source Nodes: [hidden_states_1, onednn_mm_3], Original ATen: [aten.gelu, _C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf23, buf24, arg18_1, arg19_1)
        del arg18_1
        del arg19_1
        buf27 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf28 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf31 = buf20; del buf20  # reuse
        buf30 = empty_strided_cpu((s72, 3072), (3072, 1), torch.float32)
        cpp_fused_add_native_layer_norm_1(buf31, buf23, arg20_1, arg21_1, buf27, buf28, s72)
        del arg20_1
        del arg21_1
        # Topologically Sorted Source Nodes: [onednn_mm_4], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf30, buf31, arg22_1, arg23_1)
        del arg22_1
        del arg23_1
        buf34 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_1], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
        buf35 = torch.ops.vllm.unified_kv_cache_update.default(reinterpret_tensor(buf30, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf30, (s72, 16, 64), (3072, 64, 1), 2048), arg24_1)
        buf36 = buf35
        del buf35
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_1, unified_attention_with_output_1, output_9], Original ATen: [aten.split_with_sizes, aten.view]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf30, (s72, 16, 64), (3072, 64, 1), 0), reinterpret_tensor(buf30, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf30, (s72, 16, 64), (3072, 64, 1), 2048), reinterpret_tensor(buf34, (s72, 16, 64), (1024, 64, 1), 0), arg24_1, None, None, buf36)
        del arg24_1
        del buf30
        del buf36
        buf39 = empty_strided_cpu((s72, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_9, unified_attention_with_output_1, onednn_mm_5], Original ATen: [aten.view]
        torch.ops._C.onednn_mm.default(buf39, buf34, arg25_1, arg26_1)
        del arg25_1
        del arg26_1
        del buf34
        buf42 = buf28; del buf28  # reuse
        buf43 = buf27; del buf27  # reuse
        buf46 = buf31; del buf31  # reuse
        buf45 = buf24; del buf24  # reuse
        cpp_fused_add_native_layer_norm_1(buf46, buf39, arg27_1, arg28_1, buf42, buf43, s72)
        del arg27_1
        del arg28_1
        del buf42
        del buf43
        # Topologically Sorted Source Nodes: [onednn_mm_6], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf45, buf46, arg29_1, arg30_1)
        del arg29_1
        del arg30_1
        buf49 = buf39; del buf39  # reuse
        buf50 = empty_strided_cpu((s72, 4096), (4096, 1), torch.float32)
        cpp_fused_gelu_onednn_mm_2(buf45, buf50, s72)
        del buf45
        # Topologically Sorted Source Nodes: [hidden_states_4, onednn_mm_7], Original ATen: [aten.gelu, _C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf49, buf50, arg31_1, arg32_1)
        del arg31_1
        del arg32_1
        buf53 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf54 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf57 = buf46; del buf46  # reuse
        buf56 = empty_strided_cpu((s72, 3072), (3072, 1), torch.float32)
        cpp_fused_add_native_layer_norm_1(buf57, buf49, arg33_1, arg34_1, buf53, buf54, s72)
        del arg33_1
        del arg34_1
        # Topologically Sorted Source Nodes: [onednn_mm_8], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf56, buf57, arg35_1, arg36_1)
        del arg35_1
        del arg36_1
        buf60 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_2], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
        buf61 = torch.ops.vllm.unified_kv_cache_update.default(reinterpret_tensor(buf56, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf56, (s72, 16, 64), (3072, 64, 1), 2048), arg37_1)
        buf62 = buf61
        del buf61
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_2, unified_attention_with_output_2, output_16], Original ATen: [aten.split_with_sizes, aten.view]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf56, (s72, 16, 64), (3072, 64, 1), 0), reinterpret_tensor(buf56, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf56, (s72, 16, 64), (3072, 64, 1), 2048), reinterpret_tensor(buf60, (s72, 16, 64), (1024, 64, 1), 0), arg37_1, None, None, buf62)
        del arg37_1
        del buf56
        del buf62
        buf65 = empty_strided_cpu((s72, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_16, unified_attention_with_output_2, onednn_mm_9], Original ATen: [aten.view]
        torch.ops._C.onednn_mm.default(buf65, buf60, arg38_1, arg39_1)
        del arg38_1
        del arg39_1
        del buf60
        buf68 = buf54; del buf54  # reuse
        buf69 = buf53; del buf53  # reuse
        buf72 = buf57; del buf57  # reuse
        buf71 = buf50; del buf50  # reuse
        cpp_fused_add_native_layer_norm_1(buf72, buf65, arg40_1, arg41_1, buf68, buf69, s72)
        del arg40_1
        del arg41_1
        del buf68
        del buf69
        # Topologically Sorted Source Nodes: [onednn_mm_10], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf71, buf72, arg42_1, arg43_1)
        del arg42_1
        del arg43_1
        buf75 = buf65; del buf65  # reuse
        buf76 = empty_strided_cpu((s72, 4096), (4096, 1), torch.float32)
        cpp_fused_gelu_onednn_mm_2(buf71, buf76, s72)
        del buf71
        # Topologically Sorted Source Nodes: [hidden_states_7, onednn_mm_11], Original ATen: [aten.gelu, _C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf75, buf76, arg44_1, arg45_1)
        del arg44_1
        del arg45_1
        buf79 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf80 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf83 = buf72; del buf72  # reuse
        buf82 = empty_strided_cpu((s72, 3072), (3072, 1), torch.float32)
        cpp_fused_add_native_layer_norm_1(buf83, buf75, arg46_1, arg47_1, buf79, buf80, s72)
        del arg46_1
        del arg47_1
        # Topologically Sorted Source Nodes: [onednn_mm_12], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf82, buf83, arg48_1, arg49_1)
        del arg48_1
        del arg49_1
        buf86 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_3], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
        buf87 = torch.ops.vllm.unified_kv_cache_update.default(reinterpret_tensor(buf82, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf82, (s72, 16, 64), (3072, 64, 1), 2048), arg50_1)
        buf88 = buf87
        del buf87
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_3, unified_attention_with_output_3, output_23], Original ATen: [aten.split_with_sizes, aten.view]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf82, (s72, 16, 64), (3072, 64, 1), 0), reinterpret_tensor(buf82, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf82, (s72, 16, 64), (3072, 64, 1), 2048), reinterpret_tensor(buf86, (s72, 16, 64), (1024, 64, 1), 0), arg50_1, None, None, buf88)
        del arg50_1
        del buf82
        del buf88
        buf91 = empty_strided_cpu((s72, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_23, unified_attention_with_output_3, onednn_mm_13], Original ATen: [aten.view]
        torch.ops._C.onednn_mm.default(buf91, buf86, arg51_1, arg52_1)
        del arg51_1
        del arg52_1
        del buf86
        buf94 = buf80; del buf80  # reuse
        buf95 = buf79; del buf79  # reuse
        buf98 = buf83; del buf83  # reuse
        buf97 = buf76; del buf76  # reuse
        cpp_fused_add_native_layer_norm_1(buf98, buf91, arg53_1, arg54_1, buf94, buf95, s72)
        del arg53_1
        del arg54_1
        del buf94
        del buf95
        # Topologically Sorted Source Nodes: [onednn_mm_14], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf97, buf98, arg55_1, arg56_1)
        del arg55_1
        del arg56_1
        buf101 = buf91; del buf91  # reuse
        buf102 = empty_strided_cpu((s72, 4096), (4096, 1), torch.float32)
        cpp_fused_gelu_onednn_mm_2(buf97, buf102, s72)
        del buf97
        # Topologically Sorted Source Nodes: [hidden_states_10, onednn_mm_15], Original ATen: [aten.gelu, _C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf101, buf102, arg57_1, arg58_1)
        del arg57_1
        del arg58_1
        buf105 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf106 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf109 = buf98; del buf98  # reuse
        buf108 = empty_strided_cpu((s72, 3072), (3072, 1), torch.float32)
        cpp_fused_add_native_layer_norm_1(buf109, buf101, arg59_1, arg60_1, buf105, buf106, s72)
        del arg59_1
        del arg60_1
        # Topologically Sorted Source Nodes: [onednn_mm_16], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf108, buf109, arg61_1, arg62_1)
        del arg61_1
        del arg62_1
        buf112 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_4], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
        buf113 = torch.ops.vllm.unified_kv_cache_update.default(reinterpret_tensor(buf108, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf108, (s72, 16, 64), (3072, 64, 1), 2048), arg63_1)
        buf114 = buf113
        del buf113
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_4, unified_attention_with_output_4, output_30], Original ATen: [aten.split_with_sizes, aten.view]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf108, (s72, 16, 64), (3072, 64, 1), 0), reinterpret_tensor(buf108, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf108, (s72, 16, 64), (3072, 64, 1), 2048), reinterpret_tensor(buf112, (s72, 16, 64), (1024, 64, 1), 0), arg63_1, None, None, buf114)
        del arg63_1
        del buf108
        del buf114
        buf117 = empty_strided_cpu((s72, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_30, unified_attention_with_output_4, onednn_mm_17], Original ATen: [aten.view]
        torch.ops._C.onednn_mm.default(buf117, buf112, arg64_1, arg65_1)
        del arg64_1
        del arg65_1
        del buf112
        buf120 = buf106; del buf106  # reuse
        buf121 = buf105; del buf105  # reuse
        buf124 = buf109; del buf109  # reuse
        buf123 = buf102; del buf102  # reuse
        cpp_fused_add_native_layer_norm_1(buf124, buf117, arg66_1, arg67_1, buf120, buf121, s72)
        del arg66_1
        del arg67_1
        del buf120
        del buf121
        # Topologically Sorted Source Nodes: [onednn_mm_18], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf123, buf124, arg68_1, arg69_1)
        del arg68_1
        del arg69_1
        buf127 = buf117; del buf117  # reuse
        buf128 = empty_strided_cpu((s72, 4096), (4096, 1), torch.float32)
        cpp_fused_gelu_onednn_mm_2(buf123, buf128, s72)
        del buf123
        # Topologically Sorted Source Nodes: [hidden_states_13, onednn_mm_19], Original ATen: [aten.gelu, _C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf127, buf128, arg70_1, arg71_1)
        del arg70_1
        del arg71_1
        buf131 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf132 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf135 = buf124; del buf124  # reuse
        buf134 = empty_strided_cpu((s72, 3072), (3072, 1), torch.float32)
        cpp_fused_add_native_layer_norm_1(buf135, buf127, arg72_1, arg73_1, buf131, buf132, s72)
        del arg72_1
        del arg73_1
        # Topologically Sorted Source Nodes: [onednn_mm_20], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf134, buf135, arg74_1, arg75_1)
        del arg74_1
        del arg75_1
        buf138 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_5], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
        buf139 = torch.ops.vllm.unified_kv_cache_update.default(reinterpret_tensor(buf134, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf134, (s72, 16, 64), (3072, 64, 1), 2048), arg76_1)
        buf140 = buf139
        del buf139
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_5, unified_attention_with_output_5, output_37], Original ATen: [aten.split_with_sizes, aten.view]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf134, (s72, 16, 64), (3072, 64, 1), 0), reinterpret_tensor(buf134, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf134, (s72, 16, 64), (3072, 64, 1), 2048), reinterpret_tensor(buf138, (s72, 16, 64), (1024, 64, 1), 0), arg76_1, None, None, buf140)
        del arg76_1
        del buf134
        del buf140
        buf143 = empty_strided_cpu((s72, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_37, unified_attention_with_output_5, onednn_mm_21], Original ATen: [aten.view]
        torch.ops._C.onednn_mm.default(buf143, buf138, arg77_1, arg78_1)
        del arg77_1
        del arg78_1
        del buf138
        buf146 = buf132; del buf132  # reuse
        buf147 = buf131; del buf131  # reuse
        buf150 = buf135; del buf135  # reuse
        buf149 = buf128; del buf128  # reuse
        cpp_fused_add_native_layer_norm_1(buf150, buf143, arg79_1, arg80_1, buf146, buf147, s72)
        del arg79_1
        del arg80_1
        del buf146
        del buf147
        # Topologically Sorted Source Nodes: [onednn_mm_22], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf149, buf150, arg81_1, arg82_1)
        del arg81_1
        del arg82_1
        buf153 = buf143; del buf143  # reuse
        buf154 = empty_strided_cpu((s72, 4096), (4096, 1), torch.float32)
        cpp_fused_gelu_onednn_mm_2(buf149, buf154, s72)
        del buf149
        # Topologically Sorted Source Nodes: [hidden_states_16, onednn_mm_23], Original ATen: [aten.gelu, _C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf153, buf154, arg83_1, arg84_1)
        del arg83_1
        del arg84_1
        buf157 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf158 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf161 = buf150; del buf150  # reuse
        buf160 = empty_strided_cpu((s72, 3072), (3072, 1), torch.float32)
        cpp_fused_add_native_layer_norm_1(buf161, buf153, arg85_1, arg86_1, buf157, buf158, s72)
        del arg85_1
        del arg86_1
        # Topologically Sorted Source Nodes: [onednn_mm_24], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf160, buf161, arg87_1, arg88_1)
        del arg87_1
        del arg88_1
        buf164 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_6], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
        buf165 = torch.ops.vllm.unified_kv_cache_update.default(reinterpret_tensor(buf160, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf160, (s72, 16, 64), (3072, 64, 1), 2048), arg89_1)
        buf166 = buf165
        del buf165
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_6, unified_attention_with_output_6, output_44], Original ATen: [aten.split_with_sizes, aten.view]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf160, (s72, 16, 64), (3072, 64, 1), 0), reinterpret_tensor(buf160, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf160, (s72, 16, 64), (3072, 64, 1), 2048), reinterpret_tensor(buf164, (s72, 16, 64), (1024, 64, 1), 0), arg89_1, None, None, buf166)
        del arg89_1
        del buf160
        del buf166
        buf169 = empty_strided_cpu((s72, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_44, unified_attention_with_output_6, onednn_mm_25], Original ATen: [aten.view]
        torch.ops._C.onednn_mm.default(buf169, buf164, arg90_1, arg91_1)
        del arg90_1
        del arg91_1
        del buf164
        buf172 = buf158; del buf158  # reuse
        buf173 = buf157; del buf157  # reuse
        buf176 = buf161; del buf161  # reuse
        buf175 = buf154; del buf154  # reuse
        cpp_fused_add_native_layer_norm_1(buf176, buf169, arg92_1, arg93_1, buf172, buf173, s72)
        del arg92_1
        del arg93_1
        del buf172
        del buf173
        # Topologically Sorted Source Nodes: [onednn_mm_26], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf175, buf176, arg94_1, arg95_1)
        del arg94_1
        del arg95_1
        buf179 = buf169; del buf169  # reuse
        buf180 = empty_strided_cpu((s72, 4096), (4096, 1), torch.float32)
        cpp_fused_gelu_onednn_mm_2(buf175, buf180, s72)
        del buf175
        # Topologically Sorted Source Nodes: [hidden_states_19, onednn_mm_27], Original ATen: [aten.gelu, _C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf179, buf180, arg96_1, arg97_1)
        del arg96_1
        del arg97_1
        buf183 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf184 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf187 = buf176; del buf176  # reuse
        buf186 = empty_strided_cpu((s72, 3072), (3072, 1), torch.float32)
        cpp_fused_add_native_layer_norm_1(buf187, buf179, arg98_1, arg99_1, buf183, buf184, s72)
        del arg98_1
        del arg99_1
        # Topologically Sorted Source Nodes: [onednn_mm_28], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf186, buf187, arg100_1, arg101_1)
        del arg100_1
        del arg101_1
        buf190 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_7], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
        buf191 = torch.ops.vllm.unified_kv_cache_update.default(reinterpret_tensor(buf186, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf186, (s72, 16, 64), (3072, 64, 1), 2048), arg102_1)
        buf192 = buf191
        del buf191
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_7, unified_attention_with_output_7, output_51], Original ATen: [aten.split_with_sizes, aten.view]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf186, (s72, 16, 64), (3072, 64, 1), 0), reinterpret_tensor(buf186, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf186, (s72, 16, 64), (3072, 64, 1), 2048), reinterpret_tensor(buf190, (s72, 16, 64), (1024, 64, 1), 0), arg102_1, None, None, buf192)
        del arg102_1
        del buf186
        del buf192
        buf195 = empty_strided_cpu((s72, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_51, unified_attention_with_output_7, onednn_mm_29], Original ATen: [aten.view]
        torch.ops._C.onednn_mm.default(buf195, buf190, arg103_1, arg104_1)
        del arg103_1
        del arg104_1
        del buf190
        buf198 = buf184; del buf184  # reuse
        buf199 = buf183; del buf183  # reuse
        buf202 = buf187; del buf187  # reuse
        buf201 = buf180; del buf180  # reuse
        cpp_fused_add_native_layer_norm_1(buf202, buf195, arg105_1, arg106_1, buf198, buf199, s72)
        del arg105_1
        del arg106_1
        del buf198
        del buf199
        # Topologically Sorted Source Nodes: [onednn_mm_30], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf201, buf202, arg107_1, arg108_1)
        del arg107_1
        del arg108_1
        buf205 = buf195; del buf195  # reuse
        buf206 = empty_strided_cpu((s72, 4096), (4096, 1), torch.float32)
        cpp_fused_gelu_onednn_mm_2(buf201, buf206, s72)
        del buf201
        # Topologically Sorted Source Nodes: [hidden_states_22, onednn_mm_31], Original ATen: [aten.gelu, _C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf205, buf206, arg109_1, arg110_1)
        del arg109_1
        del arg110_1
        buf209 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf210 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf213 = buf202; del buf202  # reuse
        buf212 = empty_strided_cpu((s72, 3072), (3072, 1), torch.float32)
        cpp_fused_add_native_layer_norm_1(buf213, buf205, arg111_1, arg112_1, buf209, buf210, s72)
        del arg111_1
        del arg112_1
        # Topologically Sorted Source Nodes: [onednn_mm_32], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf212, buf213, arg113_1, arg114_1)
        del arg113_1
        del arg114_1
        buf216 = buf205; del buf205  # reuse
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_8], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
        buf217 = torch.ops.vllm.unified_kv_cache_update.default(reinterpret_tensor(buf212, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf212, (s72, 16, 64), (3072, 64, 1), 2048), arg115_1)
        buf218 = buf217
        del buf217
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_8, unified_attention_with_output_8, output_58], Original ATen: [aten.split_with_sizes, aten.view]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf212, (s72, 16, 64), (3072, 64, 1), 0), reinterpret_tensor(buf212, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf212, (s72, 16, 64), (3072, 64, 1), 2048), reinterpret_tensor(buf216, (s72, 16, 64), (1024, 64, 1), 0), arg115_1, None, None, buf218)
        del arg115_1
        del buf212
        del buf218
        buf221 = empty_strided_cpu((s72, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_58, unified_attention_with_output_8, onednn_mm_33], Original ATen: [aten.view]
        torch.ops._C.onednn_mm.default(buf221, buf216, arg116_1, arg117_1)
        del arg116_1
        del arg117_1
        del buf216
        buf224 = buf210; del buf210  # reuse
        buf225 = buf209; del buf209  # reuse
        buf228 = buf213; del buf213  # reuse
        buf227 = buf206; del buf206  # reuse
        cpp_fused_add_native_layer_norm_1(buf228, buf221, arg118_1, arg119_1, buf224, buf225, s72)
        del arg118_1
        del arg119_1
        del buf224
        del buf225
        # Topologically Sorted Source Nodes: [onednn_mm_34], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf227, buf228, arg120_1, arg121_1)
        del arg120_1
        del arg121_1
        buf231 = buf221; del buf221  # reuse
        buf232 = empty_strided_cpu((s72, 4096), (4096, 1), torch.float32)
        cpp_fused_gelu_onednn_mm_2(buf227, buf232, s72)
        del buf227
        # Topologically Sorted Source Nodes: [hidden_states_25, onednn_mm_35], Original ATen: [aten.gelu, _C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf231, buf232, arg122_1, arg123_1)
        del arg122_1
        del arg123_1
        buf235 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf236 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf239 = buf228; del buf228  # reuse
        buf238 = empty_strided_cpu((s72, 3072), (3072, 1), torch.float32)
        cpp_fused_add_native_layer_norm_1(buf239, buf231, arg124_1, arg125_1, buf235, buf236, s72)
        del arg124_1
        del arg125_1
        # Topologically Sorted Source Nodes: [onednn_mm_36], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf238, buf239, arg126_1, arg127_1)
        del arg126_1
        del arg127_1
        buf242 = buf231; del buf231  # reuse
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_9], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
        buf243 = torch.ops.vllm.unified_kv_cache_update.default(reinterpret_tensor(buf238, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf238, (s72, 16, 64), (3072, 64, 1), 2048), arg128_1)
        buf244 = buf243
        del buf243
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_9, unified_attention_with_output_9, output_65], Original ATen: [aten.split_with_sizes, aten.view]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf238, (s72, 16, 64), (3072, 64, 1), 0), reinterpret_tensor(buf238, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf238, (s72, 16, 64), (3072, 64, 1), 2048), reinterpret_tensor(buf242, (s72, 16, 64), (1024, 64, 1), 0), arg128_1, None, None, buf244)
        del arg128_1
        del buf238
        del buf244
        buf247 = empty_strided_cpu((s72, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_65, unified_attention_with_output_9, onednn_mm_37], Original ATen: [aten.view]
        torch.ops._C.onednn_mm.default(buf247, buf242, arg129_1, arg130_1)
        del arg129_1
        del arg130_1
        del buf242
        buf250 = buf236; del buf236  # reuse
        buf251 = buf235; del buf235  # reuse
        buf254 = buf239; del buf239  # reuse
        buf253 = buf232; del buf232  # reuse
        cpp_fused_add_native_layer_norm_1(buf254, buf247, arg131_1, arg132_1, buf250, buf251, s72)
        del arg131_1
        del arg132_1
        del buf250
        del buf251
        # Topologically Sorted Source Nodes: [onednn_mm_38], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf253, buf254, arg133_1, arg134_1)
        del arg133_1
        del arg134_1
        buf257 = buf247; del buf247  # reuse
        buf258 = empty_strided_cpu((s72, 4096), (4096, 1), torch.float32)
        cpp_fused_gelu_onednn_mm_2(buf253, buf258, s72)
        del buf253
        # Topologically Sorted Source Nodes: [hidden_states_28, onednn_mm_39], Original ATen: [aten.gelu, _C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf257, buf258, arg135_1, arg136_1)
        del arg135_1
        del arg136_1
        buf261 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf262 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf265 = buf254; del buf254  # reuse
        buf264 = empty_strided_cpu((s72, 3072), (3072, 1), torch.float32)
        cpp_fused_add_native_layer_norm_1(buf265, buf257, arg137_1, arg138_1, buf261, buf262, s72)
        del arg137_1
        del arg138_1
        # Topologically Sorted Source Nodes: [onednn_mm_40], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf264, buf265, arg139_1, arg140_1)
        del arg139_1
        del arg140_1
        buf268 = buf257; del buf257  # reuse
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_10], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
        buf269 = torch.ops.vllm.unified_kv_cache_update.default(reinterpret_tensor(buf264, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf264, (s72, 16, 64), (3072, 64, 1), 2048), arg141_1)
        buf270 = buf269
        del buf269
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_10, unified_attention_with_output_10, output_72], Original ATen: [aten.split_with_sizes, aten.view]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf264, (s72, 16, 64), (3072, 64, 1), 0), reinterpret_tensor(buf264, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf264, (s72, 16, 64), (3072, 64, 1), 2048), reinterpret_tensor(buf268, (s72, 16, 64), (1024, 64, 1), 0), arg141_1, None, None, buf270)
        del arg141_1
        del buf264
        del buf270
        buf273 = empty_strided_cpu((s72, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_72, unified_attention_with_output_10, onednn_mm_41], Original ATen: [aten.view]
        torch.ops._C.onednn_mm.default(buf273, buf268, arg142_1, arg143_1)
        del arg142_1
        del arg143_1
        del buf268
        buf276 = buf262; del buf262  # reuse
        buf277 = buf261; del buf261  # reuse
        buf280 = buf265; del buf265  # reuse
        buf279 = buf258; del buf258  # reuse
        cpp_fused_add_native_layer_norm_1(buf280, buf273, arg144_1, arg145_1, buf276, buf277, s72)
        del arg144_1
        del arg145_1
        del buf276
        del buf277
        # Topologically Sorted Source Nodes: [onednn_mm_42], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf279, buf280, arg146_1, arg147_1)
        del arg146_1
        del arg147_1
        buf283 = buf273; del buf273  # reuse
        buf284 = empty_strided_cpu((s72, 4096), (4096, 1), torch.float32)
        cpp_fused_gelu_onednn_mm_2(buf279, buf284, s72)
        del buf279
        # Topologically Sorted Source Nodes: [hidden_states_31, onednn_mm_43], Original ATen: [aten.gelu, _C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf283, buf284, arg148_1, arg149_1)
        del arg148_1
        del arg149_1
        buf287 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf288 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf291 = buf280; del buf280  # reuse
        buf290 = empty_strided_cpu((s72, 3072), (3072, 1), torch.float32)
        cpp_fused_add_native_layer_norm_1(buf291, buf283, arg150_1, arg151_1, buf287, buf288, s72)
        del arg150_1
        del arg151_1
        # Topologically Sorted Source Nodes: [onednn_mm_44], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf290, buf291, arg152_1, arg153_1)
        del arg152_1
        del arg153_1
        buf294 = buf283; del buf283  # reuse
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_11], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
        buf295 = torch.ops.vllm.unified_kv_cache_update.default(reinterpret_tensor(buf290, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf290, (s72, 16, 64), (3072, 64, 1), 2048), arg154_1)
        buf296 = buf295
        del buf295
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_11, unified_attention_with_output_11, output_79], Original ATen: [aten.split_with_sizes, aten.view]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf290, (s72, 16, 64), (3072, 64, 1), 0), reinterpret_tensor(buf290, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf290, (s72, 16, 64), (3072, 64, 1), 2048), reinterpret_tensor(buf294, (s72, 16, 64), (1024, 64, 1), 0), arg154_1, None, None, buf296)
        del arg154_1
        del buf290
        del buf296
        buf299 = empty_strided_cpu((s72, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_79, unified_attention_with_output_11, onednn_mm_45], Original ATen: [aten.view]
        torch.ops._C.onednn_mm.default(buf299, buf294, arg155_1, arg156_1)
        del arg155_1
        del arg156_1
        del buf294
        buf302 = buf288; del buf288  # reuse
        buf303 = buf287; del buf287  # reuse
        buf306 = buf291; del buf291  # reuse
        buf305 = buf284; del buf284  # reuse
        cpp_fused_add_native_layer_norm_1(buf306, buf299, arg157_1, arg158_1, buf302, buf303, s72)
        del arg157_1
        del arg158_1
        del buf302
        del buf303
        # Topologically Sorted Source Nodes: [onednn_mm_46], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf305, buf306, arg159_1, arg160_1)
        del arg159_1
        del arg160_1
        buf309 = buf299; del buf299  # reuse
        buf310 = empty_strided_cpu((s72, 4096), (4096, 1), torch.float32)
        cpp_fused_gelu_onednn_mm_2(buf305, buf310, s72)
        del buf305
        # Topologically Sorted Source Nodes: [hidden_states_34, onednn_mm_47], Original ATen: [aten.gelu, _C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf309, buf310, arg161_1, arg162_1)
        del arg161_1
        del arg162_1
        buf313 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf314 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf317 = buf306; del buf306  # reuse
        buf316 = empty_strided_cpu((s72, 3072), (3072, 1), torch.float32)
        cpp_fused_add_native_layer_norm_1(buf317, buf309, arg163_1, arg164_1, buf313, buf314, s72)
        del arg163_1
        del arg164_1
        # Topologically Sorted Source Nodes: [onednn_mm_48], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf316, buf317, arg165_1, arg166_1)
        del arg165_1
        del arg166_1
        buf320 = buf309; del buf309  # reuse
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_12], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
        buf321 = torch.ops.vllm.unified_kv_cache_update.default(reinterpret_tensor(buf316, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf316, (s72, 16, 64), (3072, 64, 1), 2048), arg167_1)
        buf322 = buf321
        del buf321
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_12, unified_attention_with_output_12, output_86], Original ATen: [aten.split_with_sizes, aten.view]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf316, (s72, 16, 64), (3072, 64, 1), 0), reinterpret_tensor(buf316, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf316, (s72, 16, 64), (3072, 64, 1), 2048), reinterpret_tensor(buf320, (s72, 16, 64), (1024, 64, 1), 0), arg167_1, None, None, buf322)
        del arg167_1
        del buf316
        del buf322
        buf325 = empty_strided_cpu((s72, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_86, unified_attention_with_output_12, onednn_mm_49], Original ATen: [aten.view]
        torch.ops._C.onednn_mm.default(buf325, buf320, arg168_1, arg169_1)
        del arg168_1
        del arg169_1
        del buf320
        buf328 = buf314; del buf314  # reuse
        buf329 = buf313; del buf313  # reuse
        buf332 = buf317; del buf317  # reuse
        buf331 = buf310; del buf310  # reuse
        cpp_fused_add_native_layer_norm_1(buf332, buf325, arg170_1, arg171_1, buf328, buf329, s72)
        del arg170_1
        del arg171_1
        del buf328
        del buf329
        # Topologically Sorted Source Nodes: [onednn_mm_50], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf331, buf332, arg172_1, arg173_1)
        del arg172_1
        del arg173_1
        buf335 = buf325; del buf325  # reuse
        buf336 = empty_strided_cpu((s72, 4096), (4096, 1), torch.float32)
        cpp_fused_gelu_onednn_mm_2(buf331, buf336, s72)
        del buf331
        # Topologically Sorted Source Nodes: [hidden_states_37, onednn_mm_51], Original ATen: [aten.gelu, _C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf335, buf336, arg174_1, arg175_1)
        del arg174_1
        del arg175_1
        buf339 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf340 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf343 = buf332; del buf332  # reuse
        buf342 = empty_strided_cpu((s72, 3072), (3072, 1), torch.float32)
        cpp_fused_add_native_layer_norm_1(buf343, buf335, arg176_1, arg177_1, buf339, buf340, s72)
        del arg176_1
        del arg177_1
        # Topologically Sorted Source Nodes: [onednn_mm_52], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf342, buf343, arg178_1, arg179_1)
        del arg178_1
        del arg179_1
        buf346 = buf335; del buf335  # reuse
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_13], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
        buf347 = torch.ops.vllm.unified_kv_cache_update.default(reinterpret_tensor(buf342, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf342, (s72, 16, 64), (3072, 64, 1), 2048), arg180_1)
        buf348 = buf347
        del buf347
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_13, unified_attention_with_output_13, output_93], Original ATen: [aten.split_with_sizes, aten.view]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf342, (s72, 16, 64), (3072, 64, 1), 0), reinterpret_tensor(buf342, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf342, (s72, 16, 64), (3072, 64, 1), 2048), reinterpret_tensor(buf346, (s72, 16, 64), (1024, 64, 1), 0), arg180_1, None, None, buf348)
        del arg180_1
        del buf342
        del buf348
        buf351 = empty_strided_cpu((s72, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_93, unified_attention_with_output_13, onednn_mm_53], Original ATen: [aten.view]
        torch.ops._C.onednn_mm.default(buf351, buf346, arg181_1, arg182_1)
        del arg181_1
        del arg182_1
        del buf346
        buf354 = buf340; del buf340  # reuse
        buf355 = buf339; del buf339  # reuse
        buf358 = buf343; del buf343  # reuse
        buf357 = buf336; del buf336  # reuse
        cpp_fused_add_native_layer_norm_1(buf358, buf351, arg183_1, arg184_1, buf354, buf355, s72)
        del arg183_1
        del arg184_1
        del buf354
        del buf355
        # Topologically Sorted Source Nodes: [onednn_mm_54], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf357, buf358, arg185_1, arg186_1)
        del arg185_1
        del arg186_1
        buf361 = buf351; del buf351  # reuse
        buf362 = empty_strided_cpu((s72, 4096), (4096, 1), torch.float32)
        cpp_fused_gelu_onednn_mm_2(buf357, buf362, s72)
        del buf357
        # Topologically Sorted Source Nodes: [hidden_states_40, onednn_mm_55], Original ATen: [aten.gelu, _C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf361, buf362, arg187_1, arg188_1)
        del arg187_1
        del arg188_1
        buf365 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf366 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf369 = buf358; del buf358  # reuse
        buf368 = empty_strided_cpu((s72, 3072), (3072, 1), torch.float32)
        cpp_fused_add_native_layer_norm_1(buf369, buf361, arg189_1, arg190_1, buf365, buf366, s72)
        del arg189_1
        del arg190_1
        # Topologically Sorted Source Nodes: [onednn_mm_56], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf368, buf369, arg191_1, arg192_1)
        del arg191_1
        del arg192_1
        buf372 = buf361; del buf361  # reuse
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_14], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
        buf373 = torch.ops.vllm.unified_kv_cache_update.default(reinterpret_tensor(buf368, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf368, (s72, 16, 64), (3072, 64, 1), 2048), arg193_1)
        buf374 = buf373
        del buf373
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_14, unified_attention_with_output_14, output_100], Original ATen: [aten.split_with_sizes, aten.view]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf368, (s72, 16, 64), (3072, 64, 1), 0), reinterpret_tensor(buf368, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf368, (s72, 16, 64), (3072, 64, 1), 2048), reinterpret_tensor(buf372, (s72, 16, 64), (1024, 64, 1), 0), arg193_1, None, None, buf374)
        del arg193_1
        del buf368
        del buf374
        buf377 = empty_strided_cpu((s72, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_100, unified_attention_with_output_14, onednn_mm_57], Original ATen: [aten.view]
        torch.ops._C.onednn_mm.default(buf377, buf372, arg194_1, arg195_1)
        del arg194_1
        del arg195_1
        del buf372
        buf380 = buf366; del buf366  # reuse
        buf381 = buf365; del buf365  # reuse
        buf384 = buf369; del buf369  # reuse
        buf383 = buf362; del buf362  # reuse
        cpp_fused_add_native_layer_norm_1(buf384, buf377, arg196_1, arg197_1, buf380, buf381, s72)
        del arg196_1
        del arg197_1
        del buf380
        del buf381
        # Topologically Sorted Source Nodes: [onednn_mm_58], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf383, buf384, arg198_1, arg199_1)
        del arg198_1
        del arg199_1
        buf387 = buf377; del buf377  # reuse
        buf388 = empty_strided_cpu((s72, 4096), (4096, 1), torch.float32)
        cpp_fused_gelu_onednn_mm_2(buf383, buf388, s72)
        del buf383
        # Topologically Sorted Source Nodes: [hidden_states_43, onednn_mm_59], Original ATen: [aten.gelu, _C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf387, buf388, arg200_1, arg201_1)
        del arg200_1
        del arg201_1
        buf391 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf392 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf395 = buf384; del buf384  # reuse
        buf394 = empty_strided_cpu((s72, 3072), (3072, 1), torch.float32)
        cpp_fused_add_native_layer_norm_1(buf395, buf387, arg202_1, arg203_1, buf391, buf392, s72)
        del arg202_1
        del arg203_1
        # Topologically Sorted Source Nodes: [onednn_mm_60], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf394, buf395, arg204_1, arg205_1)
        del arg204_1
        del arg205_1
        buf398 = buf387; del buf387  # reuse
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_15], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
        buf399 = torch.ops.vllm.unified_kv_cache_update.default(reinterpret_tensor(buf394, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf394, (s72, 16, 64), (3072, 64, 1), 2048), arg206_1)
        buf400 = buf399
        del buf399
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_15, unified_attention_with_output_15, output_107], Original ATen: [aten.split_with_sizes, aten.view]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf394, (s72, 16, 64), (3072, 64, 1), 0), reinterpret_tensor(buf394, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf394, (s72, 16, 64), (3072, 64, 1), 2048), reinterpret_tensor(buf398, (s72, 16, 64), (1024, 64, 1), 0), arg206_1, None, None, buf400)
        del arg206_1
        del buf394
        del buf400
        buf403 = empty_strided_cpu((s72, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_107, unified_attention_with_output_15, onednn_mm_61], Original ATen: [aten.view]
        torch.ops._C.onednn_mm.default(buf403, buf398, arg207_1, arg208_1)
        del arg207_1
        del arg208_1
        del buf398
        buf406 = buf392; del buf392  # reuse
        buf407 = buf391; del buf391  # reuse
        buf410 = buf395; del buf395  # reuse
        buf409 = buf388; del buf388  # reuse
        cpp_fused_add_native_layer_norm_1(buf410, buf403, arg209_1, arg210_1, buf406, buf407, s72)
        del arg209_1
        del arg210_1
        del buf406
        del buf407
        # Topologically Sorted Source Nodes: [onednn_mm_62], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf409, buf410, arg211_1, arg212_1)
        del arg211_1
        del arg212_1
        buf413 = buf403; del buf403  # reuse
        buf414 = empty_strided_cpu((s72, 4096), (4096, 1), torch.float32)
        cpp_fused_gelu_onednn_mm_2(buf409, buf414, s72)
        del buf409
        # Topologically Sorted Source Nodes: [hidden_states_46, onednn_mm_63], Original ATen: [aten.gelu, _C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf413, buf414, arg213_1, arg214_1)
        del arg213_1
        del arg214_1
        buf417 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf418 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf421 = buf410; del buf410  # reuse
        buf420 = empty_strided_cpu((s72, 3072), (3072, 1), torch.float32)
        cpp_fused_add_native_layer_norm_1(buf421, buf413, arg215_1, arg216_1, buf417, buf418, s72)
        del arg215_1
        del arg216_1
        # Topologically Sorted Source Nodes: [onednn_mm_64], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf420, buf421, arg217_1, arg218_1)
        del arg217_1
        del arg218_1
        buf424 = buf413; del buf413  # reuse
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_16], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
        buf425 = torch.ops.vllm.unified_kv_cache_update.default(reinterpret_tensor(buf420, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf420, (s72, 16, 64), (3072, 64, 1), 2048), arg219_1)
        buf426 = buf425
        del buf425
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_16, unified_attention_with_output_16, output_114], Original ATen: [aten.split_with_sizes, aten.view]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf420, (s72, 16, 64), (3072, 64, 1), 0), reinterpret_tensor(buf420, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf420, (s72, 16, 64), (3072, 64, 1), 2048), reinterpret_tensor(buf424, (s72, 16, 64), (1024, 64, 1), 0), arg219_1, None, None, buf426)
        del arg219_1
        del buf420
        del buf426
        buf429 = empty_strided_cpu((s72, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_114, unified_attention_with_output_16, onednn_mm_65], Original ATen: [aten.view]
        torch.ops._C.onednn_mm.default(buf429, buf424, arg220_1, arg221_1)
        del arg220_1
        del arg221_1
        del buf424
        buf432 = buf418; del buf418  # reuse
        buf433 = buf417; del buf417  # reuse
        buf436 = buf421; del buf421  # reuse
        buf435 = buf414; del buf414  # reuse
        cpp_fused_add_native_layer_norm_1(buf436, buf429, arg222_1, arg223_1, buf432, buf433, s72)
        del arg222_1
        del arg223_1
        del buf432
        del buf433
        # Topologically Sorted Source Nodes: [onednn_mm_66], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf435, buf436, arg224_1, arg225_1)
        del arg224_1
        del arg225_1
        buf439 = buf429; del buf429  # reuse
        buf440 = empty_strided_cpu((s72, 4096), (4096, 1), torch.float32)
        cpp_fused_gelu_onednn_mm_2(buf435, buf440, s72)
        del buf435
        # Topologically Sorted Source Nodes: [hidden_states_49, onednn_mm_67], Original ATen: [aten.gelu, _C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf439, buf440, arg226_1, arg227_1)
        del arg226_1
        del arg227_1
        buf443 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf444 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf447 = buf436; del buf436  # reuse
        buf446 = empty_strided_cpu((s72, 3072), (3072, 1), torch.float32)
        cpp_fused_add_native_layer_norm_1(buf447, buf439, arg228_1, arg229_1, buf443, buf444, s72)
        del arg228_1
        del arg229_1
        # Topologically Sorted Source Nodes: [onednn_mm_68], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf446, buf447, arg230_1, arg231_1)
        del arg230_1
        del arg231_1
        buf450 = buf439; del buf439  # reuse
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_17], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
        buf451 = torch.ops.vllm.unified_kv_cache_update.default(reinterpret_tensor(buf446, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf446, (s72, 16, 64), (3072, 64, 1), 2048), arg232_1)
        buf452 = buf451
        del buf451
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_17, unified_attention_with_output_17, output_121], Original ATen: [aten.split_with_sizes, aten.view]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf446, (s72, 16, 64), (3072, 64, 1), 0), reinterpret_tensor(buf446, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf446, (s72, 16, 64), (3072, 64, 1), 2048), reinterpret_tensor(buf450, (s72, 16, 64), (1024, 64, 1), 0), arg232_1, None, None, buf452)
        del arg232_1
        del buf446
        del buf452
        buf455 = empty_strided_cpu((s72, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_121, unified_attention_with_output_17, onednn_mm_69], Original ATen: [aten.view]
        torch.ops._C.onednn_mm.default(buf455, buf450, arg233_1, arg234_1)
        del arg233_1
        del arg234_1
        del buf450
        buf458 = buf444; del buf444  # reuse
        buf459 = buf443; del buf443  # reuse
        buf462 = buf447; del buf447  # reuse
        buf461 = buf440; del buf440  # reuse
        cpp_fused_add_native_layer_norm_1(buf462, buf455, arg235_1, arg236_1, buf458, buf459, s72)
        del arg235_1
        del arg236_1
        del buf458
        del buf459
        # Topologically Sorted Source Nodes: [onednn_mm_70], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf461, buf462, arg237_1, arg238_1)
        del arg237_1
        del arg238_1
        buf465 = buf455; del buf455  # reuse
        buf466 = empty_strided_cpu((s72, 4096), (4096, 1), torch.float32)
        cpp_fused_gelu_onednn_mm_2(buf461, buf466, s72)
        del buf461
        # Topologically Sorted Source Nodes: [hidden_states_52, onednn_mm_71], Original ATen: [aten.gelu, _C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf465, buf466, arg239_1, arg240_1)
        del arg239_1
        del arg240_1
        buf469 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf470 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf473 = buf462; del buf462  # reuse
        buf472 = empty_strided_cpu((s72, 3072), (3072, 1), torch.float32)
        cpp_fused_add_native_layer_norm_1(buf473, buf465, arg241_1, arg242_1, buf469, buf470, s72)
        del arg241_1
        del arg242_1
        # Topologically Sorted Source Nodes: [onednn_mm_72], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf472, buf473, arg243_1, arg244_1)
        del arg243_1
        del arg244_1
        buf476 = buf465; del buf465  # reuse
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_18], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
        buf477 = torch.ops.vllm.unified_kv_cache_update.default(reinterpret_tensor(buf472, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf472, (s72, 16, 64), (3072, 64, 1), 2048), arg245_1)
        buf478 = buf477
        del buf477
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_18, unified_attention_with_output_18, output_128], Original ATen: [aten.split_with_sizes, aten.view]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf472, (s72, 16, 64), (3072, 64, 1), 0), reinterpret_tensor(buf472, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf472, (s72, 16, 64), (3072, 64, 1), 2048), reinterpret_tensor(buf476, (s72, 16, 64), (1024, 64, 1), 0), arg245_1, None, None, buf478)
        del arg245_1
        del buf472
        del buf478
        buf481 = empty_strided_cpu((s72, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_128, unified_attention_with_output_18, onednn_mm_73], Original ATen: [aten.view]
        torch.ops._C.onednn_mm.default(buf481, buf476, arg246_1, arg247_1)
        del arg246_1
        del arg247_1
        del buf476
        buf484 = buf470; del buf470  # reuse
        buf485 = buf469; del buf469  # reuse
        buf488 = buf473; del buf473  # reuse
        buf487 = buf466; del buf466  # reuse
        cpp_fused_add_native_layer_norm_1(buf488, buf481, arg248_1, arg249_1, buf484, buf485, s72)
        del arg248_1
        del arg249_1
        del buf484
        del buf485
        # Topologically Sorted Source Nodes: [onednn_mm_74], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf487, buf488, arg250_1, arg251_1)
        del arg250_1
        del arg251_1
        buf491 = buf481; del buf481  # reuse
        buf492 = empty_strided_cpu((s72, 4096), (4096, 1), torch.float32)
        cpp_fused_gelu_onednn_mm_2(buf487, buf492, s72)
        del buf487
        # Topologically Sorted Source Nodes: [hidden_states_55, onednn_mm_75], Original ATen: [aten.gelu, _C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf491, buf492, arg252_1, arg253_1)
        del arg252_1
        del arg253_1
        buf495 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf496 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf499 = buf488; del buf488  # reuse
        buf498 = empty_strided_cpu((s72, 3072), (3072, 1), torch.float32)
        cpp_fused_add_native_layer_norm_1(buf499, buf491, arg254_1, arg255_1, buf495, buf496, s72)
        del arg254_1
        del arg255_1
        # Topologically Sorted Source Nodes: [onednn_mm_76], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf498, buf499, arg256_1, arg257_1)
        del arg256_1
        del arg257_1
        buf502 = buf491; del buf491  # reuse
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_19], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
        buf503 = torch.ops.vllm.unified_kv_cache_update.default(reinterpret_tensor(buf498, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf498, (s72, 16, 64), (3072, 64, 1), 2048), arg258_1)
        buf504 = buf503
        del buf503
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_19, unified_attention_with_output_19, output_135], Original ATen: [aten.split_with_sizes, aten.view]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf498, (s72, 16, 64), (3072, 64, 1), 0), reinterpret_tensor(buf498, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf498, (s72, 16, 64), (3072, 64, 1), 2048), reinterpret_tensor(buf502, (s72, 16, 64), (1024, 64, 1), 0), arg258_1, None, None, buf504)
        del arg258_1
        del buf498
        del buf504
        buf507 = empty_strided_cpu((s72, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_135, unified_attention_with_output_19, onednn_mm_77], Original ATen: [aten.view]
        torch.ops._C.onednn_mm.default(buf507, buf502, arg259_1, arg260_1)
        del arg259_1
        del arg260_1
        del buf502
        buf510 = buf496; del buf496  # reuse
        buf511 = buf495; del buf495  # reuse
        buf514 = buf499; del buf499  # reuse
        buf513 = buf492; del buf492  # reuse
        cpp_fused_add_native_layer_norm_1(buf514, buf507, arg261_1, arg262_1, buf510, buf511, s72)
        del arg261_1
        del arg262_1
        del buf510
        del buf511
        # Topologically Sorted Source Nodes: [onednn_mm_78], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf513, buf514, arg263_1, arg264_1)
        del arg263_1
        del arg264_1
        buf517 = buf507; del buf507  # reuse
        buf518 = empty_strided_cpu((s72, 4096), (4096, 1), torch.float32)
        cpp_fused_gelu_onednn_mm_2(buf513, buf518, s72)
        del buf513
        # Topologically Sorted Source Nodes: [hidden_states_58, onednn_mm_79], Original ATen: [aten.gelu, _C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf517, buf518, arg265_1, arg266_1)
        del arg265_1
        del arg266_1
        buf521 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf522 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf525 = buf514; del buf514  # reuse
        buf524 = empty_strided_cpu((s72, 3072), (3072, 1), torch.float32)
        cpp_fused_add_native_layer_norm_1(buf525, buf517, arg267_1, arg268_1, buf521, buf522, s72)
        del arg267_1
        del arg268_1
        # Topologically Sorted Source Nodes: [onednn_mm_80], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf524, buf525, arg269_1, arg270_1)
        del arg269_1
        del arg270_1
        buf528 = buf517; del buf517  # reuse
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_20], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
        buf529 = torch.ops.vllm.unified_kv_cache_update.default(reinterpret_tensor(buf524, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf524, (s72, 16, 64), (3072, 64, 1), 2048), arg271_1)
        buf530 = buf529
        del buf529
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_20, unified_attention_with_output_20, output_142], Original ATen: [aten.split_with_sizes, aten.view]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf524, (s72, 16, 64), (3072, 64, 1), 0), reinterpret_tensor(buf524, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf524, (s72, 16, 64), (3072, 64, 1), 2048), reinterpret_tensor(buf528, (s72, 16, 64), (1024, 64, 1), 0), arg271_1, None, None, buf530)
        del arg271_1
        del buf524
        del buf530
        buf533 = empty_strided_cpu((s72, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_142, unified_attention_with_output_20, onednn_mm_81], Original ATen: [aten.view]
        torch.ops._C.onednn_mm.default(buf533, buf528, arg272_1, arg273_1)
        del arg272_1
        del arg273_1
        del buf528
        buf536 = buf522; del buf522  # reuse
        buf537 = buf521; del buf521  # reuse
        buf540 = buf525; del buf525  # reuse
        buf539 = buf518; del buf518  # reuse
        cpp_fused_add_native_layer_norm_1(buf540, buf533, arg274_1, arg275_1, buf536, buf537, s72)
        del arg274_1
        del arg275_1
        del buf536
        del buf537
        # Topologically Sorted Source Nodes: [onednn_mm_82], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf539, buf540, arg276_1, arg277_1)
        del arg276_1
        del arg277_1
        buf543 = buf533; del buf533  # reuse
        buf544 = empty_strided_cpu((s72, 4096), (4096, 1), torch.float32)
        cpp_fused_gelu_onednn_mm_2(buf539, buf544, s72)
        del buf539
        # Topologically Sorted Source Nodes: [hidden_states_61, onednn_mm_83], Original ATen: [aten.gelu, _C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf543, buf544, arg278_1, arg279_1)
        del arg278_1
        del arg279_1
        buf547 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf548 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf551 = buf540; del buf540  # reuse
        buf550 = empty_strided_cpu((s72, 3072), (3072, 1), torch.float32)
        cpp_fused_add_native_layer_norm_1(buf551, buf543, arg280_1, arg281_1, buf547, buf548, s72)
        del arg280_1
        del arg281_1
        # Topologically Sorted Source Nodes: [onednn_mm_84], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf550, buf551, arg282_1, arg283_1)
        del arg282_1
        del arg283_1
        buf554 = buf543; del buf543  # reuse
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_21], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
        buf555 = torch.ops.vllm.unified_kv_cache_update.default(reinterpret_tensor(buf550, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf550, (s72, 16, 64), (3072, 64, 1), 2048), arg284_1)
        buf556 = buf555
        del buf555
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_21, unified_attention_with_output_21, output_149], Original ATen: [aten.split_with_sizes, aten.view]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf550, (s72, 16, 64), (3072, 64, 1), 0), reinterpret_tensor(buf550, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf550, (s72, 16, 64), (3072, 64, 1), 2048), reinterpret_tensor(buf554, (s72, 16, 64), (1024, 64, 1), 0), arg284_1, None, None, buf556)
        del arg284_1
        del buf550
        del buf556
        buf559 = empty_strided_cpu((s72, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_149, unified_attention_with_output_21, onednn_mm_85], Original ATen: [aten.view]
        torch.ops._C.onednn_mm.default(buf559, buf554, arg285_1, arg286_1)
        del arg285_1
        del arg286_1
        del buf554
        buf562 = buf548; del buf548  # reuse
        buf563 = buf547; del buf547  # reuse
        buf566 = buf551; del buf551  # reuse
        buf565 = buf544; del buf544  # reuse
        cpp_fused_add_native_layer_norm_1(buf566, buf559, arg287_1, arg288_1, buf562, buf563, s72)
        del arg287_1
        del arg288_1
        del buf562
        del buf563
        # Topologically Sorted Source Nodes: [onednn_mm_86], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf565, buf566, arg289_1, arg290_1)
        del arg289_1
        del arg290_1
        buf569 = buf559; del buf559  # reuse
        buf570 = empty_strided_cpu((s72, 4096), (4096, 1), torch.float32)
        cpp_fused_gelu_onednn_mm_2(buf565, buf570, s72)
        del buf565
        # Topologically Sorted Source Nodes: [hidden_states_64, onednn_mm_87], Original ATen: [aten.gelu, _C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf569, buf570, arg291_1, arg292_1)
        del arg291_1
        del arg292_1
        buf573 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf574 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf577 = buf566; del buf566  # reuse
        buf576 = empty_strided_cpu((s72, 3072), (3072, 1), torch.float32)
        cpp_fused_add_native_layer_norm_1(buf577, buf569, arg293_1, arg294_1, buf573, buf574, s72)
        del arg293_1
        del arg294_1
        # Topologically Sorted Source Nodes: [onednn_mm_88], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf576, buf577, arg295_1, arg296_1)
        del arg295_1
        del arg296_1
        buf580 = buf569; del buf569  # reuse
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_22], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
        buf581 = torch.ops.vllm.unified_kv_cache_update.default(reinterpret_tensor(buf576, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf576, (s72, 16, 64), (3072, 64, 1), 2048), arg297_1)
        buf582 = buf581
        del buf581
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_22, unified_attention_with_output_22, output_156], Original ATen: [aten.split_with_sizes, aten.view]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf576, (s72, 16, 64), (3072, 64, 1), 0), reinterpret_tensor(buf576, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf576, (s72, 16, 64), (3072, 64, 1), 2048), reinterpret_tensor(buf580, (s72, 16, 64), (1024, 64, 1), 0), arg297_1, None, None, buf582)
        del arg297_1
        del buf576
        del buf582
        buf585 = empty_strided_cpu((s72, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_156, unified_attention_with_output_22, onednn_mm_89], Original ATen: [aten.view]
        torch.ops._C.onednn_mm.default(buf585, buf580, arg298_1, arg299_1)
        del arg298_1
        del arg299_1
        del buf580
        buf588 = buf574; del buf574  # reuse
        buf589 = buf573; del buf573  # reuse
        buf592 = buf577; del buf577  # reuse
        buf591 = buf570; del buf570  # reuse
        cpp_fused_add_native_layer_norm_1(buf592, buf585, arg300_1, arg301_1, buf588, buf589, s72)
        del arg300_1
        del arg301_1
        del buf588
        del buf589
        # Topologically Sorted Source Nodes: [onednn_mm_90], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf591, buf592, arg302_1, arg303_1)
        del arg302_1
        del arg303_1
        buf595 = buf585; del buf585  # reuse
        buf596 = empty_strided_cpu((s72, 4096), (4096, 1), torch.float32)
        cpp_fused_gelu_onednn_mm_2(buf591, buf596, s72)
        del buf591
        # Topologically Sorted Source Nodes: [hidden_states_67, onednn_mm_91], Original ATen: [aten.gelu, _C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf595, buf596, arg304_1, arg305_1)
        del arg304_1
        del arg305_1
        buf599 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf600 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf603 = buf592; del buf592  # reuse
        buf602 = empty_strided_cpu((s72, 3072), (3072, 1), torch.float32)
        cpp_fused_add_native_layer_norm_1(buf603, buf595, arg306_1, arg307_1, buf599, buf600, s72)
        del arg306_1
        del arg307_1
        # Topologically Sorted Source Nodes: [onednn_mm_92], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf602, buf603, arg308_1, arg309_1)
        del arg308_1
        del arg309_1
        buf606 = buf595; del buf595  # reuse
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_23], Original ATen: [aten.split_with_sizes, aten.view, vllm.unified_kv_cache_update]
        buf607 = torch.ops.vllm.unified_kv_cache_update.default(reinterpret_tensor(buf602, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf602, (s72, 16, 64), (3072, 64, 1), 2048), arg310_1)
        buf608 = buf607
        del buf607
        # Topologically Sorted Source Nodes: [kv_cache_dummy_dep_23, unified_attention_with_output_23, output_163], Original ATen: [aten.split_with_sizes, aten.view]
        torch.ops.vllm.unified_attention_with_output.default(reinterpret_tensor(buf602, (s72, 16, 64), (3072, 64, 1), 0), reinterpret_tensor(buf602, (s72, 16, 64), (3072, 64, 1), 1024), reinterpret_tensor(buf602, (s72, 16, 64), (3072, 64, 1), 2048), reinterpret_tensor(buf606, (s72, 16, 64), (1024, 64, 1), 0), arg310_1, None, None, buf608)
        del arg310_1
        del buf602
        del buf608
        buf611 = empty_strided_cpu((s72, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_163, unified_attention_with_output_23, onednn_mm_93], Original ATen: [aten.view]
        torch.ops._C.onednn_mm.default(buf611, buf606, arg311_1, arg312_1)
        del arg311_1
        del arg312_1
        del buf606
        buf614 = buf600; del buf600  # reuse
        buf615 = buf599; del buf599  # reuse
        buf618 = buf603; del buf603  # reuse
        buf617 = buf596; del buf596  # reuse
        cpp_fused_add_native_layer_norm_1(buf618, buf611, arg313_1, arg314_1, buf614, buf615, s72)
        del arg313_1
        del arg314_1
        del buf614
        del buf615
        # Topologically Sorted Source Nodes: [onednn_mm_94], Original ATen: [_C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf617, buf618, arg315_1, arg316_1)
        del arg315_1
        del arg316_1
        buf621 = buf611; del buf611  # reuse
        buf622 = empty_strided_cpu((s72, 4096), (4096, 1), torch.float32)
        cpp_fused_gelu_onednn_mm_2(buf617, buf622, s72)
        del buf617
        # Topologically Sorted Source Nodes: [hidden_states_70, onednn_mm_95], Original ATen: [aten.gelu, _C.onednn_mm]
        torch.ops._C.onednn_mm.default(buf621, buf622, arg317_1, arg318_1)
        del arg317_1
        del arg318_1
        del buf622
        buf625 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf626 = empty_strided_cpu((s72, 1), (1, s72), torch.float32)
        buf628 = buf618; del buf618  # reuse
        cpp_fused_add_native_layer_norm_1(buf628, buf621, arg319_1, arg320_1, buf625, buf626, s72)
        del arg319_1
        del arg320_1
        return (buf628, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = 8192
    arg1_1 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.int32)
    arg2_1 = rand_strided((250048, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg3_1 = 8192
    arg4_1 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.int64)
    arg5_1 = rand_strided((8194, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((1, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    import pickle
    global arg11_1
    arg11_1 = pickle.loads(b'\x80\x04\x95d\x00\x00\x00\x00\x00\x00\x00\x8c\x16vllm.utils.torch_utils\x94\x8c\tLayerName\x94\x93\x94)\x81\x94}\x94\x8c\x05value\x94\x8c*bert.encoder.layer.0.attention.output.attn\x94sb.')
    arg12_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg14_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg18_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg20_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    import pickle
    global arg24_1
    arg24_1 = pickle.loads(b'\x80\x04\x95d\x00\x00\x00\x00\x00\x00\x00\x8c\x16vllm.utils.torch_utils\x94\x8c\tLayerName\x94\x93\x94)\x81\x94}\x94\x8c\x05value\x94\x8c*bert.encoder.layer.1.attention.output.attn\x94sb.')
    arg25_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg27_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg31_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg33_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    import pickle
    global arg37_1
    arg37_1 = pickle.loads(b'\x80\x04\x95d\x00\x00\x00\x00\x00\x00\x00\x8c\x16vllm.utils.torch_utils\x94\x8c\tLayerName\x94\x93\x94)\x81\x94}\x94\x8c\x05value\x94\x8c*bert.encoder.layer.2.attention.output.attn\x94sb.')
    arg38_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg40_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg44_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg46_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    import pickle
    global arg50_1
    arg50_1 = pickle.loads(b'\x80\x04\x95d\x00\x00\x00\x00\x00\x00\x00\x8c\x16vllm.utils.torch_utils\x94\x8c\tLayerName\x94\x93\x94)\x81\x94}\x94\x8c\x05value\x94\x8c*bert.encoder.layer.3.attention.output.attn\x94sb.')
    arg51_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg53_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg57_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg59_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    import pickle
    global arg63_1
    arg63_1 = pickle.loads(b'\x80\x04\x95d\x00\x00\x00\x00\x00\x00\x00\x8c\x16vllm.utils.torch_utils\x94\x8c\tLayerName\x94\x93\x94)\x81\x94}\x94\x8c\x05value\x94\x8c*bert.encoder.layer.4.attention.output.attn\x94sb.')
    arg64_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg66_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg70_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg72_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    import pickle
    global arg76_1
    arg76_1 = pickle.loads(b'\x80\x04\x95d\x00\x00\x00\x00\x00\x00\x00\x8c\x16vllm.utils.torch_utils\x94\x8c\tLayerName\x94\x93\x94)\x81\x94}\x94\x8c\x05value\x94\x8c*bert.encoder.layer.5.attention.output.attn\x94sb.')
    arg77_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg79_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg83_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg85_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    import pickle
    global arg89_1
    arg89_1 = pickle.loads(b'\x80\x04\x95d\x00\x00\x00\x00\x00\x00\x00\x8c\x16vllm.utils.torch_utils\x94\x8c\tLayerName\x94\x93\x94)\x81\x94}\x94\x8c\x05value\x94\x8c*bert.encoder.layer.6.attention.output.attn\x94sb.')
    arg90_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg92_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg96_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg98_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    import pickle
    global arg102_1
    arg102_1 = pickle.loads(b'\x80\x04\x95d\x00\x00\x00\x00\x00\x00\x00\x8c\x16vllm.utils.torch_utils\x94\x8c\tLayerName\x94\x93\x94)\x81\x94}\x94\x8c\x05value\x94\x8c*bert.encoder.layer.7.attention.output.attn\x94sb.')
    arg103_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg105_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg109_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg111_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    import pickle
    global arg115_1
    arg115_1 = pickle.loads(b'\x80\x04\x95d\x00\x00\x00\x00\x00\x00\x00\x8c\x16vllm.utils.torch_utils\x94\x8c\tLayerName\x94\x93\x94)\x81\x94}\x94\x8c\x05value\x94\x8c*bert.encoder.layer.8.attention.output.attn\x94sb.')
    arg116_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg118_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg122_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg124_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    import pickle
    global arg128_1
    arg128_1 = pickle.loads(b'\x80\x04\x95d\x00\x00\x00\x00\x00\x00\x00\x8c\x16vllm.utils.torch_utils\x94\x8c\tLayerName\x94\x93\x94)\x81\x94}\x94\x8c\x05value\x94\x8c*bert.encoder.layer.9.attention.output.attn\x94sb.')
    arg129_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg131_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg135_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg137_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    import pickle
    global arg141_1
    arg141_1 = pickle.loads(b'\x80\x04\x95e\x00\x00\x00\x00\x00\x00\x00\x8c\x16vllm.utils.torch_utils\x94\x8c\tLayerName\x94\x93\x94)\x81\x94}\x94\x8c\x05value\x94\x8c+bert.encoder.layer.10.attention.output.attn\x94sb.')
    arg142_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg144_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg148_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg150_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    import pickle
    global arg154_1
    arg154_1 = pickle.loads(b'\x80\x04\x95e\x00\x00\x00\x00\x00\x00\x00\x8c\x16vllm.utils.torch_utils\x94\x8c\tLayerName\x94\x93\x94)\x81\x94}\x94\x8c\x05value\x94\x8c+bert.encoder.layer.11.attention.output.attn\x94sb.')
    arg155_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg157_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg161_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg163_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    import pickle
    global arg167_1
    arg167_1 = pickle.loads(b'\x80\x04\x95e\x00\x00\x00\x00\x00\x00\x00\x8c\x16vllm.utils.torch_utils\x94\x8c\tLayerName\x94\x93\x94)\x81\x94}\x94\x8c\x05value\x94\x8c+bert.encoder.layer.12.attention.output.attn\x94sb.')
    arg168_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg170_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg174_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg176_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    import pickle
    global arg180_1
    arg180_1 = pickle.loads(b'\x80\x04\x95e\x00\x00\x00\x00\x00\x00\x00\x8c\x16vllm.utils.torch_utils\x94\x8c\tLayerName\x94\x93\x94)\x81\x94}\x94\x8c\x05value\x94\x8c+bert.encoder.layer.13.attention.output.attn\x94sb.')
    arg181_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg183_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg187_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg189_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    import pickle
    global arg193_1
    arg193_1 = pickle.loads(b'\x80\x04\x95e\x00\x00\x00\x00\x00\x00\x00\x8c\x16vllm.utils.torch_utils\x94\x8c\tLayerName\x94\x93\x94)\x81\x94}\x94\x8c\x05value\x94\x8c+bert.encoder.layer.14.attention.output.attn\x94sb.')
    arg194_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg196_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg200_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg202_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    import pickle
    global arg206_1
    arg206_1 = pickle.loads(b'\x80\x04\x95e\x00\x00\x00\x00\x00\x00\x00\x8c\x16vllm.utils.torch_utils\x94\x8c\tLayerName\x94\x93\x94)\x81\x94}\x94\x8c\x05value\x94\x8c+bert.encoder.layer.15.attention.output.attn\x94sb.')
    arg207_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg209_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg213_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg215_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    import pickle
    global arg219_1
    arg219_1 = pickle.loads(b'\x80\x04\x95e\x00\x00\x00\x00\x00\x00\x00\x8c\x16vllm.utils.torch_utils\x94\x8c\tLayerName\x94\x93\x94)\x81\x94}\x94\x8c\x05value\x94\x8c+bert.encoder.layer.16.attention.output.attn\x94sb.')
    arg220_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg222_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg226_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg228_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    import pickle
    global arg232_1
    arg232_1 = pickle.loads(b'\x80\x04\x95e\x00\x00\x00\x00\x00\x00\x00\x8c\x16vllm.utils.torch_utils\x94\x8c\tLayerName\x94\x93\x94)\x81\x94}\x94\x8c\x05value\x94\x8c+bert.encoder.layer.17.attention.output.attn\x94sb.')
    arg233_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg235_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg239_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg241_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    import pickle
    global arg245_1
    arg245_1 = pickle.loads(b'\x80\x04\x95e\x00\x00\x00\x00\x00\x00\x00\x8c\x16vllm.utils.torch_utils\x94\x8c\tLayerName\x94\x93\x94)\x81\x94}\x94\x8c\x05value\x94\x8c+bert.encoder.layer.18.attention.output.attn\x94sb.')
    arg246_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg248_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg252_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg254_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    import pickle
    global arg258_1
    arg258_1 = pickle.loads(b'\x80\x04\x95e\x00\x00\x00\x00\x00\x00\x00\x8c\x16vllm.utils.torch_utils\x94\x8c\tLayerName\x94\x93\x94)\x81\x94}\x94\x8c\x05value\x94\x8c+bert.encoder.layer.19.attention.output.attn\x94sb.')
    arg259_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg261_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg265_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg267_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    import pickle
    global arg271_1
    arg271_1 = pickle.loads(b'\x80\x04\x95e\x00\x00\x00\x00\x00\x00\x00\x8c\x16vllm.utils.torch_utils\x94\x8c\tLayerName\x94\x93\x94)\x81\x94}\x94\x8c\x05value\x94\x8c+bert.encoder.layer.20.attention.output.attn\x94sb.')
    arg272_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg274_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg278_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg280_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    import pickle
    global arg284_1
    arg284_1 = pickle.loads(b'\x80\x04\x95e\x00\x00\x00\x00\x00\x00\x00\x8c\x16vllm.utils.torch_utils\x94\x8c\tLayerName\x94\x93\x94)\x81\x94}\x94\x8c\x05value\x94\x8c+bert.encoder.layer.21.attention.output.attn\x94sb.')
    arg285_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg287_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg290_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg291_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg292_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg293_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg294_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg295_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg296_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    import pickle
    global arg297_1
    arg297_1 = pickle.loads(b'\x80\x04\x95e\x00\x00\x00\x00\x00\x00\x00\x8c\x16vllm.utils.torch_utils\x94\x8c\tLayerName\x94\x93\x94)\x81\x94}\x94\x8c\x05value\x94\x8c+bert.encoder.layer.22.attention.output.attn\x94sb.')
    arg298_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg299_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg300_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg301_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg302_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg303_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg304_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg305_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg306_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg307_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg308_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg309_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    import pickle
    global arg310_1
    arg310_1 = pickle.loads(b'\x80\x04\x95e\x00\x00\x00\x00\x00\x00\x00\x8c\x16vllm.utils.torch_utils\x94\x8c\tLayerName\x94\x93\x94)\x81\x94}\x94\x8c\x05value\x94\x8c+bert.encoder.layer.23.attention.output.attn\x94sb.')
    arg311_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg312_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg313_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg314_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg315_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg316_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg317_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg318_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg319_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg320_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
