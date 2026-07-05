
#include <cstdint>
#include <immintrin.h>

extern "C" __m256i __avx512_vnni_chk_kernel_1(__m256i src, __m256i a, __m256i b) {
    return _mm256_dpbusd_epi32(src, a, b);
}

extern "C" __m512i __avx512_vnni_chk_kernel_2(__m512i src, __m512i a, __m512i b) {
    return _mm512_dpbusd_epi32(src, a, b);
}
