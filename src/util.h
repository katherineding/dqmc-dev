#pragma once

#include <string.h> //memset and memcpy
/**
 * the use of _mm_malloc and _mm_free is an optimization for allocating 
 * aligned memory for SIMD. 
 * 
 * Using MEM_ALIGN = 64 works with up to AVX-512 instruction set, since
 * 512 bits = 64 bytes
 * 
 * The intel verion of xmmintrin.h picks up _mm_malloc and _mm_free from
 * malloc.h, or otherwise, use the intel proprietary library libirc
 * 
 * However, the AOCL/clang header xmmintrin.h picks up _mm_malloc and _mm_free
 * from its own header mm_malloc.h, which on Linux systems, defines
 * _mm_malloc() and _mm_free() as wrappers over posix_memalign() and free()
 * 
 * We basically only care about Linux systems, so we can use
 * posix_memalign or some POSIX specific type function
 * 
 * Avoid aligned_alloc() specified in the C11 standard, since it requires
 * size (in  bytes) be a multiple of alignment. Which negates the purpose of 
 * using _mm_align for optimization.
 * 
 * Note we don't use any actual xmmintrinsics fuctions, only
 * _mm_malloc and _mm_free. But including xmmintrin.h is portable between
 * AOCC and ICX/ICC compilers.
 * 
 * Really if we don't care about optimization, just using malloc() and free() 
 * is also fine.
 *
 */
#include <xmmintrin.h>

// #ifdef USE_CPLX
// 	#include <complex.h>
// 	typedef double complex num;
// #else
// 	typedef double num;
// #endif

#define MEM_ALIGN 64

#define my_malloc(size) _mm_malloc((size), MEM_ALIGN)
#define my_free(ptr) _mm_free((ptr))
#define my_copy(dest, src, N) memcpy((dest), (src), (N)*sizeof((src)[0]))

static inline void *my_calloc(size_t size)
{
	void *p = my_malloc(size);
	if (p != NULL) memset(p, 0, size);
	return p;
}
