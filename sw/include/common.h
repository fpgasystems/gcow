#include <stddef.h>
#include <stdint.h>
#include <limits.h>
#include <math.h>

/* default compression parameters */
#define ZFP_MIN_BITS     1 /* minimum number of bits per block */
#define ZFP_MAX_BITS 16658 /* maximum number of bits per block */
#define ZFP_MAX_PREC    64 /* maximum precision supported */
#define ZFP_MIN_EXP  -1074 /* minimum floating-point base-2 exponent */

/* number of bits per header entry */
#define ZFP_MAGIC_BITS       32 /* number of magic word bits */
#define ZFP_META_BITS        52 /* number of field metadata bits */
#define ZFP_MODE_SHORT_BITS  12 /* number of mode bits in short format */
#define ZFP_MODE_LONG_BITS   64 /* number of mode bits in long format */
#define ZFP_HEADER_MAX_BITS 148 /* max number of header bits */
#define ZFP_MODE_SHORT_MAX  ((1u << ZFP_MODE_SHORT_BITS) - 2)

#define BLOCK_SIZE_2D 16
#define BLOCK_SIZE_4D 256
#define BLOCK_SIZE(dim) (1 << (2 * (dim)))

#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

/* Number of exponent bits (float32) */
#define EBITS 8
/* IEEE offset (127) for negative exponents. */
#define EBIAS ((1 << (EBITS - 1)) - 1)
/* Negabinary mask (float32) */
#define NBMASK 0xaaaaaaaau

#if __STDC_VERSION__ >= 199901L
#define FABS(x)     fabsf(x)
/* x = f * 2^e (returns e and f) */
#define FREXP(x, e) frexpf(x, e)
/* x = f * 2^e (returqns x) */
#define LDEXP(f, e) ldexpf(f, e)
#else
#define FABS(x)     (float)fabs(x)
#define FREXP(x, e) (void)frexp(x, e)
#define LDEXP(f, e) (float)ldexp(f, e)
#endif

typedef unsigned int uint;

/* Forward definition */
typedef struct stream stream;
/* True if max compressed size exceeds maxbits */
int exceeded_maxbits(uint maxbits, uint maxprec, uint size);
/**
 * @brief Get the maximum number of bit planes to encode.
 * @param maxexp Maximum block floating-point exponent.
 * @param maxprec Maximum number of bit planes to encode.
 * @param minexp Minimum block floating-point exponent.
 * @param dim Number of dimensions.
 * @return Maximum number of bit planes to encode.
*/
uint get_precision(int maxexp, uint maxprec, int minexp, int dim);