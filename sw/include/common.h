#include <stddef.h>
#include <stdint.h>
#include <limits.h>

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

/* Forward definitions */
typedef struct stream stream;

int exceeded_maxbits(uint maxbits, uint maxprec, uint size);