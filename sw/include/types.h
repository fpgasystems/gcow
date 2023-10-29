#ifndef TYPES_H
#define TYPES_H

#include "common.h"

typedef unsigned int uint;
typedef int8_t int8;
typedef uint8_t uint8;
typedef int16_t int16;
typedef uint16_t uint16;
typedef int32_t int32; 
typedef uint32_t uint32;
typedef int64_t int64;
typedef uint64_t uint64;

/* Types matching float32 (common DNN param type) */
typedef uint32 word;
/* Maximum number of bits in a buffered word */
#define WORD_BITS ((size_t)(sizeof(word) * CHAR_BIT))


/* ZFP Compression mode */
typedef enum {
  zfp_null            = 0, /* an invalid configuration of the 4 params */
  zfp_expert          = 1, /* expert mode (4 params set manually) */
  zfp_fixed_rate      = 2, /* fixed rate mode */
  zfp_fixed_precision = 3, /* fixed precision mode */
  zfp_fixed_accuracy  = 4, /* fixed accuracy mode */
  zfp_reversible      = 5  /* reversible (lossless) mode */
} zfp_mode;

/**
 * @brief Uncompressed array
 * @param nx/y/z/w size of the array in the x/y/z/w dimension
 * @param sx/y/z/w stride of the array in the x/y/z/w dimension
 * @param data pointer to the array data
 * @param minbits minimum number of bits to store per block
 * @param maxbits maximum number of bits to store per block
 * @param maxprec maximum number of bit planes to store
 * @param minexp minimum floating point bit plane number to store
 * @note Zero for unused dimensions, and zero stride for contiguous a[nw][nz][ny][nx]
*/
typedef struct
{
  void* data;
  size_t nx, ny, nz, nw;
  ptrdiff_t sx, sy, sz, sw;
} zfp_input;

typedef struct {
  uint minbits;       /* minimum number of bits to store per block */
  uint maxbits;       /* maximum number of bits to store per block */
  uint maxprec;       /* maximum number of bit planes to store */
  int minexp;         /* minimum floating point bit plane number to store */
  stream* data;       /* compressed bit stream */
  // zfp_execution exec; /* execution policy and parameters */
} zfp_output;


#endif // TYPES_H