#ifndef TYPES_H
#define TYPES_H

#include <stdarg.h>

#include "common.h"

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;

typedef unsigned int uint;
typedef int8_t int8;
typedef uint8_t uint8;
typedef int16_t int16;
typedef uint16_t uint16;
typedef int32_t int32;
typedef uint32_t uint32;
typedef int64_t int64;
typedef uint64_t uint64;

/* Use the maximum word size by default for IO highest speed (irrespective of the input data type since it's just a stream of bits) */
typedef uint64 stream_word;
/* Maximum number of bits in a buffered stream word */
#define SWORD_BITS ((size_t)(sizeof(stream_word) * CHAR_BIT))

/* ZFP Compression mode */
typedef enum {
  zfp_null            = 0, /* an invalid configuration of the 4 params */
  zfp_expert          = 1, /* expert mode (4 params set manually) */
  zfp_fixed_rate      = 2, /* fixed rate mode */
  zfp_fixed_precision = 3, /* fixed precision mode */
  zfp_fixed_accuracy  = 4, /* fixed accuracy mode */
  zfp_reversible      = 5  /* reversible (lossless) mode */
} zfp_mode;

/* Data type */
typedef enum {
  dtype_none   = 0, /* unspecified type */
  dtype_int32  = 1, /* 32-bit signed integer */
  dtype_int64  = 2, /* 64-bit signed integer */
  dtype_float  = 3, /* single precision floating point */
  dtype_double = 4  /* double precision floating point */
} data_type;

/**
 * @brief Uncompressed array
 * @note Zero for unused dimensions, and zero stride for contiguous a[nw][nz][ny][nx]
*/
typedef struct {
  data_type dtype;          /* data type of the scale values */
  void* data;               /* pointer to the array data */
  size_t nx, ny, nz, nw;    /* size of the array in the x/y/z/w dimension */
  ptrdiff_t sx, sy, sz, sw; /* stride of the array in the x/y/z/w dimension */
} zfp_input;

typedef struct {
  uint minbits;       /* minimum number of bits to store per block */
  uint maxbits;       /* maximum number of bits to store per block */
  uint maxprec;       /* maximum number of bit planes to store */
  int minexp;         /* minimum floating point bit plane number to store */
  stream* data;       /* compressed bit stream */
  // zfp_execution exec; /* execution policy and parameters */
} zfp_output;


#define index(i, j) ((i) + 4 * (j))
//* Ordering coefficients (i, j) by i + j, then i^2 + j^2
//* (Similar to the zig-zag ordering of JPEG.)
static const uchar PERM_2D[16] = {
  index(0, 0), /*  0 : 0 */

  index(1, 0), /*  1 : 1 */
  index(0, 1), /*  2 : 1 */

  index(1, 1), /*  3 : 2 */

  index(2, 0), /*  4 : 2 */
  index(0, 2), /*  5 : 2 */

  index(2, 1), /*  6 : 3 */
  index(1, 2), /*  7 : 3 */

  index(3, 0), /*  8 : 3 */
  index(0, 3), /*  9 : 3 */

  index(2, 2), /* 10 : 4 */

  index(3, 1), /* 11 : 4 */
  index(1, 3), /* 12 : 4 */

  index(3, 2), /* 13 : 5 */
  index(2, 3), /* 14 : 5 */

  index(3, 3), /* 15 : 6 */
};
#undef index


/**
 * @brief Set output accuracy parameters.
 * @param output Output stream.
 * @param tolerance Absolute error tolerance.
 * @return Maximum error tolerance (x=1) for the given precision.
*/
double set_zfp_output_accuracy(zfp_output *output, double tolerance);
zfp_input *alloc_zfp_input(void);
zfp_output *alloc_zfp_output(void);
void free_zfp_input(zfp_input* input);
void free_zfp_output(zfp_output* output);
void cleanup(zfp_input *input, zfp_output *output);
zfp_input *init_zfp_input(void *data, data_type dtype, uint dim, ...);
zfp_output *init_zfp_output(const zfp_input *input);
uint is_reversible(const zfp_output* output);
uint get_input_dimension(const zfp_input* input);
size_t get_input_num_blocks(const zfp_input* input);
size_t get_input_size(const zfp_input* input, size_t* shape);
size_t get_dtype_size(data_type dtype);
uint get_input_precision(const zfp_input* input);
size_t get_max_output_bytes(const zfp_output *output, const zfp_input *input);

#endif // TYPES_H