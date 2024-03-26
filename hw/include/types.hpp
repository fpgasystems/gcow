#ifndef TYPES_HPP
#define TYPES_HPP

//* Enablement for axis signals
#define AXIS_ENABLE_DATA 0b00000001
#define AXIS_ENABLE_DEST 0b00000010
#define AXIS_ENABLE_ID   0b00000100
#define AXIS_ENABLE_KEEP 0b00001000
#define AXIS_ENABLE_LAST 0b00010000
#define AXIS_ENABLE_STRB 0b00100000
#define AXIS_ENABLE_USER 0b01000000
// #define MAX_NUM_BLOCKS_2D 2
#define NUM_WRITE_QUEUES 3

#include <ap_int.h>
#include <stdarg.h>
#include <hls_stream.h>
// #include "ap_axi_sdata.h"

#include "common.hpp"

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

using bit_t = ap_uint<1>;
// bit_t True = 1;
// bit_t False = 0;
#define BUFFER_SIZE (BLOCK_SIZE_2D*32)
using buffer_t = ap_uint<BUFFER_SIZE>; // maximum # of encoding bits for a 2D block.
using residual_t = ap_uint<512>;

// typedef ap_axis<32, 0, 0, 0> sdata;
// typedef ap_axiu<32, 0, 0, 0> udata;

//* Data block (a parallel unit)
template <typename T, size_t N>
struct block {
  size_t id;
  T data[N];
};

using fblock_2d_t = block<float, BLOCK_SIZE_2D>;
using iblock_2d_t = block<int32, BLOCK_SIZE_2D>;
using ublock_2d_t = block<uint32, BLOCK_SIZE_2D>;

//* Request to write `nbits` bits of `value` for encoding block `block_id`.
struct write_request_t {
  size_t block_id;
  uint index; // order of the request in the block
  uint nbits; // number of bits to write/read
  uint64 value; // value to write/read
  //TODO: Change to bit_t
  bool last;   // last request of the block
  //^ 17 bytes

  write_request_t(void)
    : block_id(0), index(0), nbits(0), value(0), last(false)
  {}

  write_request_t(size_t block_id, uint index, uint nbits, uint64 value, bool last)
    : block_id(block_id), index(index), nbits(nbits), value(value), last(last)
  {}
};

/* Use the maximum word size by default for IO highest speed (irrespective of the input data type since it's just a stream of bits) */
typedef ap_uint<512> stream_word;
// typedef uint64 stream_word;
/* Maximum number of bits in a buffered stream word */
#define SWORD_BITS ((size_t)(sizeof(stream_word) * CHAR_BIT))

struct outputbuf {
  buffer_t buffer;
  size_t buffered_bits;

  outputbuf(void)
    : buffer(buffer_t(0)), buffered_bits(0)
  {}
};

struct stream {
  size_t buffered_bits; /* number of buffered bits (0 <= buffered_bits < SWORD_BITS) */
  size_t block_id;      /* the id of the block currently being operated on */
  stream_word buffer;   /* incoming/outgoing bits (buffer < 2^buffered_bits) */
  volatile stream_word *begin;   /* pointer to the beginning of the output data */
  ptrdiff_t idx;     /* offset to next stream_word to be read/written */
  ptrdiff_t end;     /* offset to the end of stream (not enforced) */

  stream(void)
    : buffered_bits(0), block_id(0), buffer(stream_word(0)), begin(nullptr), idx(0), end(0)
  {}

  stream(volatile stream_word *output_data, size_t bytes)
    : buffered_bits(0), block_id(0), buffer(stream_word(0)), begin(nullptr), idx(0), end(0)
  {
    this->begin = output_data;
    this->end = bytes / sizeof(stream_word);
    this->idx = 0;
    this->buffer = stream_word(0);
    this->buffered_bits = 0;
  }

  stream(const stream &s)
  {
    buffered_bits = s.buffered_bits;
    buffer = s.buffer;
    idx = s.idx;
    begin = s.begin;
    end = s.end;
  }

  stream &operator=(const stream &s)
  {
    buffered_bits = s.buffered_bits;
    buffer = s.buffer;
    idx = s.idx;
    begin = s.begin;
    end = s.end;
    return *this;
  }
};

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
struct zfp_input {
  data_type dtype;          /* data type of the scale values */
  const float *data;              /* pointer to the input data */
  size_t nx, ny, nz, nw;    /* size of the array in the x/y/z/w dimension */
  ptrdiff_t sx, sy, sz, sw; /* stride of the array in the x/y/z/w dimension */

  zfp_input(void)
    : dtype(dtype_none), data(nullptr), nx(0), ny(0), nz(0), nw(0), sx(0), sy(0),
      sz(0), sw(0)
  {}

  zfp_input(data_type dtype, size_t shape[DIM_MAX], const uint dim)
    : dtype(dtype), data(nullptr), nx(0), ny(0), nz(0), nw(0), sx(0), sy(0), sz(0),
      sw(0)
  {
    for (int i = 0; i < dim; i++) {
      switch (i) {
        case 0:
          this->nx = shape[i];
          break;
        case 1:
          this->ny = shape[i];
          break;
        case 2:
          this->nz = shape[i];
          break;
        case 3:
          this->nw = shape[i];
          break;
      }
    }
  }
};

struct zfp_output {
  uint minbits;       /* minimum number of bits to store per block */
  uint maxbits;       /* maximum number of bits to store per block */
  uint maxprec;       /* maximum number of bit planes to store */
  int minexp;         /* minimum floating point bit plane number to store */
  stream data;       /* compressed bit stream */
  // zfp_execution exec; /* execution policy and parameters */

  zfp_output()
    : minbits(ZFP_MIN_BITS), maxbits(ZFP_MAX_BITS), maxprec(ZFP_MAX_PREC),
      minexp(ZFP_MIN_EXP), data(stream())
  {}
};


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
double set_zfp_output_accuracy(zfp_output &output, double tolerance);

uint is_reversible(const zfp_output &output);

uint get_input_dimension(const zfp_input &input);

size_t get_input_num_blocks(const zfp_input &input);

size_t get_input_size(const zfp_input &input);

size_t get_dtype_size(data_type dtype);

uint get_input_precision(const zfp_input &input);

size_t get_max_output_bytes(const zfp_output &output, const zfp_input &input);

uint get_precision(int maxexp, uint maxprec, int minexp, int dim);

int exceeded_maxbits(uint maxbits, uint maxprec, uint size);

#endif // TYPES_HPP