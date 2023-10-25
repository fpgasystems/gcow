#ifndef TYPES_H
#define TYPES_H

#include <stddef.h>

#define BLOCK_SIZE_2D 16
#define BLOCK_SIZE_4D 256
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

typedef unsigned int uint;
typedef unsigned long long uint64;

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
  size_t nx, ny, nz, nw;
  ptrdiff_t sx, sy, sz, sw;
  void* data;

  uint minbits;
  uint maxbits;
  uint maxprec;
  int  minexp;
} zfp_specs;

#endif // TYPES_H