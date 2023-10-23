#include <stddef.h>
#include <stdio.h>

#define BLOCK_SIZE 256

typedef unsigned int uint;


/**
 * @brief Gather full 4x4x4x4 (4^D) double values from a serialized 4D array.
 * @param block Pointer to the destination block.
 * @param raw Pointer to the source array.
 * @param sx Stride in the x dimension.
 * @param sy Stride in the y dimension.
 * @param sz Stride in the z dimension.
 * @param sw Stride in the w dimension.
 * @return void
 * @note The source array must be at least 4x4x4x4 in size (no need for padding).
*/
void gather_block(double *block, const double *raw, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)
{
  uint x, y, z, w;
  for (w = 0; w < 4; w++, raw += sw - 4 * sz)
    for (z = 0; z < 4; z++, raw += sz - 4 * sy)
      for (y = 0; y < 4; y++, raw += sy - 4 * sx)
        for (x = 0; x < 4; x++, raw += sx)
          *block++ = *raw;
}

//TODO: Partial gather function

void encode_block(int *encoded, const double *block)
{
  //TODO: Implement encoding
  for (int i = 0; i < BLOCK_SIZE; i++)
    encoded[i] = (int) block[i];
}

void encode_block_strided(int *encoded, const double *raw, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)
{
  double block[BLOCK_SIZE];
  
  //TODO: Cache alignment?
  gather_block(block, raw, sx, sy, sz, sw);
  encode_block(encoded, block);
}
