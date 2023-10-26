#include <stddef.h>
#include <stdio.h>

#include "encode.h"
#include "types.h"


/**
 * @brief Pad a partial block to 4 values.
 * @param block Pointer to the block.
 * @param n Number of elements in the block (NOT #elements to pad).
 * @param s Stride of the block elements.
 * @return void
 * @note When padding a vector having ≥1 element, the last element is copied to
 *  the next position. When padding a vector having 0 elements, the all elements
 *  are set to 0.
 *  For example, for a 2x3 block, the last two rows are all set to zeros first,
 *  when padding row-wise. Then, when going through the columns, the zeros will
 *  be filled with the last value of the first row:
 *   * Raw stream:
 *      1 2 3 _
 *      4 5 6 _
 *      _ _ _ _
 *      _ _ _ _
 *   * First, padding over rows (dim-x):
 *      1 2 3 1
 *      4 5 6 4
 *      0 0 0 0
 *      0 0 0 0
 *   * Then, padding over columns (dim-y):
 *      1 2 3 1
 *      4 5 6 4
 *      4 5 6 4
 *      1 2 3 1
*/
void pad_partial_block(double* block, size_t n, ptrdiff_t s)
{
  switch (n) {
  case 0:
    block[0 * s] = 0;
  /* FALLTHROUGH */
  case 1:
    //* Fill the next position by copying the previous value.
    block[1 * s] = block[0 * s];
  /* FALLTHROUGH */
  case 2:
    block[2 * s] = block[1 * s];
  /* FALLTHROUGH */
  case 3:
    block[3 * s] = block[0 * s];
  /* FALLTHROUGH */
  default:
    break;
  }
}

void gather_2d_block(double *block, const double *raw,
                     ptrdiff_t sx, ptrdiff_t sy)
{
  uint x, y;
  for (y = 0; y < 4; y++, raw += sy - 4 * sx)
    for (x = 0; x < 4; x++, raw += sx)
      *block++ = *raw;
}

void gather_partial_2d_block(double *block, const double *raw,
                             size_t nx, size_t ny,
                             ptrdiff_t sx, ptrdiff_t sy)
{
  size_t x, y;
  for (y = 0; y < ny; y++, raw += sy - (ptrdiff_t)nx * sx) {
    for (x = 0; x < nx; x++, raw += sx) {
      block[4 * y + x] = *raw;
    }
    //* Pad horizontally to 4.
    pad_partial_block(block + 4 * y, nx, 1);
  }
  for (x = 0; x < 4; x++)
    //* Pad vertically to 4 (stride = 4).
    pad_partial_block(block + x, ny, 4);
}

void gather_4d_block(double *block, const double *raw,
                     ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)
{
  uint x, y, z, w;
  for (w = 0; w < 4; w++, raw += sw - 4 * sz)
    for (z = 0; z < 4; z++, raw += sz - 4 * sy)
      for (y = 0; y < 4; y++, raw += sy - 4 * sx)
        for (x = 0; x < 4; x++, raw += sx)
          *block++ = *raw;
}

void gather_partial_4d_block(double *block, const double *raw,
                             size_t nx, size_t ny, size_t nz, size_t nw,
                             ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)
{
  size_t x, y, z, w;
  for (w = 0; w < nw; w++, raw += sw - (ptrdiff_t)nz * sz) {
    for (z = 0; z < nz; z++, raw += sz - (ptrdiff_t)ny * sy) {
      for (y = 0; y < ny; y++, raw += sy - (ptrdiff_t)nx * sx) {
        for (x = 0; x < nx; x++, raw += sx) {
          block[64 * w + 16 * z + 4 * y + x] = *raw;
        }
        //* Pad x dimension if the number of elements gathered is less than 4.
        pad_partial_block(block + 64 * w + 16 * z + 4 * y, nx, 1);
      }
      for (x = 0; x < 4; x++)
        pad_partial_block(block + 64 * w + 16 * z + x, ny, 4);
    }
    for (y = 0; y < 4; y++)
      for (x = 0; x < 4; x++)
        pad_partial_block(block + 64 * w + 4 * y + x, nz, 16);
  }
  for (z = 0; z < 4; z++)
    for (y = 0; y < 4; y++)
      for (x = 0; x < 4; x++)
        pad_partial_block(block + 16 * z + 4 * y + x, nw, 64);
}

void encode_2d_block(uint64 *encoded, const double *block)
{
  //TODO: Implement 2d encoding
  for (int i = 0; i < BLOCK_SIZE_2D; i++)
    encoded[i] = (int) block[i];
}

void encode_4d_block(uint64 *encoded, const double *block)
{
  //TODO: Implement 4d encoding
  for (int i = 0; i < BLOCK_SIZE_4D; i++)
    encoded[i] = (int) block[i];
}

void encode_strided_2d_block(uint64 *encoded, const double *raw,
                             ptrdiff_t sx, ptrdiff_t sy)
{
  double block[BLOCK_SIZE_2D];

  //TODO: Cache alignment?
  gather_2d_block(block, raw, sx, sy);
  encode_2d_block(encoded, block);
}

void encode_strided_partial_2d_block(uint64 *encoded, const double *raw,
                                     size_t nx, size_t ny, ptrdiff_t sx, ptrdiff_t sy)
{
  double block[BLOCK_SIZE_2D];

  //TODO: Cache alignment?
  gather_partial_2d_block(block, raw, nx, ny, sx, sy);
  encode_2d_block(encoded, block);
}

void encode_strided_4d_block(uint64 *encoded, const double *raw,
                             ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)
{
  double block[BLOCK_SIZE_4D];

  //TODO: Cache alignment?
  gather_4d_block(block, raw, sx, sy, sz, sw);
  encode_4d_block(encoded, block);
}

void encode_strided_partial_4d_block(uint64 *encoded, const double *raw,
                                     size_t nx, size_t ny, size_t nz, size_t nw,
                                     ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)
{
  double block[BLOCK_SIZE_4D];

  //TODO: Cache alignment?
  gather_partial_4d_block(block, raw, nx, ny, nz, nw, sx, sy, sz, sw);
  encode_4d_block(encoded, block);
}

void compress_2d(uint64 *compressed, const zfp_specs* specs)
{
  const double* data = (const double*)specs->data;
  size_t nx = specs->nx;
  size_t ny = specs->ny;
  ptrdiff_t sx = specs->sx ? specs->sx : 1;
  ptrdiff_t sy = specs->sy ? specs->sy : (ptrdiff_t)nx;
  size_t x, y;

  //* Compress array one block of 4x4 values at a time
  uint64 *encoded = compressed;
  for (y = 0; y < ny; y += 4)
    for (x = 0; x < nx; x += 4, encoded += 16) {
      const double *raw = data + sx * (ptrdiff_t)x + sy * (ptrdiff_t)y;
      // uint64 *encoded = compressed + (x / 4) + (y / 4) * (nx / 4);
      if (nx - x < 4 || ny - y < 4)
        encode_strided_partial_2d_block(encoded, raw, MIN(nx - x, 4u), MIN(ny - y, 4u),
                                        sx, sy);
      else
        encode_strided_2d_block(encoded, raw, sx, sy);
    }
}
