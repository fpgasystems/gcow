#include "encode.hpp"

extern "C" {
  void gather(const float *in_block, size_t nx, size_t ny, float *out_block)
  {
#pragma HLS INTERFACE mode=m_axi port=in_block offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=out_block offset=slave bundle=gmem0

    // const float* data = (const float*)in_block;
    // ptrdiff_t sx = 1;
    // ptrdiff_t sy = (ptrdiff_t)nx;

    // //* Compress array one block of 4x4 values at a time
    // size_t i = 0;
    // for (size_t y = 0; y < ny; y += 4) {
    //   // printf("Encoding block [%ld, *]\n", y);
    //   for (size_t x = 0; x < nx; x += 4) {
    //     const float *raw = data + sx * (ptrdiff_t)x + sy * (ptrdiff_t)y;
    //     float fblock[BLOCK_SIZE_2D];

    //     if (nx - x < 4 || ny - y < 4) {
    //       gather_partial_2d_block(fblock, raw, MIN(nx - x, 4u), MIN(ny - y, 4u), sx, sy);
    //     } else {
    //       gather_2d_block(fblock, raw, sx, sy);
    //     }

    //     //* Copy block to output array seperately to avoid pointer aliasing.
    //     for (size_t j = 0; j < BLOCK_SIZE_2D; j++) {
    //       out_block[i++] = fblock[j];
    //     }
    //   }
    // }
  }
}