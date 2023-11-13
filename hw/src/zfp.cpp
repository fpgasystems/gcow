#include <stdio.h>

#include "types.hpp"
#include "stream.hpp"
#include "encode.hpp"
#include "zfp.hpp"


size_t zfp_compress(zfp_output &output, const zfp_input &input)
{
  switch (get_input_dimension(input)) {
    case 2:
      zfp_compress_2d(output, input);
      break;
    case 3:
      //TODO
      break;
    case 4:
      //TODO
      break;
    default:
      break;
  }

  stream_flush(output.data);
  return stream_size_bytes(output.data);
}


void zfp_compress_2d(zfp_output &output, const zfp_input &input)
{
  uint dim = 2;
  const float* data = input.data;
  size_t nx = input.nx;
  size_t ny = input.ny;
  ptrdiff_t sx = input.sx ? input.sx : 1;
  ptrdiff_t sy = input.sy ? input.sy : (ptrdiff_t)nx;

  //* Compress array one block of 4x4 values at a time
LOOP_ENCODE_BLOCKS_2D:
  for (size_t y = 0; y < ny; y += 4) {
LOOP_ENCODE_BLOCKS_2D_INNER:
    for (size_t x = 0; x < nx; x += 4) {
      //TODO: Use hls stream for reading blocks.
      const float *raw = data + sx * (ptrdiff_t)x + sy * (ptrdiff_t)y;
      float fblock[BLOCK_SIZE_2D];

      if (nx - x < 4 || ny - y < 4) {
        gather_partial_2d_block(fblock, raw, MIN(nx - x, 4u), MIN(ny - y, 4u), sx, sy);
      } else {
        gather_2d_block(fblock, raw, sx, sy);
      }
      encode_fblock(output, fblock, dim);
    }
  }
}