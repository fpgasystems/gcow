#ifndef DECODE_H
#define DECODE_H

#include <stddef.h>

#include "types.h"

uint decode_fblock(zfp_output* output, float* fblock, size_t dim);
void scatter_2d_block(const float *block, float *raw,
                      ptrdiff_t sx, ptrdiff_t sy);
void scatter_partial_2d_block(const float *block, float *raw,
                              size_t nx, size_t ny, ptrdiff_t sx, ptrdiff_t sy);

#endif // DECODE_H