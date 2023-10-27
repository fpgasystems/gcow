/* Exposed functions of encode.c */
#ifndef ENCODE_H
#define ENCODE_H

#include <stddef.h>

#include "types.h"

/**
 * @brief Gather full 4x4x4x4 (4^D) float values from a serialized 4D matrix.
 * @param block Pointer to the destination block.
 * @param raw Pointer to the source array.
 * @param sx/y/z/w Stride in the x/y/z/w dimension.
 * @return void
 * @note The source array must be at least 4x4x4x4 in size (no need for padding).
*/
void gather_4d_block(float *block, const float *raw,
                     ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw);

/**
 * @brief Gather partial nx*ny*nz*nw float values from a serialized 4D matrix
 *  and pad the block to 4x4x4x4 (4^D) values.
 * @param block Pointer to the destination block.
 * @param raw Pointer to the source array.
 * @param nx/y/z/w Number of elements to gather in the x/y/z/w dimension.
 * @param sx/y/z/w Stride in the x/y/z/w dimension.
 * @return void
*/
void gather_partial_4d_block(float *block, const float *raw,
                             size_t nx, size_t ny, size_t nz, size_t nw,
                             ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw);

void encode_4d_block(uint32 *encoded, const float *fblock);

void encode_strided_4d_block(uint32 *encoded, const float *raw,
                             ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw);

void encode_strided_partial_4d_block(uint32 *encoded, const float *raw,
                                     size_t nx, size_t ny, size_t nz, size_t nw,
                                     ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw);


void gather_2d_block(float *block, const float *raw, ptrdiff_t sx,
                     ptrdiff_t sy);

void gather_partial_2d_block(float *block, const float *raw,
                             size_t nx, size_t ny, ptrdiff_t sx, ptrdiff_t sy);

void encode_2d_block(uint32 *encoded, const float *fblock);

void encode_strided_2d_block(uint32 *encoded, const float *raw, ptrdiff_t sx,
                             ptrdiff_t sy);

void encode_strided_partial_2d_block(uint32 *encoded, const float *raw,
                                     size_t nx, size_t ny, ptrdiff_t sx, ptrdiff_t sy);

void compress_2d(uint32 *compressed, const zfp_specs* specs);

#endif // ENCODE_H