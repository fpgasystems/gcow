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

/**
 * @brief Get the normalized floating-point exponent for x >= 0.
 * @param x Floating-point value.
 * @return Normalized floating-point exponent.
 * @note In case x==0, the exponent is set to -EBIAS.
*/
int get_scaler_exponent(float x);

/**
 * @brief Compute maximum floating-point exponent in block of n values.
 * @param block Pointer to the block.
 * @param n Number of elements in the block.
 * @return Maximum floating-point exponent.
*/
int get_block_exponent(const float *block, uint n);

void fwd_cast_block(int32 *iblock, const float *fblock, uint n, int emax);

void fwd_decorrelate_2d_block(int32 *iblock);

void fwd_reorder_int2uint(uint32* ublock, const int32* iblock,
                          const uchar* perm, uint n);

void gather_2d_block(float *block, const float *raw, ptrdiff_t sx,
                     ptrdiff_t sy);

void gather_partial_2d_block(float *block, const float *raw,
                             size_t nx, size_t ny, ptrdiff_t sx, ptrdiff_t sy);

/**
 * @brief Get the maximum number of bit planes to encode.
 * @param maxexp Maximum block floating-point exponent.
 * @param maxprec Maximum number of bit planes to encode.
 * @param minexp Minimum block floating-point exponent.
 * @param dim Number of dimensions.
 * @return Maximum number of bit planes to encode.
*/
uint get_precision(int maxexp, uint maxprec, int minexp, int dim);

uint encode_all_bitplanes(stream *const s, const uint32 *const ublock,
                          uint maxprec, uint block_size);

uint encode_partial_bitplanes(stream *const s,
                              const uint32 *const ublock,
                              uint maxbits, uint maxprec, uint block_size);

uint encode_fblock(zfp_output* output, const float *fblock, size_t dim);
uint encode_iblock(stream *const out_data, uint minbits, uint maxbits,
                   uint maxprec, int32 *iblock, size_t dim);
#endif // ENCODE_H