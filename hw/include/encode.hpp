/* Exposed functions of encode.c */
#ifndef ENCODE_HPP
#define ENCODE_HPP

#include <stddef.h>

#include "types.hpp"

// /**
//  * @brief Gather full 4x4x4x4 (4^D) float values from a serialized 4D matrix.
//  * @param block Pointer to the destination block.
//  * @param raw Pointer to the source array.
//  * @param sx/y/z/w Stride in the x/y/z/w dimension.
//  * @return void
//  * @note The source array must be at least 4x4x4x4 in size (no need for padding).
// */
// void gather_4d_block(float *block, const float *raw,
//                      ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw);

// /**
//  * @brief Gather partial nx*ny*nz*nw float values from a serialized 4D matrix
//  *  and pad the block to 4x4x4x4 (4^D) values.
//  * @param block Pointer to the destination block.
//  * @param raw Pointer to the source array.
//  * @param nx/y/z/w Number of elements to gather in the x/y/z/w dimension.
//  * @param sx/y/z/w Stride in the x/y/z/w dimension.
//  * @return void
// */
// void gather_partial_4d_block(float *block, const float *raw,
//                              size_t nx, size_t ny, size_t nz, size_t nw,
//                              ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw);

/**
 * @brief Get the normalized floating-point exponent for x >= 0.
 * @param x Floating-point value.
 * @return Normalized floating-point exponent.
 * @note In case x==0, the exponent is set to -EBIAS.
*/
int get_scaler_exponent(float x);

void fwd_cast(hls::stream<int32> &out_integer, hls::stream<float> &in_float, uint dim, int emax);

void fwd_decorrelate_2d_block(volatile int32 *iblock);

void fwd_reorder_int2uint_block(volatile uint32 *ublock, volatile const int32* iblock,
                          const uchar* perm, uint n);

float quantize_scaler(float x, int e);

void compute_block_exponent_2d(
  size_t in_total_blocks,
  hls::stream<fblock_2d_t> &in_fblock, 
  const zfp_output &output, 
  hls::stream<int> &out_emax,
  hls::stream<uint> &out_bemax,
  hls::stream<uint> &out_maxprec,
  hls::stream<fblock_2d_t> &out_fblock);

void compute_block_emax_2d(
  size_t in_total_blocks,
  hls::stream<fblock_2d_t> in_fblock[FIFO_WIDTH], 
  const zfp_output &output, 
  hls::stream<int> out_emax[FIFO_WIDTH],
  hls::stream<uint> out_bemax[FIFO_WIDTH],
  hls::stream<uint> out_maxprec[FIFO_WIDTH],
  hls::stream<fblock_2d_t> out_fblock[FIFO_WIDTH]);

void chunk_blocks_2d_par(
  hls::stream<fblock_2d_t> fblock[FIFO_WIDTH], const zfp_input &input);

void fwd_float2int_2d_par(
  size_t in_total_blocks,
  hls::stream<int> in_emax[FIFO_WIDTH],
  hls::stream<uint> in_bemax[FIFO_WIDTH],
  hls::stream<fblock_2d_t> in_fblock[FIFO_WIDTH],
  hls::stream<iblock_2d_t> out_iblock[FIFO_WIDTH],
  hls::stream<uint> out_bemax[FIFO_WIDTH]);

void fwd_decorrelate_2d_par(
  size_t in_total_blocks,
  hls::stream<uint> in_bemax[FIFO_WIDTH],
  hls::stream<iblock_2d_t> in_iblock[FIFO_WIDTH],
  hls::stream<iblock_2d_t> out_iblock[FIFO_WIDTH],
  hls::stream<uint> out_bemax[FIFO_WIDTH]);

void fwd_reorder_int2uint_2d_par(
  size_t in_total_blocks,
  hls::stream<uint> in_bemax[FIFO_WIDTH],
  hls::stream<iblock_2d_t> in_iblock[FIFO_WIDTH],
  hls::stream<ublock_2d_t> out_ublock[FIFO_WIDTH],
  hls::stream<uint> out_bemax[FIFO_WIDTH]);

void encode_bitplanes_2d_par(
  size_t in_total_blocks,
  hls::stream<uint> in_bemax[FIFO_WIDTH],
  hls::stream<uint> in_maxprec[FIFO_WIDTH],
  hls::stream<ublock_2d_t> in_ublock[FIFO_WIDTH],
  zfp_output &output,
  hls::stream<write_request_t> bitplane_queues[FIFO_WIDTH]);

void fwd_float2int_2d(
  size_t in_total_blocks,
  hls::stream<int> &in_emax,
  hls::stream<uint> &in_bemax,
  hls::stream<fblock_2d_t> &in_fblock,
  hls::stream<iblock_2d_t> &out_iblock,
  hls::stream<uint> &out_bemax);

void fwd_decorrelate_2d(
  size_t in_total_blocks,
  hls::stream<uint> &in_bemax,
  hls::stream<iblock_2d_t> &in_iblock,
  hls::stream<iblock_2d_t> &out_iblock,
  hls::stream<uint> &out_bemax);

void fwd_reorder_int2uint_2d(
  size_t in_total_blocks,
  hls::stream<uint> &in_bemax,
  hls::stream<iblock_2d_t> &in_iblock,
  hls::stream<ublock_2d_t> &out_ublock,
  hls::stream<uint> &out_bemax);

void encode_bitplanes_2d(
  size_t in_total_blocks,
  hls::stream<uint> &in_bemax,
  hls::stream<uint> &in_maxprec,
  hls::stream<ublock_2d_t> &in_ublock,
  zfp_output &output,
  hls::stream<write_request_t> &bitplane_queue);

void encode_padding(
  size_t in_total_blocks,
  hls::stream<uint> &in_bits,
  hls::stream<uint> &in_minbits,
  hls::stream<write_request_t> &padding_queue);

void gather_2d_block(hls::stream<float> fblock[BLOCK_SIZE_2D], const float *raw,
                     ptrdiff_t sx, ptrdiff_t sy);

void gather_partial_2d_block(hls::stream<float> fblock[BLOCK_SIZE_2D], const float *raw,
                             size_t nx, size_t ny,
                             ptrdiff_t sx, ptrdiff_t sy);

uint encode_all_bitplanes(stream &s, volatile const uint32 *const ublock,
                          uint maxprec, uint block_size);

uint encode_fblock(zfp_output &output, hls::stream<float> fblock[BLOCK_SIZE_2D],
                   size_t dim);

uint encode_iblock(stream &out_data, uint minbits, uint maxbits,
                   uint maxprec, volatile int32 *iblock, size_t dim);

void chunk_blocks_2d(hls::stream<fblock_2d_t> &fblock, const zfp_input &input);

// void encode_partial_bitplanes(stream &s, volatile const uint32 *const ublock,
//                               uint maxbits, uint maxprec, uint block_size, uint *encoded_bits);

// void encode_all_bitplanes(stream &s, volatile const uint32 *const ublock,
//                           uint maxprec, uint block_size, uint *encoded_bits);

#endif // ENCODE_HPP