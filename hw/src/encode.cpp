// Description: Encoding functions for ZFP compression.
// Documentation: ./include/encode.hpp

#include <stddef.h>
#include <stdio.h>
#include <math.h>

#include "encode.hpp"
#include "stream.hpp"
#include "types.hpp"
#include "io.hpp"

#include <bitset>

/**
 * @brief Pad a partial row/column to 4 values.
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
void pad_partial_block(volatile float *block, size_t n, ptrdiff_t s)
{
#pragma HLS INLINE

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
  //* The padding values won't be read until they are filled.
}

void gather_2d_block(float *block, const float *raw,
                     ptrdiff_t sx, ptrdiff_t sy)
{
#pragma HLS INLINE

  gather_2d_outer: for (size_t y = 0; y < 4; y++, raw += sy - 4 * sx)
    gather_2d_inner: for (size_t x = 0; x < 4; x++, raw += sx) {
      #pragma HLS PIPELINE II=1
      //TODO: Burst read with batch size of 512.

      *block++ = *raw;
    }
}

void gather_partial_2d_block(volatile float *block, volatile const float *raw,
                             size_t nx, size_t ny,
                             ptrdiff_t sx, ptrdiff_t sy)
{
#pragma HLS INLINE

  size_t x, y;
  gather_partial_2d_outer: 
  for (y = 0; y < ny; y++, raw += sy - (ptrdiff_t)nx * sx) {
    gather_partial_2d_inner: 
    for (x = 0; x < nx; x++, raw += sx) {
      #pragma HLS PIPELINE II=1

      block[4 * y + x] = *raw;
    }
    //* Pad horizontally to 4.
    pad_partial_block(block + 4 * y, nx, 1);
  }

  pad_vertical_loop:
  for (x = 0; x < 4; x++) {
    #pragma HLS UNROLL factor=4

    //* Pad vertically to 4 (stride = 4).
    pad_partial_block(block + x, ny, 4);
  }
}

void compute_block_exponent_2d(
  size_t in_total_blocks,
  hls::stream<fblock_2d_t> &in_fblock, 
  const zfp_output &output, 
  hls::stream<int> &out_emax,
  hls::stream<uint> &out_bemax,
  hls::stream<uint> &out_maxprec,
  hls::stream<fblock_2d_t> &out_fblock)
{
// #pragma HLS PIPELINE II=1
  emax_block_loop: 
  for (size_t block_id = 0; block_id < in_total_blocks; block_id++) {
    #pragma HLS PIPELINE II=1
    #pragma HLS UNROLL factor=64
    
    //* Blocking read.
    fblock_2d_t fblock_buf = in_fblock.read();
    //* Immediately relay the read block to the next module.
    out_fblock.write(fblock_buf);

    float emax = 0;
    //~ 1: Find the maximum floating point and return its exponent as the block exponent.
    emax_loop: for (uint i = 0; i < BLOCK_SIZE_2D; i++) {
      #pragma HLS PIPELINE II=1
      float f = FABS(fblock_buf.data[i]);
      emax = MAX(emax, f);
    }

    //~ 2: Get the exponent of the maximum floating point.
    int emax_out = -EBIAS;
    if (emax > 0) {
      //* Get exponent of emax.
      FREXP(emax, &emax_out);
      //* Clamp exponent in case x is subnormal; may still result in overflow.
      //* E.g., smallest number: 2^(-126) = 1.1754944e-38, which is subnormal.
      emax_out = MAX(emax_out, 1 - EBIAS);
    }
    uint maxprec = get_precision(emax_out, output.maxprec, output.minexp, 2);
    uint biased_emax = (maxprec)? (uint)(emax_out + EBIAS) : 0;

    out_maxprec.write(maxprec);
    out_bemax.write(biased_emax);
    out_emax.write(emax_out);
  }
}

void emax_producer(
  uint pe_idx,
  size_t in_total_blocks,
  hls::stream<fblock_2d_t> &in_fblock, 
  // const zfp_output &output, 
  uint maxprec,
  int minexp,
  hls::stream<int> &out_emax,
  hls::stream<uint> &out_bemax,
  hls::stream<uint> &out_maxprec,
  hls::stream<fblock_2d_t> &out_fblock)
{
#pragma HLS INLINE off

  emax_block_loop: 
  for (size_t block_id = pe_idx; block_id < in_total_blocks; block_id += FIFO_WIDTH) {
    
    //* Blocking read.
    fblock_2d_t fblock_buf = in_fblock.read();
    //* Immediately relay the read block to the next module.
    out_fblock.write(fblock_buf);

    float emax = 0;
    //~ 1: Find the maximum floating point and return its exponent as the block exponent.
    emax_loop: 
    for (uint i = 0; i < BLOCK_SIZE_2D; i++) {
      #pragma HLS PIPELINE II=1
      float f = FABS(fblock_buf.data[i]);
      emax = MAX(emax, f);
    }

    //~ 2: Get the exponent of the maximum floating point.
    int emax_out = -EBIAS;
    if (emax > 0) {
      //* Get exponent of emax.
      FREXP(emax, &emax_out);
      //* Clamp exponent in case x is subnormal; may still result in overflow.
      //* E.g., smallest number: 2^(-126) = 1.1754944e-38, which is subnormal.
      emax_out = MAX(emax_out, 1 - EBIAS);
    }
    uint maxprec = get_precision(emax_out, maxprec, minexp, 2);
    uint biased_emax = (maxprec)? (uint)(emax_out + EBIAS) : 0;

    out_emax.write(emax_out);
    out_maxprec.write(maxprec);
    out_bemax.write(biased_emax);
  }
}


//* Block-level parallel version of `compute_block_exponent_2d`.
void compute_block_emax_2d(
  size_t in_total_blocks,
  hls::stream<fblock_2d_t> in_fblock[FIFO_WIDTH], 
  const zfp_output &output, 
  hls::stream<int> out_emax[FIFO_WIDTH],
  hls::stream<uint> out_bemax[FIFO_WIDTH],
  hls::stream<uint> out_maxprec[FIFO_WIDTH],
  hls::stream<fblock_2d_t> out_fblock[FIFO_WIDTH])
{
#pragma HLS PIPELINE II=1

  emax_dispatch_loop:
  for (uint pe_idx = 0; pe_idx < FIFO_WIDTH; pe_idx++) {
    #pragma HLS UNROLL factor=32
    
    hls::stream<fblock_2d_t> &s_in_fblock = in_fblock[pe_idx];
    #pragma HLS DEPENDENCE variable=in_fblock class=array inter false
    hls::stream<int> &s_out_emax = out_emax[pe_idx];
    #pragma HLS DEPENDENCE variable=out_emax class=array inter false
    hls::stream<uint> &s_out_bemax = out_bemax[pe_idx];
    #pragma HLS DEPENDENCE variable=out_bemax class=array inter false
    hls::stream<uint> &s_out_maxprec = out_maxprec[pe_idx];
    #pragma HLS DEPENDENCE variable=out_maxprec class=array inter false
    hls::stream<fblock_2d_t> &s_out_fblock = out_fblock[pe_idx];
    #pragma HLS DEPENDENCE variable=out_fblock class=array inter false
    
    emax_producer(pe_idx, in_total_blocks, s_in_fblock, output.maxprec, output.minexp, 
      s_out_emax, s_out_bemax, s_out_maxprec, s_out_fblock);
  }
}

void fwd_float2int_2d(
  size_t in_total_blocks,
  hls::stream<int> &in_emax,
  hls::stream<uint> &in_bemax,
  hls::stream<fblock_2d_t> &in_fblock,
  hls::stream<iblock_2d_t> &out_iblock,
  hls::stream<uint> &out_bemax)
{
#pragma HLS INLINE off

  fwd_f2i_block_loop: 
  for (size_t block_id = 0; block_id < in_total_blocks; block_id++) {
    int emax = in_emax.read();
    uint biased_emax = in_bemax.read();
    //* Immediately relay the read emax to the next module.
    out_bemax.write(biased_emax);

    fblock_2d_t fblock_buf = in_fblock.read();
    iblock_2d_t iblock_buf;
    iblock_buf.id = fblock_buf.id;

    //* Encode block only if *biased* exponent is nonzero.
    if (biased_emax) {
      #pragma HLS PIPELINE II=1
      //* Compute power-of-two scale factor for all floats in the block
      //* relative to emax of the block.
      //! Use `emax` instead of `biased_emax`.
      float scale = quantize_scaler(1.0f, emax);
      fwd_f2i_loop: 
      for (uint i = 0; i < BLOCK_SIZE_2D; i++) {
        #pragma HLS UNROLL factor=16

        //* Compute p-bit int y = s*x where x is floating and |y| <= 2^(p-2) - 1
        iblock_buf.data[i] = (int32)(scale * fblock_buf.data[i]);
      }
    }
    //* Relay the block even if it's all zeros.
    //TODO: Find a way to avoid this.
    out_iblock.write(iblock_buf);
  }
}

void f2i_caster(
  uint pe_idx,
  size_t in_total_blocks,
  hls::stream<int> &in_emax,
  hls::stream<uint> &in_bemax,
  hls::stream<fblock_2d_t> &in_fblock,
  hls::stream<iblock_2d_t> &out_iblock,
  hls::stream<uint> &out_bemax)
{
  for (size_t block_id = pe_idx; block_id < in_total_blocks; block_id += FIFO_WIDTH) {
    int emax = in_emax.read();
    uint biased_emax = in_bemax.read();
    //* Immediately relay the read emax to the next module.
    out_bemax.write(biased_emax);

    fblock_2d_t fblock_buf = in_fblock.read();
    iblock_2d_t iblock_buf;
    iblock_buf.id = fblock_buf.id;

    //* Encode block only if *biased* exponent is nonzero.
    if (biased_emax) {
      #pragma HLS PIPELINE II=1
      //* Compute power-of-two scale factor for all floats in the block
      //* relative to emax of the block.
      //! Use `emax` instead of `biased_emax`.
      float scale = quantize_scaler(1.0f, emax);
      fwd_f2i_loop: 
      for (uint i = 0; i < BLOCK_SIZE_2D; i++) {
        #pragma HLS UNROLL factor=16

        //* Compute p-bit int y = s*x where x is floating and |y| <= 2^(p-2) - 1
        iblock_buf.data[i] = (int32)(scale * fblock_buf.data[i]);
      }
    }
    //* Relay the block even if it's all zeros.
    out_iblock.write(iblock_buf);
  }
}

void fwd_float2int_2d_par(
  size_t in_total_blocks,
  hls::stream<int> in_emax[FIFO_WIDTH],
  hls::stream<uint> in_bemax[FIFO_WIDTH],
  hls::stream<fblock_2d_t> in_fblock[FIFO_WIDTH],
  hls::stream<iblock_2d_t> out_iblock[FIFO_WIDTH],
  hls::stream<uint> out_bemax[FIFO_WIDTH])
{
#pragma HLS PIPELINE II=1

  fwd_f2i_dispatch_loop: 
  for (uint pe_idx = 0; pe_idx < FIFO_WIDTH; pe_idx++) {
    #pragma HLS UNROLL factor=8

    hls::stream<int> &in_emax_s = in_emax[pe_idx];
    #pragma HLS DEPENDENCE variable=in_emax class=array inter false
    hls::stream<uint> &in_bemax_s = in_bemax[pe_idx];
    #pragma HLS DEPENDENCE variable=in_bemax class=array inter false
    hls::stream<fblock_2d_t> &in_fblock_s = in_fblock[pe_idx];
    #pragma HLS DEPENDENCE variable=in_fblock class=array inter false
    hls::stream<iblock_2d_t> &out_iblock_s = out_iblock[pe_idx];
    #pragma HLS DEPENDENCE variable=out_iblock class=array inter false
    hls::stream<uint> &out_bemax_s = out_bemax[pe_idx];
    #pragma HLS DEPENDENCE variable=out_bemax class=array inter false

    f2i_caster(pe_idx, in_total_blocks, 
      in_emax_s, in_bemax_s, in_fblock_s, out_iblock_s, out_bemax_s);
  }
}

/**
 * @brief Map floating-point number x to integer relative to exponent e.
 * @param x Floating-point number.
 * @param e Exponent.
 * @return Integer in floating-point format.
 * @note When e is block-floating-point exponent (emax), this function maps
 *  all floats x relative to emax of the block.
*/
float quantize_scaler(float x, int e)
{
#pragma HLS inline
  //* `((int)(8 * sizeof(float)) - 2) - e` calculates the difference in exponents to
  //* achieve the desired quantization relative to e.
  return LDEXP(x, ((int)(CHAR_BIT * sizeof(float)) - 2) - e);
}

void fwd_cast(hls::stream<int32> &out_integer, hls::stream<float> &in_float, uint dim, uint emax)
{
  size_t block_size = BLOCK_SIZE(dim);

  //* Compute power-of-two scale factor for all floats in the block
  //* relative to emax of the block.
  float scale = quantize_scaler(1.0f, emax);
  fwd_cast_loop: for (; block_size--;) {
    #pragma HLS PIPELINE II=1

    //? Compute p-bit int y = s*x where x is floating and |y| <= 2^(p-2) - 1
    out_integer << (int32)(scale * in_float.read());
  }
}

void fwd_lift_vector(volatile int32 *p, ptrdiff_t s)
{
#pragma HLS INLINE

  //* Gather 4-vector [x y z w] from p.
  int32 x, y, z, w;
  x = *p;
  p += s;
  y = *p;
  p += s;
  z = *p;
  p += s;
  w = *p;
  p += s;

  /*
  ** non-orthogonal transform
  **        ( 4  4  4  4) (x)
  ** 1/16 * ( 5  1 -1 -5) (y)
  **        (-4  4  4 -4) (z)
  **        (-2  6 -6  2) (w)

  * The above is essentially the transform matix
  * (similar to the cosine transform matrix in JPEG)
  */
  x += w;
  x >>= 1;
  w -= x;
  z += y;
  z >>= 1;
  y -= z;
  x += z;
  x >>= 1;
  z -= x;
  w += y;
  w >>= 1;
  y -= w;
  w += y >> 1;
  y -= w >> 1;

  /**
  * ^ After applying the transform matrix
  * ^ on a 4-vector (x = 10, y = 20, z = 30, and w = 40):
    x' = (4x + 4y + 4z + 4w)  / 16 = 25
    y' = (5x + y - z - 5w)    / 16 = -11.875
    z' = (-4x + 4y + 4z - 4w) / 16 = 2.5
    w' = (-2x + 6y - 6z + 2w) / 16 = 8.75
   */

  p -= s;
  *p = w;
  p -= s;
  *p = z;
  p -= s;
  *p = y;
  p -= s;
  *p = x;

  /**
  * ^ The result is then stored in reverse order.
    [w', z', y', x'] = [25, -11.875, 2.5, 8.75]
  */
}

void fwd_decorrelate_2d(
  size_t in_total_blocks,
  hls::stream<uint> &in_bemax,
  hls::stream<iblock_2d_t> &in_iblock,
  hls::stream<iblock_2d_t> &out_iblock,
  hls::stream<uint> &out_bemax)
{
  fwd_decorrelate_block_loop: 
  for (size_t block_id = 0; block_id < in_total_blocks; block_id++) {
    #pragma HLS PIPELINE II=1

    //& II=4 w/o optimization.
    //* Blocking reads.
    uint biased_emax = in_bemax.read();
    iblock_2d_t iblock_buf = in_iblock.read();
    out_bemax.write(biased_emax);

    //* Encode block only if biased exponent is nonzero.
    if (biased_emax) {
      fwd_decorrelate_2d_block(iblock_buf.data);
    }
    //* Relay the block even if nothing is done.
    out_iblock.write(iblock_buf);
  }
}

void decorrelater(
  uint pe_idx,
  size_t in_total_blocks,
  hls::stream<uint> &in_bemax,
  hls::stream<iblock_2d_t> &in_iblock,
  hls::stream<iblock_2d_t> &out_iblock,
  hls::stream<uint> &out_bemax)
{
#pragma HLS INLINE off

  decorrelater_loop:
  for (size_t block_id = pe_idx; block_id < in_total_blocks; block_id += FIFO_WIDTH) {
    uint biased_emax = in_bemax.read();
    iblock_2d_t iblock_buf = in_iblock.read();
    out_bemax.write(biased_emax);

    //* Encode block only if biased exponent is nonzero.
    if (biased_emax) {
      fwd_decorrelate_2d_block(iblock_buf.data);
    }
    //* Relay the block even if nothing is done.
    out_iblock.write(iblock_buf);
  }
}

void fwd_decorrelate_2d_par(
  size_t in_total_blocks,
  hls::stream<uint> in_bemax[FIFO_WIDTH],
  hls::stream<iblock_2d_t> in_iblock[FIFO_WIDTH],
  hls::stream<iblock_2d_t> out_iblock[FIFO_WIDTH],
  hls::stream<uint> out_bemax[FIFO_WIDTH])
{
#pragma HLS PIPELINE II=1

  fwd_decorrelate_dispatch_loop: 
  for (uint pe_idx = 0; pe_idx < FIFO_WIDTH; pe_idx++) {
    #pragma HLS UNROLL factor=8

    hls::stream<uint> &in_bemax_s = in_bemax[pe_idx];
    #pragma HLS DEPENDENCE variable=in_bemax class=array inter false
    hls::stream<iblock_2d_t> &in_iblock_s = in_iblock[pe_idx];
    #pragma HLS DEPENDENCE variable=in_iblock class=array inter false
    hls::stream<iblock_2d_t> &out_iblock_s = out_iblock[pe_idx];
    #pragma HLS DEPENDENCE variable=out_iblock class=array inter false
    hls::stream<uint> &out_bemax_s = out_bemax[pe_idx];
    #pragma HLS DEPENDENCE variable=out_bemax class=array inter false

    decorrelater(pe_idx, in_total_blocks, 
      in_bemax_s, in_iblock_s, out_iblock_s, out_bemax_s);
  }
}

void fwd_decorrelate_2d_block(volatile int32 *iblock)
{
// #pragma HLS INLINE
#pragma PIPELINE II=1

  uint x, y;
  /* transform along x */
  fwd_decorrelate_x_loop: for (y = 0; y < 4; y++) {
    #pragma HLS UNROLL factor=4
    // #pragma HLS PIPELINE II=1
    fwd_lift_vector(iblock + 4 * y, 1);
  }
  /* transform along y */
  fwd_decorrelate_y_loop: for (x = 0; x < 4; x++) {
    #pragma HLS UNROLL factor=4
    // #pragma HLS PIPELINE II=1
    fwd_lift_vector(iblock + 1 * x, 4);
  }
}

/* Map two's complement signed integer to negabinary unsigned integer */
uint32 twoscomplement_to_negabinary(int32 x)
{
#pragma HLS INLINE

  return ((uint32)x + NBMASK) ^ NBMASK;
}


void fwd_reorder_int2uint_2d(
  size_t in_total_blocks,
  hls::stream<uint> &in_bemax,
  hls::stream<iblock_2d_t> &in_iblock,
  hls::stream<ublock_2d_t> &out_ublock,
  hls::stream<uint> &out_bemax)
{
  fwd_reorder_block_loop: for (size_t block_id = 0; block_id < in_total_blocks; block_id++) {
    //* Blocking reads.
    uint biased_emax = in_bemax.read();
    iblock_2d_t iblock_buf = in_iblock.read();
    out_bemax.write(biased_emax);

    ublock_2d_t ublock_buf;
    ublock_buf.id = iblock_buf.id;

    //* Encode block only if biased exponent is nonzero.
    if (biased_emax) {
      fwd_reorder_int2uint_block(ublock_buf.data, iblock_buf.data, PERM_2D, BLOCK_SIZE_2D);
    }
    //* Relay the block even if nothing is done.
    out_ublock.write(ublock_buf);
  }
}

void reorder_i2u_producer(
  uint pe_idx,
  size_t in_total_blocks,
  hls::stream<uint> &in_bemax,
  hls::stream<iblock_2d_t> &in_iblock,
  hls::stream<ublock_2d_t> &out_ublock,
  hls::stream<uint> &out_bemax)
{
#pragma HLS INLINE off

  fwd_reorder_producer_loop: 
  for (size_t block_id = pe_idx; block_id < in_total_blocks; block_id += FIFO_WIDTH) {
    uint biased_emax = in_bemax.read();
    iblock_2d_t iblock_buf = in_iblock.read();
    out_bemax.write(biased_emax);

    ublock_2d_t ublock_buf;
    ublock_buf.id = iblock_buf.id;

    //* Encode block only if biased exponent is nonzero.
    if (biased_emax) {
      fwd_reorder_int2uint_block(ublock_buf.data, iblock_buf.data, PERM_2D, BLOCK_SIZE_2D);
    }
    //* Relay the block even if nothing is done.
    out_ublock.write(ublock_buf);
  }
}

void fwd_reorder_int2uint_2d_par(
  size_t in_total_blocks,
  hls::stream<uint> in_bemax[FIFO_WIDTH],
  hls::stream<iblock_2d_t> in_iblock[FIFO_WIDTH],
  hls::stream<ublock_2d_t> out_ublock[FIFO_WIDTH],
  hls::stream<uint> out_bemax[FIFO_WIDTH])
{
#pragma HLS PIPELINE II=1

  fwd_reorder_dispatch_loop: 
  for (uint pe_idx = 0; pe_idx < FIFO_WIDTH; pe_idx++) {
    #pragma HLS UNROLL factor=2

    hls::stream<uint> &in_bemax_s = in_bemax[pe_idx];
    #pragma HLS DEPENDENCE variable=in_bemax class=array inter false
    hls::stream<iblock_2d_t> &in_iblock_s = in_iblock[pe_idx];
    #pragma HLS DEPENDENCE variable=in_iblock class=array inter false
    hls::stream<ublock_2d_t> &out_ublock_s = out_ublock[pe_idx];
    #pragma HLS DEPENDENCE variable=out_ublock class=array inter false
    hls::stream<uint> &out_bemax_s = out_bemax[pe_idx];
    #pragma HLS DEPENDENCE variable=out_bemax class=array inter false

    reorder_i2u_producer(pe_idx, in_total_blocks, 
      in_bemax_s, in_iblock_s, out_ublock_s, out_bemax_s);
  }
}

/* Reorder signed coefficients and convert to unsigned integer */
void fwd_reorder_int2uint_block(volatile uint32 *ublock, volatile const int32 *iblock,
                          const uchar* perm, uint n)
{
#pragma HLS INLINE

  //TODO: Storing perm locally.
  fwd_reorder_i2u_loop: for (; n--;) {
    *ublock++ = twoscomplement_to_negabinary(iblock[*perm++]);
  }
  // while (n--);
  //? Why `--n` doesn't pass the emulation test? (last value is always 0)
}

/* Compress <= 64 (1-3D) unsigned integers with rate contraint */
void encode_partial_bitplanes(volatile const uint32 *const ublock,
                              hls::stream<write_request_t> &write_queue, 
                              size_t block_id, uint &index,
                              uint maxbits, uint maxprec, uint block_size, uint *encoded_bits)
{
  uint intprec = (uint)(CHAR_BIT * sizeof(int32));
  //* `kmin` is the cutoff of the least significant bit plane to encode.
  uint kmin = intprec > maxprec ? intprec - maxprec : 0;
  uint bits = maxbits;
  uint k, m, n;
  uint bit;
  uint64 x;

  //* Encode one bit plane at a time from MSB to LSB
  partial_bitplanes_loop: for (k = intprec, n = 0; bits && k-- > kmin;) {
    //^ Step 1: Extract bit plane #k to x
    x = 0;
    partial_bitplanes_transpose_loop: for (uint i = 0; i < block_size; i++)
      //* Below puts the `k`th bit of `data[i]` into the `i`th bit of `x`.
      //* I.e., transposing into the bit plane format.
      x += (uint64)((ublock[i] >> k) & 1u) << i;

    //^ Step 2: Encode first n bits of bit plane verbatim.
    //* Bound the total encoded bit `n` by the `maxbits`.
    //& `n` is the number of bits in `x` that have been encoded so far.
    m = MIN(n, bits);
    bits -= m;
    //* (The first n bits "are encoded verbatim.")
    write_queue.write( write_request_t(block_id, index++, m, x, false) );
    x >>= m;
    // stream_write_bits(s, x, m, &x);

    //^ Step 3: Bitplane embedded (unary run-length) encode remainder of bit plane.
    //* Shift `x` right by 1 bit and increment `n` until `x` becomes 0.
    partial_bitplanes_embed_loop: for (; bits && n < block_size; x >>= 1, n++) {
      //* The number of bits in `x` still to be encoded.
      bits--;
      //* Group test: If `x` is not 0, then write a 1 bit.
      //& `!!` is used to convert a value to its corresponding boolean value, e.g., !!5 == true.
      //& `stream_write_bit()` returns the bit (1/0) that was written.
      bit = !!x;
      write_queue.write( write_request_t(block_id, index++, 1, bit, false) );
      // stream_write_bit(s, bit, &bit);
      if (bit) {
        //^ Positive group test (x != 0) -> Scan for one-bit.
        partial_bitplanes_embed_inner_loop: for (; bits && n < block_size - 1; x >>= 1, n++) {
          //* `n` is incremented for every encoded bit and accumulated across all bitplanes.
          bits--;
          //* Continue writing 0's until a 1 bit is found.
          //& `x & 1u` is used to extract the least significant (right-most) bit of `x`.
          bit = x & 1u;
          write_queue.write( write_request_t(block_id, index++, 1, bit, false) );
          // stream_write_bit(s, bit, &bit);
          if (bit)
            //* After writing a 1 bit, break out for another group test
            //* (to see whether the bitplane code `x` turns 0 after encoding `n` of its bits).
            break;
        }
      } else {
        //^ Negative group test (x == 0) -> Done with bit plane.
        break;
      }
    }
  }
  // //* Write an empty request to signal the end of this stage.
  // write_queue.write( write_request_t(block_id, 0, (uint64)0, true/* last bit for this stage */) );

  //* Returns the number of bits written (constrained by `maxbits`).
  *encoded_bits = maxbits - bits;
}

/* Compress <= 64 (1-3D) unsigned integers without rate contraint */
void encode_all_bitplanes(volatile const uint32 *const ublock,
                          hls::stream<write_request_t> &write_queue, 
                          size_t block_id, uint &index,
                          uint maxprec, uint block_size, uint *encoded_bits)
{
  // uint64 offset = stream_woffset(s);
  uint64 bits = 0;
  uint intprec = (uint)(CHAR_BIT * sizeof(uint32));
  //* `kmin` is the cutoff of the least significant bit plane to encode.
  //* E.g., if `intprec` is 32 and `maxprec` is 17, only [31:16] bit planes are encoded.
  uint kmin = intprec > maxprec ? intprec - maxprec : 0;
  uint bit;
  uint k, n;

  /* encode one bit plane at a time from MSB to LSB */
  all_bitplanes_loop: 
  for (k = intprec, n = 0; k-- > kmin;) {
    //^ Step 1: extract bit plane #k of every block to x.
    stream_word x(0);

    all_bitplanes_transpose_loop: 
    for (uint i = 0; i < block_size; i++) {
      #pragma HLS UNROLL factor=16

      x += ( (stream_word(ublock[i]) >> k) & stream_word(1) ) << i;
    }

    //^ Step 2: encode first n bits of bit plane.
    bits += n;
    write_queue.write( write_request_t(block_id, index++, n, x, false) );
    // if (n > 0) {
    //   std::bitset<64> b(x);
    //   std::cout << "Encoded " << n << " bits: " << b << std::endl;
    // }
    x >>= n;

    //^ Step 3: unary run-length encode remainder of bit plane.
    all_bitplanes_embed_loop: for (; n < block_size; x >>= 1, n++) {
      bit = !!x;
      write_queue.write( write_request_t(block_id, index++, 1, bit, false) );
      // std::cout << "Encoded: " << bit << std::endl;
      bits++;
      // stream_write_bit(s, bit, &bit);
      if (!bit) {
        //^ Negative group test (x == 0) -> Done with all bit planes.
        break;
      }
      all_bitplanes_embed_inner_loop: for (; n < block_size - 1; x >>= 1, n++) {
        //* Continue writing 0's until a 1 bit is found.
        //& `x & 1u` is used to extract the least significant (right-most) bit of `x`.
        bit = x & stream_word(1);
        write_queue.write( write_request_t(block_id, index++, 1, bit, false) );
        // std::cout << "Encoded: " << bit << std::endl;
        bits++;
        // stream_write_bit(s, bit, &bit);
        if (bit) {
          //* After encoding a 1 bit, break out for another group test
          //* (to see whether the bitplane code `x` turns 0 after encoding `n` of its bits).
          //* I.e., for every 1 bit encoded, do a group test on the rest.
          break;
        }
      }
    }
  }
  // //* Write an empty request to signal the end of this stage.
  // write_queue.write( write_request_t(block_id, 0, (uint64)0, true/* last bit for this stage */) );
  
  //* Returns the number of bits written.
  *encoded_bits = bits;
  // *encoded_bits = (uint)(stream_woffset(s) - offset);
}

void encode_bitplanes_2d(
  size_t in_total_blocks,
  hls::stream<uint> &in_bemax,
  hls::stream<uint> &in_maxprec,
  hls::stream<ublock_2d_t> &in_ublock,
  zfp_output &output,
  hls::stream<write_request_t> &bitplane_queue)
{
  encode_bitplanes_loop: 
  for (size_t block_id = 0; block_id < in_total_blocks; block_id++) {
    #pragma HLS PIPELINE II=1
    #pragma HLS UNROLL factor=16

    //* Blocking reads.
    uint biased_emax = in_bemax.read();
    uint maxprec = in_maxprec.read();
    ublock_2d_t ublock_buf = in_ublock.read();

    uint bits = 1;
    uint index = 0;
    uint minbits = output.minbits; 
    //* Encode block only if *biased* exponent is nonzero.
    if (biased_emax) {
      //~ First, encode block exponent.
      bits += EBITS;
      bitplane_queue.write(
        //! Encode biased emax (NOT emax itself).
        write_request_t(block_id, index++, bits, (uint64)(2 * biased_emax + 1), false));

      uint encoded_bits = 0;
      uint maxbits = output.maxbits - bits;
      //* Adjust the minimum number of bits required.
      minbits -= MIN(bits, output.minbits);
      //~ Bitplane coding with the fastest implementation.
      if (exceeded_maxbits(maxbits, maxprec, BLOCK_SIZE_2D)) {
        //* Encode partial bitplanes with rate constraint.
        encode_partial_bitplanes(
          ublock_buf.data, bitplane_queue, ublock_buf.id, index, maxbits, maxprec, BLOCK_SIZE_2D, &encoded_bits);
      } else {
        //* Encode all bitplanes without rate constraint.
        encode_all_bitplanes(
          ublock_buf.data, bitplane_queue, ublock_buf.id, index, maxprec, BLOCK_SIZE_2D, &encoded_bits);
      }
      bits += encoded_bits;
    } else {
      //* Write single zero-bit to encode the entire block.
      bitplane_queue.write(
        write_request_t(block_id, index++, bits, (uint64)0, false));
    }

    //~ Encode padding bits.
    if (bits < minbits) {
      bitplane_queue.write(
        write_request_t(block_id, index++, minbits - bits, (uint64)0, true));
      bits = minbits;
    } else {
      //* Write an empty request to signal the end of this stage.
      bitplane_queue.write(
        write_request_t(block_id, index++, 0, (uint64)0, true/* last bit for this stage */));
    }
  }
}

void embeded_coding(
  size_t pe_idx,
  size_t total_blocks,
  uint in_minbits,
  uint in_maxbits,
  hls::stream<uint> &bemax,
  hls::stream<uint> &maxprec,
  hls::stream<ublock_2d_t> &ublock,
  hls::stream<write_request_t> &bitplane_queue)
{
// #pragma HLS INLINE recursive

  embeded_coding_loop:
  for (size_t block_id = pe_idx; block_id < total_blocks; block_id += FIFO_WIDTH) {

    uint bits = 1;
    uint index = 0;
    uint minbits = in_minbits; 
    //* Blocking reads.
    uint biased_emax = bemax.read();
    uint prec = maxprec.read();
    ublock_2d_t ublock_buf = ublock.read();
    #pragma HLS BIND_STORAGE variable=ublock_buf impl=SLR

    //* Encode block only if *biased* exponent is nonzero.
    if (biased_emax) {
      //~ First, encode block exponent.
      bits += EBITS;
      bitplane_queue.write(
        //! Encode biased emax (NOT emax itself).
        write_request_t(block_id, index++, bits, (uint64)(2 * biased_emax + 1), false));

      uint encoded_bits = 0;
      #pragma HLS BIND_STORAGE variable=encoded_bits impl=SLR
      uint maxbits = in_maxbits - bits;
      //* Adjust the minimum number of bits required.
      minbits -= MIN(bits, in_minbits);
      //~ Bitplane coding with the fastest implementation.
      // if (exceeded_maxbits(maxbits, prec, BLOCK_SIZE_2D)) {
      //   //* Encode partial bitplanes with rate constraint.
      //   encode_partial_bitplanes(
      //     ublock_buf.data, bitplane_queue, ublock_buf.id, index, maxbits, prec, BLOCK_SIZE_2D, &encoded_bits);
      // } else {
        //* Encode all bitplanes without rate constraint.
        encode_all_bitplanes(
          ublock_buf.data, bitplane_queue, ublock_buf.id, index, prec, BLOCK_SIZE_2D, &encoded_bits);
      // }
      bits += encoded_bits;
    } else {
      //* Write single zero-bit to encode the entire block.
      bitplane_queue.write(
        write_request_t(block_id, index++, bits, (uint64)0, false));
    }

    //~ Encode padding bits.
    if (bits < minbits) {
      bitplane_queue.write(
        write_request_t(block_id, index++, minbits - bits, (uint64)0, true));
      bits = minbits;
    } else {
      //* Write an empty request to signal the end of this stage.
      bitplane_queue.write(
        write_request_t(block_id, index++, 0, (uint64)0, true/* last bit for this stage */));
    }
  }
}

void encode_bitplanes_2d_par(
  size_t in_total_blocks,
  hls::stream<uint> in_bemax[FIFO_WIDTH],
  hls::stream<uint> in_maxprec[FIFO_WIDTH],
  hls::stream<ublock_2d_t> in_ublock[FIFO_WIDTH],
  zfp_output &output,
  hls::stream<write_request_t> bitplane_queues[FIFO_WIDTH])
{
//* Make sure the elements in the loop body are separate PEs.
#pragma HLS PIPELINE II=1

  size_t total_blocks = in_total_blocks;
  uint minbits = output.minbits;
  uint maxbits = output.maxbits;

  encode_bitplanes_loop:
  for (uint pe_idx = 0; pe_idx < FIFO_WIDTH; pe_idx++) {
    //* Fully unroll the loop body.
    #pragma HLS UNROLL

    hls::stream<uint> &bemax = in_bemax[pe_idx];
    #pragma HLS DEPENDENCE variable=in_bemax class=array inter false
    hls::stream<uint> &maxprec = in_maxprec[pe_idx];
    #pragma HLS DEPENDENCE variable=in_maxprec class=array inter false
    hls::stream<ublock_2d_t> &ublock = in_ublock[pe_idx];
    #pragma HLS DEPENDENCE variable=in_ublock class=array inter false
    hls::stream<write_request_t> &bitplane_queue = bitplane_queues[pe_idx];
    #pragma HLS DEPENDENCE variable=bitplane_queues class=array inter false

    embeded_coding(pe_idx, total_blocks, minbits, maxbits, bemax, maxprec, ublock, bitplane_queue);
  }
}



void chunk_blocks_2d(hls::stream<fblock_2d_t> &fblock, const zfp_input &input)
{
  size_t nx = input.nx;
  size_t ny = input.ny;
  ptrdiff_t sx = input.sx ? input.sx : 1;
  ptrdiff_t sy = input.sy ? input.sy : (ptrdiff_t)nx;

  size_t block_id = 0;
  chunk_blocks_outer: 
  for (size_t y = 0; y < ny; y += 4) {
    chunk_blocks_inner: 
    for (size_t x = 0; x < nx; x += 4, block_id++) {
      // #pragma HLS UNROLL factor=16
      #pragma HLS PIPELINE II=1

      const float *raw = input.data + sx * (ptrdiff_t)x + sy * (ptrdiff_t)y;
      fblock_2d_t fblock_buf;
      fblock_buf.id = block_id;
      size_t bx = MIN(nx - x, 4u);
      size_t by = MIN(ny - y, 4u);

      if (bx == 4 && by == 4) {
        gather_2d_block(fblock_buf.data, raw, sx, sy);
      } else {
        gather_partial_2d_block(fblock_buf.data, raw, bx, by, sx, sy);
      }

      fblock.write(fblock_buf);
    }
  }
}

void chunk_2d(
  const float *data,
  size_t total_blocks,
  uint pe_idx,
  // size_t block_id,
  // size_t x, size_t y,
  size_t nx, size_t ny,
  ptrdiff_t sx, ptrdiff_t sy,
  hls::stream<fblock_2d_t> &fblock)
{
#pragma HLS INLINE off

  chunk_blocks_outer: 
    for (size_t y = 0; y < ny; y += 4) {
      chunk_blocks_inner: 
      for (size_t x = 4*pe_idx; x < nx; x += 4*(FIFO_WIDTH)) {
        #pragma HLS PIPELINE II=1

        size_t block_id = x/4 + y/4 * (nx/4);

        std::cout << "PE " << pe_idx << ": [" << block_id << "] x: " << x << " y: " << y << std::endl;

        fblock_2d_t fblock_buf;
        fblock_buf.id = block_id;
        size_t bx = MIN(nx - x, 4u);
        size_t by = MIN(ny - y, 4u);
        const float *raw = data + sx * (ptrdiff_t)x + sy * (ptrdiff_t)y;

        if (bx == 4 && by == 4) {
          std::cout << "Full Gathering" << std::endl;
          gather_2d_block(fblock_buf.data, raw, sx, sy);
        } else {
          std::cout << "Partial Gathering" << std::endl;
          gather_partial_2d_block(fblock_buf.data, raw, bx, by, sx, sy);
        }

        fblock.write(fblock_buf);
      }
  }
}

void chunk_blocks_2d_par(
  size_t total_blocks,
  hls::stream<fblock_2d_t> fblocks[FIFO_WIDTH], 
  const zfp_input &input)
{
#pragma HLS PIPELINE II=1

  // std::cout << "Total blocks: " << total_blocks << std::endl;

  size_t nx = input.nx;
  size_t ny = input.ny;
  ptrdiff_t sx = input.sx ? input.sx : 1;
  ptrdiff_t sy = input.sy ? input.sy : (ptrdiff_t)nx;
  const float *data = input.data;
  #pragma HLS DEPENDENCE variable=input inter false

  chunk_dispatch_loop:
  for (uint pe_idx = 0; pe_idx < FIFO_WIDTH; pe_idx++) {
    #pragma HLS UNROLL factor=2
    
    hls::stream<fblock_2d_t> &fblock = fblocks[pe_idx];
    #pragma HLS DEPENDENCE variable=fblocks class=array inter false

    chunk_2d(data, total_blocks, pe_idx, nx, ny, sx, sy, fblock);
  }
}

void chunk_2d(
  const float *data,
  size_t block_id,
  size_t x, size_t y,
  size_t nx, size_t ny,
  ptrdiff_t sx, ptrdiff_t sy,
  hls::stream<fblock_2d_t> &fblock)
{
  fblock_2d_t fblock_buf;
  fblock_buf.id = block_id;
  size_t bx = MIN(nx - x, 4u);
  size_t by = MIN(ny - y, 4u);
  const float *raw = data + sx * (ptrdiff_t)x + sy * (ptrdiff_t)y;

  if (bx == 4 && by == 4) {
    gather_2d_block(fblock_buf.data, raw, sx, sy);
  } else {
    gather_partial_2d_block(fblock_buf.data, raw, bx, by, sx, sy);
  }

  fblock.write(fblock_buf);
}

void chunk_blocks_2d_seq(
  hls::stream<fblock_2d_t> fblocks[FIFO_WIDTH], 
  const zfp_input &input)
{
#pragma HLS DATAFLOW

  size_t nx = input.nx;
  size_t ny = input.ny;
  ptrdiff_t sx = input.sx ? input.sx : 1;
  ptrdiff_t sy = input.sy ? input.sy : (ptrdiff_t)nx;

  size_t block_id = 0;
  chunk_blocks_outer: 
  for (size_t y = 0; y < ny; y += 4) {
    chunk_blocks_inner: 
    for (size_t x = 0; x < nx; x += 4, block_id++) {
      #pragma HLS PIPELINE II=1

      uint fifo_idx = FIFO_INDEX(block_id);
      hls::stream<fblock_2d_t> &fblock = fblocks[fifo_idx];
      #pragma HLS DEPENDENCE variable=fblock inter false
      
      chunk_2d(input.data, block_id, x, y, nx, ny, sx, sy, fblock);
    }
  }
}