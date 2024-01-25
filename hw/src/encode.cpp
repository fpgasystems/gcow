// Description: Encoding functions for ZFP compression.
// Documentation: ./include/encode.hpp

#include <stddef.h>
#include <stdio.h>
#include <math.h>

#include "encode.hpp"
#include "stream.hpp"
#include "types.hpp"
#include "io.hpp"


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
#pragma HLS inline

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

void gather_2d_block(hls::stream<float> fblock[BLOCK_SIZE_2D], const float *raw,
                     ptrdiff_t sx, ptrdiff_t sy)
{
LOOP_GATHER_2D_BLOCK:
  for (size_t y = 0; y < 4; y++, raw += sy - 4 * sx)
LOOP_GATHER_2D_BLOCK_INNER:
    for (size_t x = 0, i = 0; x < 4; x++, raw += sx, i++) {
      fblock[i] << *raw;
    }
}

void gather_partial_2d_block(volatile float *block, volatile const float *raw,
                             size_t nx, size_t ny,
                             ptrdiff_t sx, ptrdiff_t sy)
{
  size_t x, y;
LOOP_GATHER_PARTIAL_2D_BLOCK:
  for (y = 0; y < ny; y++, raw += sy - (ptrdiff_t)nx * sx) {
LOOP_GATHER_PARTIAL_2D_BLOCK_INNER:
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

void compute_block_exponent_2d(
  size_t in_total_blocks,
  hls::stream<fblock_2d_t> &in_fblock, 
  const zfp_output &output, 
  hls::stream<uint> &out_emax,
  hls::stream<uint> &out_maxprec,
  hls::stream<fblock_2d_t> &out_fblock)
{
  emax_block_loop: for (size_t block_id = 0; block_id < in_total_blocks; block_id++) {
    //* Blocking read.
    fblock_2d_t fblock_buf = in_fblock.read();
    //* Immediately relay the read block to the next module.
    out_fblock.write(fblock_buf);

    float emax = 0;
    //~ 1: Find the maximum floating point and return its exponent as the block exponent.
    emax_loop: for (uint i = 0; i < BLOCK_SIZE_2D; i++) {
      #pragma HLS pipeline II=2
      float f = FABS(fblock_buf.data[i]);
      emax = MAX(emax, f);
    }

    //~ 2: Get the exponent of the maximum floating point.
    int e = -EBIAS;
    if (emax > 0) {
      //* Get exponent of emax.
      FREXP(emax, &e);
      //* Clamp exponent in case x is subnormal; may still result in overflow.
      //* E.g., smallest number: 2^(-126) = 1.1754944e-38, which is subnormal.
      e = MAX(e, 1 - EBIAS);
    }
    uint maxprec = get_precision(e, output.maxprec, output.minexp, 2);
    //! Block exponent: float -> int -> uint.
    uint e_out = maxprec ? (uint)(e + EBIAS) : 0;

    out_maxprec.write(maxprec);
    out_emax.write(e_out);
  }
}

void fwd_float2int_2d(
  size_t in_total_blocks,
  hls::stream<uint> &in_emax,
  hls::stream<fblock_2d_t> &in_fblock,
  hls::stream<iblock_2d_t> &out_iblock,
  hls::stream<uint> &out_emax)
{
  fwd_f2i_block_loop: for (size_t block_id = 0; block_id < in_total_blocks; block_id++) {
    //* Blocking reads.
    uint emax = in_emax.read();
    fblock_2d_t fblock_buf = in_fblock.read();
    iblock_2d_t iblock_buf;
    iblock_buf.id = fblock_buf.id;

    //* Encode block only if biased exponent is nonzero.
    if (emax) {
      //* Compute power-of-two scale factor for all floats in the block
      //* relative to emax of the block.
      float scale = quantize_scaler(1.0f, emax);
      fwd_f2i_loop: for (uint i = 0; i < BLOCK_SIZE_2D; i++) {
        #pragma HLS pipeline II=2

        //* Compute p-bit int y = s*x where x is floating and |y| <= 2^(p-2) - 1
        iblock_buf.data[i] = (int32)(scale * fblock_buf.data[i]);
      }
    }
    //* Relay the block even if it's all zeros.
    //TODO: Find a way to avoid this.
    out_iblock.write(iblock_buf);
    out_emax.write(emax);
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
    #pragma HLS pipeline II=1

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
  hls::stream<uint> &in_emax,
  hls::stream<iblock_2d_t> &in_iblock,
  hls::stream<iblock_2d_t> &out_iblock,
  hls::stream<uint> &out_emax)
{
  fwd_decorrelate_block_loop: for (size_t block_id = 0; block_id < in_total_blocks; block_id++) {
    //* Blocking reads.
    uint emax = in_emax.read();
    iblock_2d_t iblock_buf = in_iblock.read();

    //* Encode block only if biased exponent is nonzero.
    if (emax) {
      fwd_decorrelate_2d_block(iblock_buf.data);
    }
    //* Relay the block even if nothing is done.
    out_iblock.write(iblock_buf);
    out_emax.write(emax);
  }
}

void fwd_decorrelate_2d_block(volatile int32 *iblock)
{
#pragma HLS INLINE

  uint x, y;
  /* transform along x */
  fwd_decorrelate_x_loop: for (y = 0; y < 4; y++)
    fwd_lift_vector(iblock + 4 * y, 1);
  /* transform along y */
  fwd_decorrelate_y_loop: for (x = 0; x < 4; x++)
    fwd_lift_vector(iblock + 1 * x, 4);
}

/* Map two's complement signed integer to negabinary unsigned integer */
uint32 twoscomplement_to_negabinary(int32 x)
{
#pragma HLS INLINE

  return ((uint32)x + NBMASK) ^ NBMASK;
}


void fwd_reorder_int2uint_2d(
  size_t in_total_blocks,
  hls::stream<uint> &in_emax,
  hls::stream<iblock_2d_t> &in_iblock,
  hls::stream<ublock_2d_t> &out_ublock,
  hls::stream<uint> &out_emax)
{
  fwd_reorder_block_loop: for (size_t block_id = 0; block_id < in_total_blocks; block_id++) {
    //* Blocking reads.
    uint emax = in_emax.read();
    iblock_2d_t iblock_buf = in_iblock.read();
    ublock_2d_t ublock_buf;
    ublock_buf.id = iblock_buf.id;

    //* Encode block only if biased exponent is nonzero.
    if (emax) {
      fwd_reorder_int2uint_block(ublock_buf.data, iblock_buf.data, PERM_2D, BLOCK_SIZE_2D);
    }
    //* Relay the block even if nothing is done.
    out_ublock.write(ublock_buf);
    out_emax.write(emax);
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
  /* Make a copy of bit stream to avoid aliasing */
  //! CHANGE: No copy is made here!
  // stream s(out_data);
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
  all_bitplanes_loop: for (k = intprec, n = 0; k-- > kmin;) {
    //^ Step 1: extract bit plane #k of every block to x.
    stream_word x(0);

    all_bitplanes_transpose_loop: for (uint i = 0; i < block_size; i++) {
      x += ( (stream_word(ublock[i]) >> k) & stream_word(1) ) << i;
    }

    //^ Step 2: encode first n bits of bit plane.
    bits += n;
    write_queue.write( write_request_t(block_id, index++, n, x, false) );
    x >>= n;

    //^ Step 3: unary run-length encode remainder of bit plane.
    all_bitplanes_embed_loop: for (; n < block_size; x >>= 1, n++) {
      bit = !!x;
      write_queue.write( write_request_t(block_id, index++, 1, bit, false) );
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
        bits++;
        // stream_write_bit(s, bit, &bit);
        if (bit) {
          //* After writing a 1 bit, break out for another group test
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
  hls::stream<uint> &in_emax,
  hls::stream<uint> &in_maxprec,
  hls::stream<ublock_2d_t> &in_ublock,
  zfp_output &output,
  hls::stream<write_request_t> &bitplane_queue)
{
  encode_bitplanes_block_loop: for (size_t block_id = 0; block_id < in_total_blocks; block_id++) {
    //* Blocking reads.
    uint bits = 1;
    uint emax = in_emax.read();
    uint maxprec = in_maxprec.read();
    ublock_2d_t ublock_buf = in_ublock.read();
    uint minbits = output.minbits; 
    uint index = 0;

    //* Encode block only if biased exponent is nonzero.
    if (emax) {
      //~ First, encode block exponent.
      bits += EBITS;
      bitplane_queue.write(
        write_request_t(block_id, index++, bits, (uint64)(2 * emax + 1), false));

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
      // bitplane_queue.write(write_request_t(block_id, bits, (uint64)0, true));
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

// void encode_padding(
//   size_t in_total_blocks,
//   hls::stream<uint> &in_bits,
//   hls::stream<uint> &in_minbits,
//   hls::stream<write_request_t> &padding_queue)
// {
//   encode_padding_block_loop: for (size_t block_id = 0; block_id < in_total_blocks; block_id++) {
//     //* Blocking reads.
//     uint bits = in_bits.read();
//     uint minbits = in_minbits.read();

//     //* Padding to the minimum number of bits required.
//     if (bits < minbits) {
//       padding_queue.write(
//         write_request_t(block_id, minbits - bits, (uint64)0, true));
//     } else {
//       //* Write an empty request to signal the end of this stage.
//       padding_queue.write(
//         write_request_t(block_id, 0, (uint64)0, true));
//     }
//   }
// }


void chunk_blocks_2d(hls::stream<fblock_2d_t> &fblock, const zfp_input &input)
{
  size_t nx = input.nx;
  size_t ny = input.ny;
  ptrdiff_t sx = input.sx ? input.sx : 1;
  ptrdiff_t sy = input.sy ? input.sy : (ptrdiff_t)nx;

  partition_blocks_outer: for (size_t y = 0; y < ny; y += 4) {
    partition_blocks_inner: for (size_t x = 0, block_id = 0; x < nx; x += 4, block_id++) {
      const float *raw = input.data + sx * (ptrdiff_t)x + sy * (ptrdiff_t)y;
      size_t bx = MIN(nx - x, 4u);
      size_t by = MIN(ny - y, 4u);
      fblock_2d_t fblock_buf;
      fblock_buf.id = block_id;

      collect_block_outer: for (size_t i = 0; i < by; i++, raw += sy - (ptrdiff_t)bx * sx) {
        collect_block_inner: for (size_t j = 0; j < bx; j++, raw += sx) {
          #pragma HLS PIPELINE II=1
          fblock_buf.data[4 * i + j] = *raw;
        }
      }
      //* Padding partial blocks (each padding operation involves multiple reads/writes -> II ≈ 4 or 5)
      if (bx < 4) {
        //* Fully unrolled, giving each statement sperate HW unit. 
        // #pragma HLS PIPELINE II=1
        pad_x_loop: for (size_t i = 0; i < by; i++) {
          // #pragma HLS UNROLL factor=4
          //* Pad horizontally to 4.
          pad_partial_block(fblock_buf.data + 4 * i, bx, 1);
        }
      }

      if (by < 4) {
        // #pragma HLS PIPELINE II=1
        pad_y_loop: for (size_t j = 0; j < 4; j++) {
          // #pragma HLS UNROLL factor=4
          //* Pad vertically to 4 (stride = 4).
          pad_partial_block(fblock_buf.data + j, by, 4);
        }
      }

      fblock.write(fblock_buf);
    }
  }
}