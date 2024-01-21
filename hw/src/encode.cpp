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

void get_block_exponent(
  size_t in_total_blocks,
  hls::stream<fblock_2d_t> &in_fblock, 
  const zfp_output &output, 
  hls::stream<int> &out_emax,
  hls::stream<uint> &out_maxprec,
  hls::stream<fblock_2d_t> &out_fblock) 
{
  //! Can't modify the input signal directly. First, read it into a buffer.
  size_t total_blocks = in_total_blocks;
  emax_block_loop: for (; total_blocks--; ) {
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
      //* Get exponent of x.
      FREXP(emax, &e);
      //* Clamp exponent in case x is subnormal; may still result in overflow.
      //* E.g., smallest number: 2^(-126) = 1.1754944e-38, which is subnormal.
      e = MAX(e, 1 - EBIAS);
    }
    uint maxprec = get_precision(e, output.maxprec, output.minexp, 2);
    e = maxprec ? (uint)(e + EBIAS) : 0;
    
    out_maxprec.write(maxprec);
    out_emax.write(e);
  }
}

void fwd_float2int_2d(
  hls::stream<fblock_2d_t> &in_fblock,
  int emax,
  hls::stream<iblock_2d_t> &out_iblock)
{
  fblock_2d_t fblock_buf = in_fblock.read();
  iblock_2d_t iblock_buf;
  iblock_buf.id = fblock_buf.id;

  // uint64 tmp;
  // stream s;
  // //! Writing to output.data will give return-value error.
  // stream_write_bits(s, (2 * emax + 1), 2, &tmp);
  //* Compute power-of-two scale factor for all floats in the block
    //* relative to emax of the block.
    float scale = quantize_scaler(1.0f, emax);
    fwd_f2i_loop: for (uint i = 0; i < BLOCK_SIZE_2D; i++) {
      #pragma HLS pipeline II=2

      //* Compute p-bit int y = s*x where x is floating and |y| <= 2^(p-2) - 1
      iblock_buf.data[i] = (int32)(scale * fblock_buf.data[i]);
    }
  out_iblock.write(iblock_buf);
}

void fwd_blockfloats2ints(
  size_t in_total_blocks,
  hls::stream<fblock_2d_t> &in_fblock,
  hls::stream<int> &in_emax,
  uint minbits,
  hls::stream<write_request_t> &write_queue,
  hls::stream<int> &out_emax,
  hls::stream<uint> &out_bits,
  hls::stream<iblock_2d_t> &out_iblock)
{
  fwd_blockf2i_block_loop: 
  for (size_t block_id = 0; block_id < in_total_blocks; block_id++) {
    //* Blocking reads.
    fblock_2d_t fblock_buf = in_fblock.read();
    iblock_2d_t iblock_buf;
    iblock_buf.id = fblock_buf.id;
    int emax = in_emax.read();

    uint bits = 1;
    write_request_t wrequest;
    size_t block_size = BLOCK_SIZE_2D;

    //* Encode block only if biased exponent is nonzero
    if (emax) {
      //* Encode emax
      bits += EBITS;
      wrequest = {block_id, bits, (uint64)(2 * emax + 1), false};

      //* Compute power-of-two scale factor for all floats in the block
      //* relative to emax of the block.
      float scale = quantize_scaler(1.0f, emax);
      fwd_blockf2i_loop: for (uint i = 0; i < block_size; i++) {
        #pragma HLS pipeline II=2

        //* Compute p-bit int y = s*x where x is floating and |y| <= 2^(p-2) - 1
        iblock_buf.data[i] = (int32)(scale * fblock_buf.data[i]);
      }
      bits = MIN(bits, minbits);
    } else {
      //* Encode single zero-bit to indicate that all values are zero
      wrequest = {block_id, 1, 0, false};
      // if (minbits > bits) {
      //   bits = minbits;
      //   //TODO: Padding.
      //   stream_pad(output->data, output->minbits - bits);
      //   bits = output->minbits;
      // }
    }
    write_queue.write(wrequest);
    out_iblock.write(iblock_buf);
    out_emax.write(emax);
    out_bits.write(bits);
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

void fwd_cast(hls::stream<int32> &out_integer, hls::stream<float> &in_float, uint dim, int emax)
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
  hls::stream<iblock_2d_t> &in_iblock,
  hls::stream<iblock_2d_t> &out_iblock)
{
  iblock_2d_t iblock_buf = in_iblock.read();
  fwd_decorrelate_2d_block(iblock_buf.data);
  out_iblock.write(iblock_buf);
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
  hls::stream<iblock_2d_t> &in_iblock,
  hls::stream<ublock_2d_t> &out_ublock)
{
  iblock_2d_t iblock_buf = in_iblock.read();
  ublock_2d_t ublock_buf;
  ublock_buf.id = iblock_buf.id;

  fwd_reorder_int2uint_block(ublock_buf.data, iblock_buf.data, PERM_2D, BLOCK_SIZE_2D);
  out_ublock.write(ublock_buf);
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
                              hls::stream<write_request_t> &write_queue, size_t block_id,
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
    write_queue.write( write_request_t(block_id, m, x, false) );
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
      write_queue.write( write_request_t(block_id, 1, bit, false) );
      // stream_write_bit(s, bit, &bit);
      if (bit) {
        //^ Positive group test (x != 0) -> Scan for one-bit.
        partial_bitplanes_embed_inner_loop: for (; bits && n < block_size - 1; x >>= 1, n++) {
          //* `n` is incremented for every encoded bit and accumulated across all bitplanes.
          bits--;
          //* Continue writing 0's until a 1 bit is found.
          //& `x & 1u` is used to extract the least significant (right-most) bit of `x`.
          bit = x & 1u;
          write_queue.write( write_request_t(block_id, 1, bit, false) );
          // stream_write_bit(s, bit, &bit);
          if (bit)
            //* After writing a 1 bit, break out for another group test
            //* (to see whether the bitplane code `x` turns 0 after encoding `n` of its bits).
            break;
        }
      } else {
        //^ Negative group test (x == 0) -> Done with bit plane.
        //TODO: Determine the last bit here (w/o wasting a write request)
        break;
      }
    }
  }
  // //* Write an empty request to indicate the end of the block.
  // write_queue.write( write_request_t(block_id, 0, 0, true /* last bit */) );

  //* Returns the number of bits written (constrained by `maxbits`).
  *encoded_bits = maxbits - bits;
}

/* Compress <= 64 (1-3D) unsigned integers without rate contraint */
void encode_all_bitplanes(volatile const uint32 *const ublock,
                          hls::stream<write_request_t> &write_queue, size_t block_id,
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
    write_queue.write( write_request_t(block_id, n, x, false) );
    x >>= n;
    // uint64 new_x;
    // stream_write_bits(s, x, n, &new_x);
    // x = stream_word(new_x);

    //^ Step 3: unary run-length encode remainder of bit plane.
    all_bitplanes_embed_loop: for (; n < block_size; x >>= 1, n++) {
      bit = !!x;
      write_queue.write( write_request_t(block_id, 1, bit, false) );
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
        write_queue.write( write_request_t(block_id, 1, bit, false) );
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
  // //* Write an empty request to indicate the end of the block.
  // write_queue.write( write_request_t(block_id, 0, 0, true /* last bit */) );
  
  //* Returns the number of bits written.
  *encoded_bits = bits;
  // *encoded_bits = (uint)(stream_woffset(s) - offset);
}

void encode_bitplanes_2d(
  hls::stream<ublock_2d_t> &in_ublock,
  uint minbits,
  uint maxbits,
  uint maxprec,
  hls::stream<write_request_t> &write_queue,
  hls::stream<uint> &out_bits)
{
  // //! Temporary only for the stage test.
  // write_queue.write(write_request_t(0, 9, (uint64)(2 * 1 + 1), false));
  // //! ------------------------------

  ublock_2d_t ublock_buf = in_ublock.read();
  uint encoded_bits = 0;
  if (exceeded_maxbits(maxbits, maxprec, BLOCK_SIZE_2D)) {
    //* Encode partial bitplanes with rate constraint.
    encode_partial_bitplanes(
      ublock_buf.data, write_queue, ublock_buf.id, maxbits, maxprec, BLOCK_SIZE_2D, &encoded_bits);
  } else {
    //* Encode all bitplanes without rate constraint.
    encode_all_bitplanes(
      ublock_buf.data, write_queue, ublock_buf.id, maxprec, BLOCK_SIZE_2D, &encoded_bits);
  }

  // //! Temporary only for the stage test.
  // write_queue.write(write_request_t(0, 0, 0, true /* last bit */));
  // //! ------------------------------

  out_bits.write(encoded_bits);
}


// uint encode_iblock(stream &out_data, uint minbits, uint maxbits,
//                    uint maxprec, volatile int32 *iblock, size_t dim)
// {
// #pragma HLS STREAM variable=iblock depth=8 type=fifo
  
//   size_t block_size = BLOCK_SIZE(dim);
//   uint32 ublock[block_size];

//   switch (dim) {
//     case 2:
//       //* Perform forward decorrelation transform.
//       fwd_decorrelate_2d_block(iblock);
//       //* Reorder signed coefficients and convert to unsigned integer
//       fwd_reorder_int2uint(ublock, iblock, PERM_2D, block_size);
//       break;
//     //TODO: Implement other dimensions.
//     default:
//       break;
//   }

//   uint encoded_bits = 0;
//   //* Bitplane coding with the fastest implementation.
//   if (exceeded_maxbits(maxbits, maxprec, block_size)) {
//     if (block_size < BLOCK_SIZE_4D) {
//       //* Encode partial bitplanes with rate constraint.
//       encoded_bits = encode_partial_bitplanes(out_data, ublock, maxbits, maxprec,
//                                               block_size);
//     } else {
//       //TODO: Implement 4d encoding
//     }
//   } else {
//     if (block_size < BLOCK_SIZE_4D) {
//       //* Encode all bitplanes without rate constraint.
//       encoded_bits = encode_all_bitplanes(out_data, ublock, maxprec, block_size);
//     } else {
//       //TODO: Implement 4d encoding
//     }
//   }

//   //* Write at least minbits bits by padding with zeros.
//   if (encoded_bits < minbits) {
//     stream_pad(out_data, minbits - encoded_bits);
//     encoded_bits = minbits;
//   }
//   return encoded_bits;
// }

// uint encode_fblock(zfp_output &output, hls::stream<float> fblock[BLOCK_SIZE_2D],
//                    size_t dim)
// {
//   uint bits = 1;
//   size_t block_size = BLOCK_SIZE(dim);
//   float block[block_size];

//   //* Buffer the stream of floats locally in this module.
//   for (size_t i = 0; i < block_size; i++) {
//     //* Blocking reads.
//     block[i] = fblock[i].read();
//   }

//   /* block floating point transform */
//   //* Compute maximum exponent.
//   int emax = get_block_exponent(block, block_size);
//   uint maxprec = get_precision(emax, output.maxprec, output.minexp, dim);
//   //* IEEE 754 exponent bias.
//   uint e = maxprec ? (uint)(emax + EBIAS) : 0;

//   /* encode block only if biased exponent is nonzero */
//   //& Initialize block outside to circumvent LLVM stacksave intrinsic.
//   int32 iblock[block_size];
//   if (e) {
//     /* encode common exponent (emax); LSB indicates that exponent is nonzero */
//     bits += EBITS;
//     //TODO: Separate the writing process.
//     stream_write_bits(output.data, 2 * e + 1, bits);
//     /* perform forward block-floating-point transform */
//     fwd_cast_block(iblock, block, block_size, emax);
//     /* encode integer block */
//     bits += encode_iblock(
//               output.data,
//               //* Deduct the exponent bits, which are already encoded.
//               output.minbits - MIN(bits, output.minbits),
//               output.maxbits - bits,
//               maxprec,
//               iblock,
//               dim);
//   } else {
//     /* write single zero-bit to indicate that all values are zero */
//     //* Compress a block of all zeros and add padding if it's fixed-rate.
//     stream_write_bit(output.data, 0);
//     if (output.minbits > bits) {
//       stream_pad(output.data, output.minbits - bits);
//       bits = output.minbits;
//     }
//   }
//   // //* Return the number of encoded bits.
//   return bits;
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

      fblock << fblock_buf;
    }
  }
}