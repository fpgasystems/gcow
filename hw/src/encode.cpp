// Description: Encoding functions for ZFP compression.
// Documentation: ./include/encode.hpp

#include <stddef.h>
#include <stdio.h>
#include <math.h>

#include "encode.hpp"
#include "stream.hpp"
#include "types.hpp"


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
void pad_partial_block(volatile float *block, size_t n, ptrdiff_t s)
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

void gather_2d_block(volatile float *block, volatile const float *raw,
                     ptrdiff_t sx, ptrdiff_t sy)
{
LOOP_GATHER_2D_BLOCK:
  for (size_t y = 0; y < 4; y++, raw += sy - 4 * sx)
LOOP_GATHER_2D_BLOCK_INNER:
    for (size_t x = 0; x < 4; x++, raw += sx) {
      *block++ = *raw;
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

// void gather_4d_block(float *block, const float *raw,
//                      ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)
// {
//   size_t x, y, z, w;
//   for (w = 0; w < 4; w++, raw += sw - 4 * sz)
//     for (z = 0; z < 4; z++, raw += sz - 4 * sy)
//       for (y = 0; y < 4; y++, raw += sy - 4 * sx)
//         for (x = 0; x < 4; x++, raw += sx)
//           *block++ = *raw;
// }

// void gather_partial_4d_block(float *block, const float *raw,
//                              size_t nx, size_t ny, size_t nz, size_t nw,
//                              ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)
// {
//   size_t x, y, z, w;
//   for (w = 0; w < nw; w++, raw += sw - (ptrdiff_t)nz * sz) {
//     for (z = 0; z < nz; z++, raw += sz - (ptrdiff_t)ny * sy) {
//       for (y = 0; y < ny; y++, raw += sy - (ptrdiff_t)nx * sx) {
//         for (x = 0; x < nx; x++, raw += sx) {
//           block[64 * w + 16 * z + 4 * y + x] = *raw;
//         }
//         //* Pad x dimension if the number of elements gathered is less than 4.
//         pad_partial_block(block + 64 * w + 16 * z + 4 * y, nx, 1);
//       }
//       for (x = 0; x < 4; x++)
//         pad_partial_block(block + 64 * w + 16 * z + x, ny, 4);
//     }
//     for (y = 0; y < 4; y++)
//       for (x = 0; x < 4; x++)
//         pad_partial_block(block + 64 * w + 4 * y + x, nz, 16);
//   }
//   for (z = 0; z < 4; z++)
//     for (y = 0; y < 4; y++)
//       for (x = 0; x < 4; x++)
//         pad_partial_block(block + 16 * z + 4 * y + x, nw, 64);
// }

int get_scaler_exponent(float x)
{
  //* In case x==0.
  int e = -EBIAS;
  if (x > 0) {
    //* Get exponent of x.
    FREXP(x, &e);
    //* Clamp exponent in case x is subnormal; may still result in overflow.
    //* E.g., smallest number: 2^(-126) = 1.1754944e-38, which is subnormal.
    e = MAX(e, 1 - EBIAS);
  }
  return e;
}

int get_block_exponent(volatile const float *block, uint n)
{
  float max = 0;
  //* Find the maximum floating point and return its exponent as the block exponent.
  do {
    float f = FABS(*block++);
    if (max < f)
      max = f;
  } while (--n);
  return get_scaler_exponent(max);
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
  //* `((int)(8 * sizeof(float)) - 2) - e` calculates the difference in exponents to
  //* achieve the desired quantization relative to e.
  return LDEXP(x, ((int)(CHAR_BIT * sizeof(float)) - 2) - e);
}

/**
 * @brief Forward block-floating-point transform that
 *  maps a float block to signed integer block with a common exponent emax.
 * @param iblock Pointer to the destination integer block.
 * @param fblock Pointer to the source floating point block.
 * @param n Number of elements in the block.
 * @param emax Exponent of the block.
 * @return void
*/
void fwd_cast_block(volatile int32 *iblock, volatile const float *fblock,
                    uint n, int emax)
{
  //* Compute power-of-two scale factor for all floats in the block
  //* relative to emax of the block.
  float scale = quantize_scaler(1.0f, emax);
  //? Compute p-bit int y = s*x where x is floating and |y| <= 2^(p-2) - 1
  do {
    *iblock++ = (int32)(scale * *fblock++);
  } while (--n);
}

void fwd_lift_vector(volatile int32 *p, ptrdiff_t s)
{
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

void fwd_decorrelate_2d_block(volatile int32 *iblock)
{
  uint x, y;
  /* transform along x */
LOOP_FWD_DECORRELATE_2D_BLOCK_X:
  for (y = 0; y < 4; y++)
    fwd_lift_vector(iblock + 4 * y, 1);
  /* transform along y */
LOOP_FWD_DECORRELATE_2D_BLOCK_Y:
  for (x = 0; x < 4; x++)
    fwd_lift_vector(iblock + 1 * x, 4);
}

/* Map two's complement signed integer to negabinary unsigned integer */
uint32 twoscomplement_to_negabinary(int32 x)
{
  return ((uint32)x + NBMASK) ^ NBMASK;
}

/* Reorder signed coefficients and convert to unsigned integer */
void fwd_reorder_int2uint(volatile uint32 *ublock, volatile const int32 *iblock,
                          const uchar* perm, uint n)
{
  do
    *ublock++ = twoscomplement_to_negabinary(iblock[*perm++]);
  while (--n);
}

/* Compress <= 64 (1-3D) unsigned integers with rate contraint */
uint encode_partial_bitplanes(stream &s, volatile const uint32 *const ublock,
                              uint maxbits, uint maxprec, uint block_size)
{
  /* Make a copy of bit stream to avoid aliasing */
  //! CHANGE: No copy is made here!
  // stream s(out_data);
  uint intprec = (uint)(CHAR_BIT * sizeof(int32));
  //* `kmin` is the cutoff of the least significant bit plane to encode.
  uint kmin = intprec > maxprec ? intprec - maxprec : 0;
  uint bits = maxbits;
  uint i, k, m, n;
  uint64 x;

  //* Encode one bit plane at a time from MSB to LSB
LOOP_ENCODE_PARTIAL_BITPLANES:
  for (k = intprec, n = 0; bits && k-- > kmin;) {
    //^ Step 1: Extract bit plane #k to x
    x = 0;
LOOP_ENCODE_PARTIAL_BITPLANES_TRANSPOSE:
    for (i = 0; i < block_size; i++)
      //* Below puts the `k`th bit of `data[i]` into the `i`th bit of `x`.
      //* I.e., transposing into the bit plane format.
      x += (uint64)((ublock[i] >> k) & 1u) << i;

    //^ Step 2: Encode first n bits of bit plane verbatim.
    //* Bound the total encoded bit `n` by the `maxbits`.
    //& `n` is the number of bits in `x` that have been encoded so far.
    m = MIN(n, bits);
    bits -= m;
    //* (The first n bits "are encoded verbatim.")
    x = stream_write_bits(s, x, m);

    //^ Step 3: Bitplane embedded (unary run-length) encode remainder of bit plane.
    //* Shift `x` right by 1 bit and increment `n` until `x` becomes 0.
LOOP_ENCODE_PARTIAL_BITPLANES_EMBED:
    for (; bits && n < block_size; x >>= 1, n++) {
      //* The number of bits in `x` still to be encoded.
      bits--;
      //* Group test: If `x` is not 0, then write a 1 bit.
      //& `!!` is used to convert a value to its corresponding boolean value, e.g., !!5 == true.
      //& `stream_write_bit()` returns the bit (1/0) that was written.
      if (stream_write_bit(s, !!x)) {
        //^ Positive group test (x != 0) -> Scan for one-bit.
        for (; bits && n < block_size - 1; x >>= 1, n++) {
          //* `n` is incremented for every encoded bit and accumulated across all bitplanes.
          bits--;
          //* Continue writing 0's until a 1 bit is found.
          //& `x & 1u` is used to extract the least significant (right-most) bit of `x`.
          if (stream_write_bit(s, x & 1u))
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

  // out_data = s;
  //* Returns the number of bits written (constrained by `maxbits`).
  return maxbits - bits;
}

/* Compress <= 64 (1-3D) unsigned integers without rate contraint */
uint encode_all_bitplanes(stream &s, volatile const uint32 *const ublock,
                          uint maxprec, uint block_size)
{
  /* make a copy of bit stream to avoid aliasing */
  //! CHANGE: No copy is made here!
  // stream s = *out_data;
  uint64 offset = stream_woffset(s);
  uint intprec = (uint)(CHAR_BIT * sizeof(uint32));
  uint kmin = intprec > maxprec ? intprec - maxprec : 0;
  uint i, k, n;

  /* encode one bit plane at a time from MSB to LSB */
LOOP_ENCODE_ALL_BITPLANES:
  for (k = intprec, n = 0; k-- > kmin;) {
    //^ Step 1: extract bit plane #k to x.
    uint64 x = 0;
LOOP_ENCODE_ALL_BITPLANES_TRANSPOSE:
    for (i = 0; i < block_size; i++)
      x += (uint64)((ublock[i] >> k) & 1u) << i;
    //^ Step 2: encode first n bits of bit plane.
    x = stream_write_bits(s, x, n);
    //^ Step 3: unary run-length encode remainder of bit plane.
LOOP_ENCODE_ALL_BITPLANES_EMBED:
    for (; n < block_size && stream_write_bit(s, !!x); x >>= 1, n++)
      for (; n < block_size - 1 && !stream_write_bit(s, x & 1u); x >>= 1, n++)
        ;
  }

  // *out_data = s;
  //* Returns the number of bits written.
  return (uint)(stream_woffset(s) - offset);
}


uint encode_iblock(stream &out_data, uint minbits, uint maxbits,
                   uint maxprec, volatile int32 *iblock, size_t dim)
{
  size_t block_size = BLOCK_SIZE(dim);
  uint32 ublock[block_size];

  switch (dim) {
    case 2:
      //* Perform forward decorrelation transform.
      fwd_decorrelate_2d_block(iblock);
      //* Reorder signed coefficients and convert to unsigned integer
      fwd_reorder_int2uint(ublock, iblock, PERM_2D, block_size);
      break;
    //TODO: Implement other dimensions.
    default:
      break;
  }

  uint encoded_bits = 0;
  //* Bitplane coding with the fastest implementation.
  if (exceeded_maxbits(maxbits, maxprec, block_size)) {
    if (block_size < BLOCK_SIZE_4D) {
      //* Encode partial bitplanes with rate constraint.
      encoded_bits = encode_partial_bitplanes(out_data, ublock, maxbits, maxprec,
                                              block_size);
    } else {
      //TODO: Implement 4d encoding
    }
  } else {
    if (block_size < BLOCK_SIZE_4D) {
      //* Encode all bitplanes without rate constraint.
      encoded_bits = encode_all_bitplanes(out_data, ublock, maxprec, block_size);
    } else {
      //TODO: Implement 4d encoding
    }
  }

  //* Write at least minbits bits by padding with zeros.
  if (encoded_bits < minbits) {
    stream_pad(out_data, minbits - encoded_bits);
    encoded_bits = minbits;
  }
  return encoded_bits;
}

uint encode_fblock(zfp_output &output, volatile const float *fblock,
                   size_t dim)
{
  uint bits = 1;
  size_t block_size = BLOCK_SIZE(dim);

  /* block floating point transform */
  //* Compute maximum exponent.
  int emax = get_block_exponent(fblock, block_size);
  uint maxprec = get_precision(emax, output.maxprec, output.minexp, dim);
  //* IEEE 754 exponent bias.
  uint e = maxprec ? (uint)(emax + EBIAS) : 0;

  /* encode block only if biased exponent is nonzero */
  //& Initialize block outside to circumvent LLVM stacksave intrinsic.
  int32 iblock[block_size];
  if (e) {
    /* encode common exponent (emax); LSB indicates that exponent is nonzero */
    bits += EBITS;
    stream_write_bits(output.data, 2 * e + 1, bits);
    /* perform forward block-floating-point transform */
    fwd_cast_block(iblock, fblock, block_size, emax);
    /* encode integer block */
    bits += encode_iblock(
              output.data,
              //* Deduct the exponent bits, which are already encoded.
              output.minbits - MIN(bits, output.minbits),
              output.maxbits - bits,
              maxprec,
              iblock,
              dim);
  } else {
    /* write single zero-bit to indicate that all values are zero */
    //* Compress a block of all zeros and add padding if it's fixed-rate.
    stream_write_bit(output.data, 0);
    if (output.minbits > bits) {
      stream_pad(output.data, output.minbits - bits);
      bits = output.minbits;
    }
  }
  //* Return the number of encoded bits.
  return bits;
}
