// Description: Decoding functions for ZFP compression.
// Documentation: ./include/decode.h

#include <stddef.h>
#include <stdio.h>
#include <math.h>

#include "decode.h"
#include "stream.h"


float dequantize_integer(int32 x, int e)
{
  return LDEXP((float)x, e - ((int)(CHAR_BIT * sizeof(float)) - 2));
}

void bwd_cast_block(const int32 *iblock, float *fblock, uint n, int emax)
{
  /* Compute power-of-two scale factor s */
  float s = dequantize_integer(1, emax);
  /* Compute p-bit float x = s*y where |y| <= 2^(p-2) - 1 */
  do
    *fblock++ = (float)(s * *iblock++);
  while (--n);
}

void scatter_2d_block(const float *block, float *raw,
                      ptrdiff_t sx, ptrdiff_t sy)
{
  for (size_t y = 0; y < 4; y++, raw += sy - 4 * sx)
    for (size_t x = 0; x < 4; x++, raw += sx) {
      *raw = *block++;
    }
}

void scatter_partial_2d_block(const float *block, float *raw,
                              size_t nx, size_t ny, ptrdiff_t sx, ptrdiff_t sy)
{
  for (size_t y = 0; y < ny; y++, raw += sy - (ptrdiff_t)nx * sx, block += 4 - nx)
    for (size_t x = 0; x < nx; x++, raw += sx, block++)
      *raw = *block;
}

int32 negabinary_to_twoscomplement(uint32 x)
{
  return (int32)((x ^ NBMASK) - NBMASK);
}

/* reorder unsigned coefficients and convert to signed integer */
void bwd_reorder_uint2int(const uint32 *ublock, int32* iblock,
                          const uchar* perm, uint n)
{
  do
    iblock[*perm++] = negabinary_to_twoscomplement(*ublock++);
  while (--n);
}

void bwd_lift_vector(int32 *p, ptrdiff_t s)
{
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
  **       ( 4  6 -4 -1) (x)
  ** 1/4 * ( 4  2  4  5) (y)
  **       ( 4 -2  4 -5) (z)
  **       ( 4 -6 -4  1) (w)
  */
  y += w >> 1;
  w -= y >> 1;
  y += w;
  w <<= 1;
  w -= y;
  z += x;
  x <<= 1;
  x -= z;
  y += z;
  z <<= 1;
  z -= y;
  w += x;
  x <<= 1;
  x -= w;

  p -= s;
  *p = w;
  p -= s;
  *p = z;
  p -= s;
  *p = y;
  p -= s;
  *p = x;
}

void bwd_decorrelate_2d_block(int32 *iblock)
{
  uint x, y;
  /* first transform along y */
  for (x = 0; x < 4; x++)
    bwd_lift_vector(iblock + 1 * x, 4);
  /* transform along x */
  for (y = 0; y < 4; y++)
    bwd_lift_vector(iblock + 4 * y, 1);
}

uint decode_full_bitplanes(stream *s, uint32 *const ublock,
                           uint maxprec, uint block_size)
{
  size_t offset = stream_roffset(s);
  uint intprec = (uint)(CHAR_BIT * sizeof(uint32));
  uint kmin = intprec > maxprec ? intprec - maxprec : 0;
  uint i, k, n;

  /* initialize data array to all zeros */
  for (i = 0; i < block_size; i++)
    ublock[i] = 0;

  /* decode one bit plane at a time from MSB to LSB */
  for (k = intprec, n = 0; k-- > kmin;) {
    /* step 1: decode first n bits of bit plane #k */
    uint64 x = stream_read_bits(s, n);
    /* step 2: unary run-length decode remainder of bit plane */
    for (; n < block_size && stream_read_bit(s); x += (uint64)1 << n, n++)
      for (; n < block_size - 1 && !stream_read_bit(s); n++)
        ;
    /* step 3: deposit bit plane from x */
    for (i = 0; x; i++, x >>= 1)
      ublock[i] += (int32)(x & 1u) << k;
  }

  return (uint)(stream_roffset(s) - offset);
}

uint decode_partial_bitplanes(stream *const s, uint32 *const ublock,
                              uint maxbits, uint maxprec, uint block_size)
{
  uint intprec = (uint)(CHAR_BIT * sizeof(uint32));
  uint kmin = intprec > maxprec ? intprec - maxprec : 0;
  uint bits = maxbits;
  uint i, k, m, n;
  uint64 x;

  /* initialize data array to all zeros */
  for (i = 0; i < block_size; i++)
    ublock[i] = 0;

  /* decode one bit plane at a time from MSB to LSB */
  for (k = intprec, m = n = 0; bits && (m = 0, k-- > kmin);) {
    /* step 1: decode first n bits of bit plane #k */
    m = MIN(n, bits);
    bits -= m;
    x = stream_read_bits(s, m);
    /* step 2: unary run-length decode remainder of bit plane */
    for (; bits && n < block_size; n++, m = n) {
      bits--;
      if (stream_read_bit(s)) {
        /* positive group test; scan for next one-bit */
        for (; bits && n < block_size - 1; n++) {
          bits--;
          if (stream_read_bit(s))
            break;
        }
        /* set bit and continue decoding bit plane */
        x += (uint64)1 << n;
      } else {
        /* negative group test; done with bit plane */
        m = block_size;
        break;
      }
    }
    /* step 3: deposit bit plane from x */
    for (i = 0; x; i++, x >>= 1)
      ublock[i] += (uint32)(x & 1u) << k;
  }
  return maxbits - bits;
}

uint decode_iblock(stream *const out_data, uint minbits, uint maxbits,
                   uint maxprec, int32 *iblock, size_t dim)
{
  size_t block_size = BLOCK_SIZE(dim);
  uint32 ublock[block_size];
  uint decoded_bits;

  /* decode integer mantissa block */
  if (exceeded_maxbits(maxbits, maxprec, dim)) {
    if (block_size < BLOCK_SIZE_4D) {
      decoded_bits = decode_partial_bitplanes(out_data, ublock, maxbits, maxprec,
                                              dim);
    } else {
      //TODO: Implement 4d decoding.
    }
  } else {
    if (block_size < BLOCK_SIZE_4D) {
      decoded_bits = decode_full_bitplanes(out_data, ublock, maxprec, dim);
    } else {
      //TODO: Implement 4d decoding.
    }
  }

  /* read at least minbits bits */
  if (decoded_bits < minbits) {
    stream_skip(out_data, minbits - decoded_bits);
    decoded_bits = minbits;
  }
  /* reorder unsigned coefficients and convert to signed integer */
  bwd_reorder_uint2int(ublock, iblock, PERM_2D, block_size);
  /* perform decorrelating transform */
  bwd_decorrelate_2d_block(iblock);
  return decoded_bits;
}

uint decode_fblock(zfp_output* output, float* fblock, size_t dim)
{
  uint bits = 1;
  size_t block_size = BLOCK_SIZE(dim);
  int32 iblock[block_size];
  /* test if block has nonzero values */
  if (stream_read_bit(output->data)) {
    uint maxprec;
    int emax;
    /* decode common exponent */
    bits += EBITS;
    emax = (int)stream_read_bits(output->data, EBITS) - EBIAS;
    maxprec = get_precision(emax, output->maxprec, output->minexp, dim);
    /* decode integer block */
    bits += decode_iblock(
              output->data,
              output->minbits - MIN(bits, output->minbits),
              output->maxbits - bits,
              maxprec,
              iblock,
              dim);
    /* perform inverse block-floating-point transform */
    bwd_cast_block(iblock, fblock, block_size, emax);
  } else {
    /* set all values to zero */
    uint i;
    for (i = 0; i < block_size; i++)
      *fblock++ = 0;
    if (output->minbits > bits) {
      stream_skip(output->data, output->minbits - bits);
      bits = output->minbits;
    }
  }
  return bits;
}