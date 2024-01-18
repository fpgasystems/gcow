#include "types.hpp"
#include "stream.hpp"
#include <stdio.h>


double set_zfp_output_accuracy(zfp_output &output, double tolerance)
{
  int emin = ZFP_MIN_EXP;
  if (tolerance > 0) {
    /* tolerance = x * 2^emin, with 0.5 <= x < 1 */
    FREXP(tolerance, &emin);
    emin--;
    /* assert: 2^emin <= tolerance < 2^(emin+1) */
  }
  output.minbits = ZFP_MIN_BITS;
  output.maxbits = ZFP_MAX_BITS;
  output.maxprec = ZFP_MAX_PREC;
  output.minexp = emin;
  //* Returns the maximum error tolerance (x=1) for the given precision.
  return tolerance > 0 ? LDEXP(1.0, emin) : 0;
}

uint is_reversible(const zfp_output &output)
{
  return output.minexp < ZFP_MIN_EXP;
}

uint get_input_dimension(const zfp_input &input)
{
  return (input.nx && input.ny)? (input.nz ? (input.nw ? 4 : 3) : 2) : 0;
}

size_t get_input_num_blocks(const zfp_input &input)
{
#pragma HLS INLINE

  size_t bx = (input.nx + 3) / 4;
  size_t by = (input.ny + 3) / 4;
  size_t bz = (input.nz + 3) / 4;
  size_t bw = (input.nw + 3) / 4;
  switch (get_input_dimension(input)) {
    case 2:
      return bx * by;
    case 3:
      return bx * by * bz;
    case 4:
      return bx * by * bz * bw;
    default:
      return 0;
  }
}

size_t get_input_size(const zfp_input &input)
{
  return MAX(input.nx, 1u) * MAX(input.ny, 1u) * MAX(input.nz,
         1u) * MAX(input.nw, 1u);
}

size_t get_dtype_size(data_type dtype)
{
  switch (dtype) {
    case dtype_int32:
      return sizeof(int32);
    case dtype_int64:
      return sizeof(int64);
    case dtype_float:
      return sizeof(float);
    case dtype_double:
      return sizeof(double);
    default:
      return 0;
  }
}

uint get_input_precision(const zfp_input &input)
{
  return (uint)(CHAR_BIT * get_dtype_size(input.dtype));
}

size_t get_max_output_bytes(const zfp_output &output, const zfp_input &input)
{
  int reversible = is_reversible(output);
  printf("Reversible:\t%d\n", reversible);
  uint dim = get_input_dimension(input);
  size_t num_blocks = get_input_num_blocks(input);
  printf("Total 4^d blocks:\t%ld\n", num_blocks);
  uint values = (1u << (2 * dim));
  uint maxbits = 0;

  if (!dim) {
    return 0;
  }

  switch (input.dtype) {
    //TODO: Check if these are the correct maxbits.
    case dtype_int32:
      maxbits += reversible ? 5 : 0;
      break;
    case dtype_int64:
      maxbits += reversible ? 6 : 0;
      break;
    case dtype_float:
      maxbits += reversible ? 1 + 1 + 8 + 5 : 1 + 8;
      break;
    case dtype_double:
      maxbits += reversible ? 1 + 1 + 11 + 6 : 1 + 11;
      break;
    default:
      return 0;
  }
  maxbits += values - 1 + values * MIN(output.maxprec,
                                       get_input_precision(input));
  maxbits = MIN(maxbits, output.maxbits);
  maxbits = MAX(maxbits, output.minbits);
  return ((ZFP_HEADER_MAX_BITS + num_blocks * maxbits + SWORD_BITS - 1) & ~
          (SWORD_BITS - 1)) / CHAR_BIT;
}

uint get_precision(int maxexp, uint maxprec, int minexp, int dim)
{
  #pragma HLS INLINE
  return MIN(maxprec, (uint)MAX(0, maxexp - minexp + 2 * dim + 2));
}

/* True if max compressed size exceeds maxbits */
int exceeded_maxbits(uint maxbits, uint maxprec, uint size)
{
  #pragma HLS INLINE
  //* Compare the total bitplanes to the maximum number of bits per block.
  return (maxprec + 1) * size - 1 > maxbits;
}
