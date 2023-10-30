#include "types.h"

/**
 * @brief Get the maximum number of bit planes to encode.
 * @param maxexp Maximum block floating-point exponent.
 * @param maxprec Maximum number of bit planes to encode.
 * @param minexp Minimum block floating-point exponent.
 * @param dim Number of dimensions.
 * @return Maximum number of bit planes to encode.
*/
uint get_precision(int maxexp, uint maxprec, int minexp, int dim)
{
  return MIN(maxprec, (uint)MAX(0, maxexp - minexp + 2 * dim + 2));
}

/* True if max compressed size exceeds maxbits */
int exceeded_maxbits(uint maxbits, uint maxprec, uint size)
{
  //* Compare the total bitplanes to the maximum number of bits per block.
  return (maxprec + 1) * size - 1 > maxbits;
}