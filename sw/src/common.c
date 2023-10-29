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