#ifndef ENCODE_H
#define ENCODE_H

#include <stddef.h>


void gather_block(double *block, const double *raw, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw);
void encode_block(int *encoded, const double *block);
void encode_block_strided(int *encoded, const double *raw, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw);

#endif // ENCODE_H