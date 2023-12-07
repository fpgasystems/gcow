#include "encode.hpp"

extern "C" {
  void emax(const float *block, uint n, int *result)
  {
#pragma HLS INTERFACE mode=m_axi port=block offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=result offset=slave bundle=gmem0
    *result = get_block_exponent(block, n);
  }
}