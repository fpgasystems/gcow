#include "encode.hpp"

extern "C" {
  void cast(const float *fblock, uint n, int emax, int32 *iblock)
  {
#pragma HLS INTERFACE mode=m_axi port=fblock offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=iblock offset=slave bundle=gmem0
    fwd_blockfloat2int(iblock, fblock, n, emax);
  }
}