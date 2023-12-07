#include "encode.hpp"

extern "C" {
  void decorrelate(int32 *iblock)
  {
#pragma HLS INTERFACE mode=m_axi port=iblock offset=slave bundle=gmem0
    fwd_decorrelate_2d_block(iblock);
  }
}