#include "encode.hpp"

extern "C" {
  void reorder(const int32 *iblock, uint32 *ublock)
  {
#pragma HLS INTERFACE mode=m_axi port=iblock offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=ublock offset=slave bundle=gmem0
    fwd_reorder_int2uint_block(ublock, iblock, PERM_2D, BLOCK_SIZE_2D);
  }
}