#include "encode.hpp"


void read_block_reorder(
  const int32 *iblock, 
  hls::stream<iblock_2d_t> &iblock_relay,
  hls::stream<uint> &emax_relay)
{
  iblock_2d_t block_buf;
  for (int i = 0; i < BLOCK_SIZE_2D; i++) {
    block_buf.data[i] = iblock[i];
  }
  iblock_relay.write(block_buf);
  emax_relay.write(218);
}

void write_block_reorder(
  hls::stream<ublock_2d_t> &ublock_relay,
  hls::stream<uint> &emax_relay,
  uint32 *ublock)
{
  ublock_2d_t block_buf = ublock_relay.read();
  emax_relay.read();
  for (int i = 0; i < BLOCK_SIZE_2D; i++) {
    ublock[i] = block_buf.data[i];
  }
}

extern "C" {
  void reorder(const int32 *iblock, uint32 *ublock)
  {
#pragma HLS INTERFACE mode=m_axi port=iblock offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=ublock offset=slave bundle=gmem1
    // fwd_reorder_int2uint_block(ublock, iblock, PERM_2D, BLOCK_SIZE_2D);

#pragma HLS DATAFLOW
    hls::stream<uint, 32> emax_relay1;
    hls::stream<uint, 32> emax_relay2;
    hls::stream<iblock_2d_t, 512> iblock_relay;
    hls::stream<ublock_2d_t, 512> ublock_relay;

    read_block_reorder(iblock, iblock_relay, emax_relay1);
  
    fwd_reorder_int2uint_2d(1, emax_relay1, iblock_relay, ublock_relay, emax_relay2);

    write_block_reorder(ublock_relay, emax_relay2, ublock);
  }
}