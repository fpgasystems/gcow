#include "encode.hpp"


void read_block_decorrelate(int32 *iblock, hls::stream<iblock_2d_t> &block)
{
  iblock_2d_t iblock_buf;
  for (int i = 0; i < BLOCK_SIZE_2D; i++) {
    iblock_buf.data[i] = iblock[i];
  }
  block.write(iblock_buf);
}

void write_block_decorrelate(hls::stream<iblock_2d_t> &block, int32 *oblock)
{
  iblock_2d_t block_buf = block.read();
  for (int i = 0; i < BLOCK_SIZE_2D; i++) {
    oblock[i] = block_buf.data[i];
  }
}

extern "C" {
  void decorrelate(int32 *iblock)
  {
#pragma HLS INTERFACE mode=m_axi port=iblock offset=slave bundle=gmem0

#pragma HLS DATAFLOW
    hls::stream<iblock_2d_t> block;
    read_block_decorrelate(iblock, block);

    hls::stream<iblock_2d_t> block_relay;
    fwd_decorrelate_2d(block, block_relay);

    write_block_decorrelate(block_relay, iblock);
  }
}