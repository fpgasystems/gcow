#include "encode.hpp"


void feed_block_reorder(
  size_t num_blocks,
  const int32 *iblock, 
  hls::stream<iblock_2d_t> iblock_relay[FIFO_WIDTH],
  hls::stream<uint> bemax_relay[FIFO_WIDTH])
{
  iblock_2d_t block_buf;
  read_block_reorder_loop:
  for (int i = 0; i < BLOCK_SIZE_2D; i++) {
    #pragma HLS UNROLL factor=16

    block_buf.data[i] = iblock[i];
  }

  feed_block_reorder_loop:
  for (uint block_id = 0; block_id < num_blocks; block_id++) {
    #pragma HLS PIPELINE II=1
    #pragma HLS UNROLL factor=64

    uint fifo_idx = FIFO_INDEX(block_id);
    block_buf.id = block_id;
    iblock_relay[fifo_idx].write(block_buf);
    bemax_relay[fifo_idx].write(1+EBIAS);
  }
}

void drain_block_reorder(
  size_t num_blocks,
  hls::stream<ublock_2d_t> ublock_relay[FIFO_WIDTH],
  hls::stream<uint> bemax_relay[FIFO_WIDTH],
  uint32 *ublock)
{
  drain_block_reorder_loop:
  for (uint block_id = 0; block_id < num_blocks; block_id++) {
    #pragma HLS PIPELINE II=1
    #pragma HLS UNROLL factor=64

    uint fifo_idx = FIFO_INDEX(block_id);
    ublock_2d_t ublock_buf = ublock_relay[fifo_idx].read();
    uint be = bemax_relay[fifo_idx].read();
    // for (uint i = 0; i < BLOCK_SIZE_2D; i++) {
    //   #pragma HLS UNROLL
    //   ublock[block_id * BLOCK_SIZE_2D + i] = ublock_buf.data[i];
    // }
  }
}

extern "C" {
  void reorder(const int32 *iblock, const size_t num_blocks, uint32 *ublock)
  {
#pragma HLS INTERFACE mode=m_axi port=iblock offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=ublock offset=slave bundle=gmem1

#pragma HLS DATAFLOW
    hls::stream<uint, 512> bemax_relay1[FIFO_WIDTH];
    hls::stream<uint, 512> bemax_relay2[FIFO_WIDTH];
    hls::stream<iblock_2d_t, 512> iblock_relay[FIFO_WIDTH];
    hls::stream<ublock_2d_t, 512> ublock_relay[FIFO_WIDTH];

    feed_block_reorder(num_blocks, iblock, iblock_relay, bemax_relay1);
  
    // fwd_reorder_int2uint_2d(num_blocks, bemax_relay1, iblock_relay, ublock_relay, bemax_relay2);
    fwd_reorder_int2uint_2d_par(num_blocks, bemax_relay1, iblock_relay, ublock_relay, bemax_relay2);

    drain_block_reorder(num_blocks, ublock_relay, bemax_relay2, ublock);
  }
}