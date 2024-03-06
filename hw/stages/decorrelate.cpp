#include "encode.hpp"


void feed_block_decorrelate(
  size_t num_blocks, 
  int32 *in_iblock, 
  hls::stream<iblock_2d_t> iblock[FIFO_WIDTH],
  hls::stream<uint> bemax[FIFO_WIDTH])
{
  iblock_2d_t iblock_buf;
  read_block_decorrelate_loop:
  for (int i = 0; i < BLOCK_SIZE_2D; i++) {
    #pragma HLS UNROLL factor=16

    iblock_buf.data[i] = in_iblock[i];
  }

  feed_block_decorrelate_loop:
  for (uint block_id = 0; block_id < num_blocks; block_id++) {
    #pragma HLS PIPELINE II=1
    #pragma HLS UNROLL factor=64

    uint fifo_idx = FIFO_INDEX(block_id);
    iblock_buf.id = block_id;
    iblock[fifo_idx].write(iblock_buf);
    bemax[fifo_idx].write(1 + EBIAS);
  }
}

void drain_block_decorrelate(
  size_t num_blocks,
  hls::stream<iblock_2d_t> iblock[FIFO_WIDTH],
  hls::stream<uint> bemax_relay[FIFO_WIDTH],
  int32 *out_iblock)
{
  drain_block_decorrelate_loop:
  for (uint block_id = 0; block_id < num_blocks; block_id++) {
    #pragma HLS PIPELINE II=1
    #pragma HLS UNROLL factor=64

    uint fifo_idx = FIFO_INDEX(block_id);
    uint be = bemax_relay[fifo_idx].read();
    iblock_2d_t iblock_buf = iblock[fifo_idx].read();

    //! Only write the first block for verification.
    if (block_id == 0) {
      for (uint i = 0; i < BLOCK_SIZE_2D; i++) {
        #pragma HLS UNROLL
        out_iblock[i] = iblock_buf.data[i];
      }
    }
  }
}

extern "C" {
  void decorrelate(int32 *in_iblock, size_t num_blocks, int32 *out_iblock)
  {
#pragma HLS INTERFACE mode=m_axi port=in_iblock offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=out_iblock offset=slave bundle=gmem1

#pragma HLS DATAFLOW
    hls::stream<iblock_2d_t, 512> iblock[FIFO_WIDTH];
    hls::stream<uint, 512> bemax[FIFO_WIDTH];
    feed_block_decorrelate(num_blocks, in_iblock, iblock, bemax);

    hls::stream<iblock_2d_t, 512> iblock_relay[FIFO_WIDTH];
    hls::stream<uint, 512> bemax_relay[FIFO_WIDTH];
    // fwd_decorrelate_2d(num_blocks, bemax, iblock, iblock_relay, bemax_relay);
    fwd_decorrelate_2d_par(num_blocks, bemax, iblock, iblock_relay, bemax_relay);

    drain_block_decorrelate(num_blocks, iblock_relay, bemax_relay, out_iblock);
  }
}