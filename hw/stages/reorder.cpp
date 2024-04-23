#include "encode.hpp"


void reorder_block_feeder(
  uint pe_idx,
  size_t num_blocks,
  const iblock_2d_t iblock_buf,
  hls::stream<iblock_2d_t> &iblock,
  hls::stream<uint> &bemax) 
{
#pragma HLS INLINE off

  for (size_t block_id = pe_idx; block_id < num_blocks; block_id += FIFO_WIDTH) {
    iblock_2d_t block_local = iblock_buf;
    block_local.id = block_id;

    iblock.write(block_local);
    bemax.write(1 + EBIAS);
  }
}

void feed_block_reorder(
  size_t num_blocks,
  const int32 *iblock, 
  hls::stream<iblock_2d_t> iblock_relay[FIFO_WIDTH],
  hls::stream<uint> bemax_relay[FIFO_WIDTH])
{
#pragma HLS PIPELINE II=1

  iblock_2d_t block_buf;
  read_block_reorder_loop:
  for (int i = 0; i < BLOCK_SIZE_2D; i++) {
    #pragma HLS UNROLL factor=16

    block_buf.data[i] = iblock[i];
  }

  feed_block_reorder_loop:
  for (uint pe_idx = 0; pe_idx < FIFO_WIDTH; pe_idx++) {
    #pragma HLS UNROLL

    hls::stream<iblock_2d_t> &iblock_s = iblock_relay[pe_idx];
    #pragma HLS DEPENDENCE variable=iblock_relay class=array inter false
    hls::stream<uint> &bemax_s = bemax_relay[pe_idx];
    #pragma HLS DEPENDENCE variable=bemax_relay class=array inter false

    reorder_block_feeder(pe_idx, num_blocks, block_buf, iblock_s, bemax_s);
  }
}

void block_reorder_consumer(
  uint pe_idx,
  size_t num_blocks,
  hls::stream<uint> &bemax, 
  hls::stream<ublock_2d_t> &ublock)
{
#pragma HLS INLINE off

  for (size_t block_id = pe_idx; block_id < num_blocks; block_id += FIFO_WIDTH) {
    bemax.read();
    ublock.read();
  }
}

void drain_block_reorder(
  size_t num_blocks,
  hls::stream<ublock_2d_t> ublock_relay[FIFO_WIDTH],
  hls::stream<uint> bemax_relay[FIFO_WIDTH],
  uint32 *ublock)
{
#pragma HLS PIPELINE II=1

  drain_block_reorder_loop:
  for (uint pe_idx = 0; pe_idx < FIFO_WIDTH; pe_idx++) {
    #pragma HLS UNROLL

    hls::stream<ublock_2d_t> &ublock_s = ublock_relay[pe_idx];
    #pragma HLS DEPENDENCE variable=ublock_relay class=array inter false
    hls::stream<uint> &bemax_s = bemax_relay[pe_idx];
    #pragma HLS DEPENDENCE variable=bemax_relay class=array inter false

    block_reorder_consumer(pe_idx, num_blocks, bemax_s, ublock_s);

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
    hls::stream<uint, 64> bemax_relay1[FIFO_WIDTH];
    hls::stream<uint, 64> bemax_relay2[FIFO_WIDTH];
    hls::stream<iblock_2d_t, 64> iblock_relay[FIFO_WIDTH];
    hls::stream<ublock_2d_t, 64> ublock_relay[FIFO_WIDTH];

    feed_block_reorder(num_blocks, iblock, iblock_relay, bemax_relay1);
  
    // fwd_reorder_int2uint_2d(num_blocks, bemax_relay1, iblock_relay, ublock_relay, bemax_relay2);
    fwd_reorder_int2uint_2d_par(num_blocks, bemax_relay1, iblock_relay, ublock_relay, bemax_relay2);

    drain_block_reorder(num_blocks, ublock_relay, bemax_relay2, ublock);
  }
}