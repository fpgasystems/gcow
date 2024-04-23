#include "encode.hpp"


void decorrelate_block_feeder(
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

void feed_block_decorrelate(
  size_t num_blocks, 
  int32 *in_iblock, 
  hls::stream<iblock_2d_t> iblock[FIFO_WIDTH],
  hls::stream<uint> bemax[FIFO_WIDTH])
{
#pragma HLS PIPELINE II=1

  iblock_2d_t iblock_buf;
  read_block_decorrelate_loop:
  for (int i = 0; i < BLOCK_SIZE_2D; i++) {
    #pragma HLS UNROLL factor=16

    iblock_buf.data[i] = in_iblock[i];
  }

  feed_block_decorrelate_loop:
  for (uint pe_idx = 0; pe_idx < FIFO_WIDTH; pe_idx++) {
    #pragma HLS UNROLL

    hls::stream<iblock_2d_t> &iblock_s = iblock[pe_idx];
    #pragma HLS DEPENDENCE variable=iblock class=array inter false
    hls::stream<uint> &bemax_s = bemax[pe_idx];
    #pragma HLS DEPENDENCE variable=bemax class=array inter false

    decorrelate_block_feeder(pe_idx, num_blocks, iblock_buf, iblock_s, bemax_s);
  }
}

void block_decorrelate_consumer(
  uint pe_idx,
  size_t num_blocks,
  hls::stream<uint> &bemax, 
  hls::stream<iblock_2d_t> &iblock)
{
#pragma HLS INLINE off

  for (size_t block_id = pe_idx; block_id < num_blocks; block_id += FIFO_WIDTH) {
    bemax.read();
    iblock.read();
  }
}

void drain_block_decorrelate(
  size_t num_blocks,
  hls::stream<iblock_2d_t> iblock[FIFO_WIDTH],
  hls::stream<uint> bemax_relay[FIFO_WIDTH],
  int32 *out_iblock)
{
#pragma HLS PIPELINE II=1

  drain_block_decorrelate_loop:
  for (uint pe_idx = 0; pe_idx < FIFO_WIDTH; pe_idx++) {
    #pragma HLS UNROLL

    hls::stream<iblock_2d_t> &iblock_s = iblock[pe_idx];
    #pragma HLS DEPENDENCE variable=iblock class=array inter false
    hls::stream<uint> &bemax_s = bemax_relay[pe_idx];
    #pragma HLS DEPENDENCE variable=bemax_relay class=array inter false

    block_decorrelate_consumer(pe_idx, num_blocks, bemax_s, iblock_s);

    // //! Only write the first block for verification.
    // if (block_id == 0) {
    //   for (uint i = 0; i < BLOCK_SIZE_2D; i++) {
    //     #pragma HLS UNROLL
    //     out_iblock[i] = iblock_buf.data[i];
    //   }
    // }
  }
}

extern "C" {
  void decorrelate(int32 *in_iblock, size_t num_blocks, int32 *out_iblock)
  {
#pragma HLS INTERFACE mode=m_axi port=in_iblock offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=out_iblock offset=slave bundle=gmem1

#pragma HLS DATAFLOW
    hls::stream<iblock_2d_t, 64> iblock[FIFO_WIDTH];
    hls::stream<uint, 64> bemax[FIFO_WIDTH];
    feed_block_decorrelate(num_blocks, in_iblock, iblock, bemax);

    hls::stream<iblock_2d_t, 64> iblock_relay[FIFO_WIDTH];
    hls::stream<uint, 64> bemax_relay[FIFO_WIDTH];
    // fwd_decorrelate_2d(num_blocks, bemax, iblock, iblock_relay, bemax_relay);
    fwd_decorrelate_2d_par(num_blocks, bemax, iblock, iblock_relay, bemax_relay);

    drain_block_decorrelate(num_blocks, iblock_relay, bemax_relay, out_iblock);
  }
}