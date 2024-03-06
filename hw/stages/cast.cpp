#include "encode.hpp"


void feed_cast_block(
  const float *block, 
  hls::stream<fblock_2d_t> fblock[FIFO_WIDTH],
  size_t num_blocks,
  hls::stream<int> emax[FIFO_WIDTH],
  hls::stream<uint> bemax[FIFO_WIDTH])
{
  int e = 1;
  uint be = 1 + EBIAS;
  fblock_2d_t block_buf;
  read_cast_block_loop:
  for (uint i = 0; i < BLOCK_SIZE_2D; i++) {
    #pragma HLS UNROLL factor=16

    block_buf.data[i] = block[i];
  }

  feed_cast_block_loop:
  for (uint block_id = 0; block_id < num_blocks; block_id++) {
    #pragma HLS PIPELINE II=1
    #pragma HLS UNROLL factor=64

    uint fifo_idx = FIFO_INDEX(block_id);
    block_buf.id = block_id;
    fblock[fifo_idx].write(block_buf);
    emax[fifo_idx].write(e);
    bemax[fifo_idx].write(be);
  }
}

void drain_cast_ouput(
  size_t num_blocks,
  hls::stream<iblock_2d_t> iblock[FIFO_WIDTH], 
  hls::stream<uint> bemax_relay[FIFO_WIDTH], 
  int32 *out_iblock)
{
  drain_cast_ouput_loop:
  for (uint block_id = 0; block_id < num_blocks; block_id++) {
    #pragma HLS PIPELINE II=1
    #pragma HLS UNROLL factor=64
    
    uint fifo_idx = FIFO_INDEX(block_id);
    iblock_2d_t iblock_buf = iblock[fifo_idx].read();
    uint be = bemax_relay[fifo_idx].read();
    // for (uint i = 0; i < BLOCK_SIZE_2D; i++) {
    //   #pragma HLS UNROLL
    //   out_iblock[block_id * BLOCK_SIZE_2D + i] = iblock_buf.data[i];
    // }
  }
}

extern "C" {
  void cast(const float *in_fblock, size_t num_blocks, int32 *out_iblock)
  {
#pragma HLS INTERFACE mode=m_axi port=in_fblock offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=out_iblock offset=slave bundle=gmem0

#pragma HLS DATAFLOW
    // fwd_blockfloat2int(iblock, fblock, n, emax);
    hls::stream<int, 512> emax[FIFO_WIDTH];
    hls::stream<uint, 512> bemax[FIFO_WIDTH];
    hls::stream<fblock_2d_t, 512> fblock[FIFO_WIDTH];
    feed_cast_block(in_fblock, fblock, num_blocks, emax, bemax);

    hls::stream<uint, 512> bemax_relay[FIFO_WIDTH];
    hls::stream<iblock_2d_t, 512> iblock[FIFO_WIDTH];
    // fwd_float2int_2d(num_blocks, emax, bemax, fblock, iblock, bemax_relay);
    fwd_float2int_2d_par(num_blocks, emax, bemax, fblock, iblock, bemax_relay);

    drain_cast_ouput(num_blocks, iblock, bemax_relay, out_iblock);
  }
}