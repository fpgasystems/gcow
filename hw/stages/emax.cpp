#include "encode.hpp"


void read_emax_block(
  const float *block, size_t num_blocks, 
  hls::stream<fblock_2d_t> fblock[FIFO_WIDTH])
{
  fblock_2d_t block_buf;
  read_emax_block_loop:
  for (int i = 0; i < BLOCK_SIZE_2D; i++) {
    #pragma HLS UNROLL factor=16

    block_buf.data[i] = block[i];
  }
  
  feed_emax_block_loop:
  for (size_t block_id = 0; block_id < num_blocks; block_id++) {
    #pragma HLS PIPELINE II=1
    #pragma HLS UNROLL factor=64

    uint fifo_idx = FIFO_INDEX(block_id);
    block_buf.id = block_id;
    fblock[fifo_idx].write(block_buf);
  }
}

void drain_emax_ouput(
  hls::stream<int> emax[FIFO_WIDTH], 
  hls::stream<uint> bemax[FIFO_WIDTH], 
  hls::stream<uint> maxprec[FIFO_WIDTH], 
  size_t num_blocks,
  hls::stream<fblock_2d_t> fblock_relay[FIFO_WIDTH], 
  uint *result)
{
  drain_emax_ouput_loop:
  for (size_t block_id = 0; block_id < num_blocks; block_id++) {
    #pragma HLS PIPELINE II=1
    
    // result[block_id] = emax.read();
    uint fifo_idx = FIFO_INDEX(block_id);
    emax[fifo_idx].read();
    bemax[fifo_idx].read();
    maxprec[fifo_idx].read();
    fblock_relay[fifo_idx].read();
  }
}

extern "C" {
  void emax(const float *block, size_t num_blocks, uint *result)
  {
#pragma HLS INTERFACE mode=m_axi port=block offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=result offset=slave bundle=gmem1

#pragma HLS DATAFLOW

  hls::stream<fblock_2d_t, 32> fblock[FIFO_WIDTH];
  read_emax_block(block, num_blocks, fblock);

  zfp_output output; 
  double tolerance = 1e-3;
  set_zfp_output_accuracy(output, tolerance);
  hls::stream<int, 32> emax[FIFO_WIDTH];
  hls::stream<uint, 32> bemax[FIFO_WIDTH];
  hls::stream<uint, 32> maxprec[FIFO_WIDTH];
  hls::stream<fblock_2d_t, 32> fblock_relay[FIFO_WIDTH];
  // compute_block_exponent_2d(num_blocks, fblock, output, emax, maxprec, fblock_relay);
  // compute_block_exponent_2d(num_blocks, fblock, output, emax, bemax, maxprec, fblock_relay);
  compute_block_emax_2d(num_blocks, fblock, output, emax, bemax, maxprec, fblock_relay);

  drain_emax_ouput(emax, bemax, maxprec, num_blocks, fblock_relay, result);
  }
}