#include "encode.hpp"


void block_feeder(
  uint pe_idx,
  size_t num_blocks,
  hls::stream<fblock_2d_t> &fblk,
  const fblock_2d_t block_buf) 
{
#pragma HLS INLINE off

  fblock_2d_t block_local = block_buf;

  block_feeder_loop:
  for (size_t block_id = pe_idx; block_id < num_blocks; block_id += FIFO_WIDTH) {

    block_local.id = block_id;
    fblk.write(block_local);
  }
}

void read_emax_block(
  const float *block, size_t num_blocks, 
  hls::stream<fblock_2d_t> fblock[FIFO_WIDTH])
{
#pragma HLS PIPELINE II=1

  fblock_2d_t block_buf;
  read_emax_block_loop:
  for (int i = 0; i < BLOCK_SIZE_2D; i++) {
    #pragma HLS UNROLL factor=16

    block_buf.data[i] = block[i];
  }

  feed_emax_block_loop:
  for (uint pe_idx = 0; pe_idx < FIFO_WIDTH; pe_idx++) {
  #pragma HLS UNROLL

    hls::stream<fblock_2d_t> &fblk = fblock[pe_idx];
    #pragma HLD DEPENDENCE variable=fblock class=array inter false

    block_feeder(pe_idx, num_blocks, fblk, block_buf);
    #pragma HLD DEPENDENCE variable=block_buf inter false

  }
}

void block_consumer(
  uint pe_idx,
  size_t num_blocks,
  hls::stream<int> &emax, 
  hls::stream<uint> &bemax, 
  hls::stream<uint> &maxprec, 
  hls::stream<fblock_2d_t> &fblock_relay)
{
#pragma HLS INLINE off

  for (size_t block_id = pe_idx; block_id < num_blocks; block_id += FIFO_WIDTH) {
    emax.read();
    bemax.read();
    maxprec.read();
    fblock_relay.read();
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
#pragma HLS PIPELINE II=1
  
  drain_emax_output_loop:
  for (uint pe_idx = 0; pe_idx < FIFO_WIDTH; pe_idx++) {
    #pragma HLS UNROLL

    hls::stream<int> &emax_in = emax[pe_idx];
    #pragma HLS DEPENDENCE variable=emax class=array inter false
    hls::stream<uint> &bemax_in = bemax[pe_idx];
    #pragma HLS DEPENDENCE variable=bemax class=array inter false
    hls::stream<uint> &maxprec_in = maxprec[pe_idx];
    #pragma HLS DEPENDENCE variable=maxprec class=array inter false
    hls::stream<fblock_2d_t> &fblock_relay_in = fblock_relay[pe_idx];
    #pragma HLS DEPENDENCE variable=fblock_relay class=array inter false

    block_consumer(pe_idx, num_blocks, emax_in, bemax_in, maxprec_in, fblock_relay_in);
  }
}

extern "C" {
  void emax(const float *block, size_t num_blocks, uint *result)
  {
#pragma HLS INTERFACE mode=m_axi port=block offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=result offset=slave bundle=gmem1

#pragma HLS DATAFLOW

  hls::stream<fblock_2d_t, 32> fblock[FIFO_WIDTH];
  #pragma HLS BIND_STORAGE variable=fblock type=fifo impl=LUTRAM
  read_emax_block(block, num_blocks, fblock);

  zfp_output output; 
  double tolerance = 1e-3;
  set_zfp_output_accuracy(output, tolerance);
  hls::stream<int, 32> emax[FIFO_WIDTH];
  hls::stream<uint, 32> bemax[FIFO_WIDTH];
  hls::stream<uint, 32> maxprec[FIFO_WIDTH];
  hls::stream<fblock_2d_t, 32> fblock_relay[FIFO_WIDTH];
  #pragma HLS BIND_STORAGE variable=fblock_relay type=fifo impl=LUTRAM
  // compute_block_exponent_2d(num_blocks, fblock, output, emax, maxprec, fblock_relay);
  // compute_block_exponent_2d(num_blocks, fblock, output, emax, bemax, maxprec, fblock_relay);
  compute_block_emax_2d(num_blocks, fblock, output, emax, bemax, maxprec, fblock_relay);

  drain_emax_ouput(emax, bemax, maxprec, num_blocks, fblock_relay, result);
  }
}