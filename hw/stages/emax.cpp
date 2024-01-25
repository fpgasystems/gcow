#include "encode.hpp"

uint num_blocks = 3;

void read_emax_block(const float *block, hls::stream<fblock_2d_t> &fblock)
{
  for (uint block_id = 0; block_id < num_blocks; block_id++) {
    fblock_2d_t block_buf;
    block_buf.id = block_id;
    for (int i = 0; i < BLOCK_SIZE_2D; i++) {
      block_buf.data[i] = block[i];
    }
    fblock.write(block_buf);
  }
}

void write_emax_ouput(
  hls::stream<uint> &emax, 
  hls::stream<uint> &maxprec, 
  hls::stream<fblock_2d_t> &fblock_relay, 
  uint *result)
{
  for (uint block_id = 0; block_id < num_blocks; block_id++) {
    result[block_id] = emax.read();
    maxprec.read();
    fblock_relay.read();
  }
}

extern "C" {
  void emax(const float *block, uint n, uint *result)
  {
#pragma HLS INTERFACE mode=m_axi port=block offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=result offset=slave bundle=gmem1

#pragma HLS DATAFLOW

  hls::stream<fblock_2d_t> fblock;
  read_emax_block(block, fblock);

  zfp_output output; 
  double tolerance = 1e-3;
  set_zfp_output_accuracy(output, tolerance);
  hls::stream<uint> emax;
  hls::stream<uint> maxprec;
  hls::stream<fblock_2d_t> fblock_relay;
  compute_block_exponent_2d(num_blocks, fblock, output, emax, maxprec, fblock_relay);

  // maxprec.read();
  // fblock_relay.read();
  // *result = emax.read();
  write_emax_ouput(emax, maxprec, fblock_relay, result);
  }
}