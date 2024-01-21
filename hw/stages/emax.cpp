#include "encode.hpp"


void read_emax_block(const float *block, hls::stream<fblock_2d_t> &fblock)
{
  fblock_2d_t block_buf;
  for (uint i = 0; i < BLOCK_SIZE_2D; i++) {
    block_buf.data[i] = block[i];
  }
  fblock.write(block_buf);
}

void write_emax_ouput(hls::stream<int> &emax, hls::stream<uint> &maxprec, hls::stream<fblock_2d_t> &fblock_relay, int *result)
{
  maxprec.read();
  fblock_relay.read();
  *result = emax.read();
}

extern "C" {
  void emax(const float *block, uint n, int *result)
  {
#pragma HLS INTERFACE mode=m_axi port=block offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=result offset=slave bundle=gmem1

#pragma HLS DATAFLOW

  hls::stream<fblock_2d_t> fblock;
  read_emax_block(block, fblock);

  zfp_output output; 
  double tolerance = 1e-3;
  set_zfp_output_accuracy(output, tolerance);

  hls::stream<int> emax;
  hls::stream<uint> maxprec;
  hls::stream<fblock_2d_t> fblock_relay;
  get_block_exponent(1, fblock, output, emax, maxprec, fblock_relay);

  // maxprec.read();
  // fblock_relay.read();
  // *result = emax.read();
  write_emax_ouput(emax, maxprec, fblock_relay, result);
  }
}