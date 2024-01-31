#include "encode.hpp"


void read_blocks(
  const float *block, size_t total_blocks, 
  hls::stream<fblock_2d_t> &fblock, hls::stream<int> &emax, hls::stream<uint> &bemax)
{
  for (size_t i = 0; i < total_blocks; i++) {
    fblock_2d_t block_buf;
    block_buf.id = i;
    for (uint j = 0; j < BLOCK_SIZE_2D; j++) {
      block_buf.data[j] = block[j];
    }
    fblock.write(block_buf);
    emax.write(1);
    bemax.write(1 + EBIAS);
  }
}

void write_blocks(
  hls::stream<ublock_2d_t> &ublock, hls::stream<uint> &bemax, 
  size_t total_blocks, int32 *result)
{
  for (size_t i = 0; i < total_blocks; i++) {
    bemax.read();
    ublock_2d_t ublock_buf = ublock.read();
    for (uint j = 0; j < BLOCK_SIZE_2D; j++) {
      result[i * BLOCK_SIZE_2D + j] = ublock_buf.data[j];
    }
  }
}

extern "C" {
  void integration(const float *in_fblock, const size_t total_blocks, int32 *out_iblock)
  {
#pragma HLS INTERFACE mode=m_axi port=in_fblock offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=out_iblock offset=slave bundle=gmem1

#pragma HLS DATAFLOW

    hls::stream<fblock_2d_t, 32> fblock;
    hls::stream<int, 32> emax;
    hls::stream<uint, 32> bemax;
    read_blocks(in_fblock, total_blocks, fblock, emax, bemax);

    hls::stream<iblock_2d_t, 32> iblock;
    hls::stream<uint, 32> bemax_relay;
    fwd_float2int_2d(total_blocks, emax, bemax, fblock, iblock, bemax_relay);
  
    hls::stream<iblock_2d_t, 32> iblock_relay;
    hls::stream<uint, 32> bemax_relay2;
    fwd_decorrelate_2d(total_blocks, bemax_relay, iblock, iblock_relay, bemax_relay2);

    hls::stream<ublock_2d_t, 32> ublock;
    hls::stream<uint, 32> bemax_relay3;  
    fwd_reorder_int2uint_2d(total_blocks, bemax_relay2, iblock_relay, ublock, bemax_relay3);


    write_blocks(ublock, bemax_relay3, total_blocks, out_iblock);
  }
}