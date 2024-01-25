#include "encode.hpp"


void read_cast_block(const float *block, hls::stream<fblock_2d_t> &fblock)
{
  fblock_2d_t block_buf;
  for (uint i = 0; i < BLOCK_SIZE_2D; i++) {
    block_buf.data[i] = block[i];
  }
  fblock.write(block_buf);
}

void write_cast_ouput(hls::stream<iblock_2d_t> &iblock, int32 *result)
{
  iblock_2d_t iblock_buf = iblock.read();
  for (uint i = 0; i < BLOCK_SIZE_2D; i++) {
    result[i] = iblock_buf.data[i];
  }
}

extern "C" {
  void cast(const float *in_fblock, uint n, int emax, int32 *out_iblock)
  {
#pragma HLS INTERFACE mode=m_axi port=in_fblock offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=out_iblock offset=slave bundle=gmem0

#pragma HLS DATAFLOW
    // fwd_blockfloat2int(iblock, fblock, n, emax);
    hls::stream<fblock_2d_t> fblock;
    read_cast_block(in_fblock, fblock);

    hls::stream<iblock_2d_t> iblock;
    // fwd_float2int_2d(fblock, emax, iblock);

    write_cast_ouput(iblock, out_iblock);
  }
}