#include "encode.hpp"


void cast_block_feeder(
  uint pe_idx,
  size_t num_blocks,
  int e,
  uint be,
  const fblock_2d_t block_buf,
  hls::stream<fblock_2d_t> &fblock,
  hls::stream<int> &emax,
  hls::stream<uint> &bemax) 
{
#pragma HLS INLINE off

  for (size_t block_id = pe_idx; block_id < num_blocks; block_id += FIFO_WIDTH) {
    fblock_2d_t block_local = block_buf;
    block_local.id = block_id;

    fblock.write(block_local);
    emax.write(e);
    bemax.write(be);
  }
}

void feed_cast_block(
  const float *block, 
  hls::stream<fblock_2d_t> fblock[FIFO_WIDTH],
  size_t num_blocks,
  hls::stream<int> emax[FIFO_WIDTH],
  hls::stream<uint> bemax[FIFO_WIDTH])
{
#pragma HLS PIPELINE II=1

  int e = 1;
  uint be = 1 + EBIAS;
  fblock_2d_t block_buf;
  read_cast_block_loop:
  for (uint i = 0; i < BLOCK_SIZE_2D; i++) {
    #pragma HLS UNROLL factor=16

    block_buf.data[i] = block[i];
  }

  feed_cast_block_loop:
  for (uint pe_idx = 0; pe_idx < FIFO_WIDTH; pe_idx++) {
    #pragma HLS UNROLL

    hls::stream<fblock_2d_t> &fblock_s = fblock[pe_idx];
    #pragma HLS DEPENDENCE variable=fblock class=array inter false
    hls::stream<int> &emax_s = emax[pe_idx];
    #pragma HLS DEPENDENCE variable=emax class=array inter false
    hls::stream<uint> &bemax_s = bemax[pe_idx];
    #pragma HLS DEPENDENCE variable=bemax class=array inter false

    cast_block_feeder(pe_idx, num_blocks, e, be, block_buf, fblock_s, emax_s, bemax_s);
  }
}

void block_cast_consumer(
  uint pe_idx,
  size_t num_blocks,
  hls::stream<uint> &bemax, 
  hls::stream<iblock_2d_t> &iblock)
{
#pragma HLS INLINE off

  for (size_t block_id = pe_idx; block_id < num_blocks; block_id += FIFO_WIDTH) {
    uint be = bemax.read();
    iblock_2d_t iblock_buf = iblock.read();
  }
}

void drain_cast_ouput(
  size_t num_blocks,
  hls::stream<iblock_2d_t> iblock[FIFO_WIDTH], 
  hls::stream<uint> bemax_relay[FIFO_WIDTH], 
  int32 *out_iblock)
{
#pragma HLS PIPELINE II=1

  drain_cast_ouput_loop:
  for (uint pe_idx = 0; pe_idx < FIFO_WIDTH; pe_idx++) {
    #pragma HLS UNROLL
    
    hls::stream<iblock_2d_t> &iblock_s = iblock[pe_idx];
    #pragma HLS DEPENDENCE variable=iblock class=array inter false
    hls::stream<uint> &bemax_relay_s = bemax_relay[pe_idx];
    #pragma HLS DEPENDENCE variable=bemax_relay class=array inter false

    block_cast_consumer(pe_idx, num_blocks, bemax_relay_s, iblock_s);

    // //* For verification.
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
    hls::stream<int, 64> emax[FIFO_WIDTH];
    hls::stream<uint, 64> bemax[FIFO_WIDTH];
    hls::stream<fblock_2d_t, 64> fblock[FIFO_WIDTH];
    feed_cast_block(in_fblock, fblock, num_blocks, emax, bemax);

    hls::stream<uint, 64> bemax_relay[FIFO_WIDTH];
    hls::stream<iblock_2d_t, 64> iblock[FIFO_WIDTH];
    // fwd_float2int_2d(num_blocks, emax, bemax, fblock, iblock, bemax_relay);
    fwd_float2int_2d_par(num_blocks, emax, bemax, fblock, iblock, bemax_relay);

    drain_cast_ouput(num_blocks, iblock, bemax_relay, out_iblock);
  }
}