#include "encode.hpp"


void drain_chunk_output_par(
  size_t num_blocks, 
  hls::stream<fblock_2d_t> fblock[FIFO_WIDTH], 
  fblock_2d_t *out_blocks)
{
  output_loop:
  for (uint block_id = 0; block_id < num_blocks; block_id++) {
    #pragma HLS PIPELINE II=1
    #pragma HLS UNROLL factor=64

    uint fifo_idx = FIFO_INDEX(block_id);
    fblock_2d_t block = fblock[fifo_idx].read();
    #pragma HLS DEPENDENCE variable=fblock class=array inter false
    // out_blocks[block_id] = block;
  }
}

void drain_chunk_output(
  size_t num_blocks, 
  hls::stream<fblock_2d_t> &fblock, 
  fblock_2d_t *out_blocks)
{
  output_loop:
  for (uint block_id = 0; block_id < num_blocks; block_id++) {
    #pragma HLS PIPELINE II=1

    fblock_2d_t block = fblock.read();
    #pragma HLS DEPENDENCE variable=fblock class=array inter false
    // out_blocks[block_id] = block;
  }
}

extern "C" {
  void chunk(const float *in_values, size_t size, fblock_2d_t *out_blocks)
  {
#pragma HLS INTERFACE mode=m_axi port=in_values offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=out_blocks offset=slave bundle=gmem1

#pragma HLS DATAFLOW

  //* Initialize input.
    size_t input_shape[DIM_MAX] = {size, size};
    zfp_input input(dtype_float, input_shape, 2);
    input.data = in_values;
    size_t num_blocks = get_input_num_blocks(input);

    // hls::stream<fblock_2d_t, 512> fblock;
    // chunk_blocks_2d(fblock, input);
    hls::stream<fblock_2d_t, 512> fblock[FIFO_WIDTH];
    chunk_blocks_2d_par(fblock, input);

    drain_chunk_output_par(num_blocks, fblock, out_blocks);
    // drain_chunk_output(num_blocks, fblock, out_blocks);
  }
}