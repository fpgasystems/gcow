#include "encode.hpp"
#include <cassert>


void chunk_consumer(
  uint pe_idx,
  size_t size,
  size_t num_blocks,
  hls::stream<fblock_2d_t> &fblock)
{
  #pragma HLS INLINE off

  for (size_t y = 0; y < size; y += 4) {
      for (size_t x = 4*pe_idx; x < size; x += 4*(FIFO_WIDTH)) {
        std::cout << "PE " << pe_idx << ", x: " << x << " y: " << y << std::endl;
        // size_t block_id = x/4 + y/4 * (size/4);
        fblock_2d_t fblk = fblock.read();
        // assert(fblk.id == block_id);
        std::cout << "\t consums block " << fblk.id << std::endl;
      }
  }
}

void drain_chunk_output_par(
  size_t size,
  size_t num_blocks, 
  hls::stream<fblock_2d_t> fblock[FIFO_WIDTH], 
  fblock_2d_t *out_blocks)
{
  output_loop:
  for (uint pe_idx = 0; pe_idx < FIFO_WIDTH; pe_idx++) {
    #pragma HLS UNROLL
    
    hls::stream<fblock_2d_t> &fblk = fblock[pe_idx];
    #pragma HLS DEPENDENCE variable=fblock class=array inter false

    chunk_consumer(pe_idx, size, num_blocks, fblk);
  }
}

void drain_chunk_output_par(
  size_t num_blocks, 
  hls::stream<fblock_2d_t> fblock[FIFO_WIDTH], 
  fblock_2d_t *out_blocks)
{
  output_loop:
  for (uint block_id = 0; block_id < num_blocks; block_id++) {
    #pragma HLS PIPELINE II=1

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
    std::cout << "total blocks: " << num_blocks << "size: " << size << std::endl;

    // hls::stream<fblock_2d_t, 512> fblock;
    // chunk_blocks_2d(fblock, input);
    hls::stream<fblock_2d_t, 512> fblock[FIFO_WIDTH];
    // chunk_blocks_2d_par(num_blocks, fblock, input);
    // drain_chunk_output_par(size, num_blocks, fblock, out_blocks);
    
    chunk_blocks_2d_seq(fblock, input);
    drain_chunk_output_par(num_blocks, fblock, out_blocks);

    // drain_chunk_output(num_blocks, fblock, out_blocks);
  }
}