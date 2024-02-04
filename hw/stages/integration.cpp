#include "encode.hpp"
#include "io.hpp"


void read_blocks(
  const float *block, size_t total_blocks, 
  hls::stream<fblock_2d_t> &fblock 
  // zfp_output &output,
  // hls::stream<int> &emax, 
  // hls::stream<uint> &bemax, hls::stream<uint> &maxprec
  )
{
  for (size_t i = 0; i < total_blocks; i++) {
    fblock_2d_t block_buf;
    block_buf.id = i;
    for (uint j = 0; j < BLOCK_SIZE_2D; j++) {
      block_buf.data[j] = block[j];
    }

    // int emax_out = 1;
    // uint prec = get_precision(emax_out, output.maxprec, output.minexp, 2);

    fblock.write(block_buf);
    // emax.write(emax_out);
    // bemax.write(emax_out + EBIAS);
    // maxprec.write(prec);
  }
}

void write_blocks(
  hls::stream<ublock_2d_t> &ublock, hls::stream<uint> &bemax, 
  size_t total_blocks, int32 *result, hls::stream<uint> &maxprec)
{
  for (size_t i = 0; i < total_blocks; i++) {
    bemax.read();
    // maxprec.read();
    ublock_2d_t ublock_buf = ublock.read();
    for (uint j = 0; j < BLOCK_SIZE_2D; j++) {
      result[i * BLOCK_SIZE_2D + j] = ublock_buf.data[j];
    }
  }
}

void write_outputs_integration(
  size_t total_blocks, stream &s, ptrdiff_t *stream_idx, 
  hls::stream<bit_t> &write_fsm_finished)
{
  await_fsm(write_fsm_finished);
  *stream_idx = s.idx;
}

extern "C" {
  void integration(
    const float *in_fblock, const size_t total_blocks, 
    stream_word *out_data, ptrdiff_t *stream_idx, int32 *out_block)
  {
#pragma HLS INTERFACE mode=m_axi port=in_fblock offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=out_data offset=slave bundle=gmem1
#pragma HLS INTERFACE mode=m_axi port=stream_idx offset=slave bundle=gmem2
#pragma HLS INTERFACE mode=m_axi port=out_block offset=slave bundle=gmem3

#pragma HLS DATAFLOW

    //* Initialize input/ouput.
    size_t size = 3;
    size_t input_shape[DIM_MAX] = {size, size};
    zfp_input input(dtype_float, input_shape, 2);
    input.data = in_fblock;
    zfp_output output;
    double tolerance = 1e-3;
    double max_error = set_zfp_output_accuracy(output, tolerance);
    size_t max_output_bytes = get_max_output_bytes(output, input);
    stream output_stream(out_data, max_output_bytes);
    output.data = output_stream;

    hls::stream<fblock_2d_t, 32> fblock;
    // read_blocks(in_fblock, total_blocks, fblock);
    chunk_blocks_2d(fblock, input);

    hls::stream<fblock_2d_t, 32> fblock_relay;
    hls::stream<int, 32> emax;
    hls::stream<uint, 32> bemax;
    hls::stream<uint, 32> maxprec;
    compute_block_exponent_2d(total_blocks, fblock, output, emax, bemax, maxprec, fblock_relay);

    hls::stream<iblock_2d_t, 32> iblock;
    hls::stream<uint, 32> bemax_relay;
    fwd_float2int_2d(total_blocks, emax, bemax, fblock_relay, iblock, bemax_relay);
  
    hls::stream<iblock_2d_t, 32> iblock_relay;
    hls::stream<uint, 32> bemax_relay2;
    fwd_decorrelate_2d(total_blocks, bemax_relay, iblock, iblock_relay, bemax_relay2);

    hls::stream<ublock_2d_t, 32> ublock;
    hls::stream<uint, 32> bemax_relay3;  
    fwd_reorder_int2uint_2d(total_blocks, bemax_relay2, iblock_relay, ublock, bemax_relay3);

    // write_blocks(ublock, bemax_relay3, total_blocks, out_block, maxprec);

    hls::stream<write_request_t, 32> write_queue;
    hls::stream<bit_t> write_fsm_finished;
    drain_write_queue_fsm(total_blocks, output_stream, write_queue, write_fsm_finished);

    encode_bitplanes_2d(total_blocks, bemax_relay3, maxprec, ublock, output, write_queue);

    write_outputs_integration(total_blocks, output_stream, stream_idx, write_fsm_finished);
  }
}