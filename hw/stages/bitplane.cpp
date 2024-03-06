#include "encode.hpp"
#include "stream.hpp"
#include "io.hpp"


void feed_block_bitplane(
  size_t total_blocks,
  const uint32 *ublock, 
  hls::stream<ublock_2d_t> block[FIFO_WIDTH], 
  uint bemax, 
  uint prec, 
  hls::stream<uint> out_bemax[FIFO_WIDTH], 
  hls::stream<uint> out_maxprec[FIFO_WIDTH])
{
  ublock_2d_t block_buf;
  read_block_bitplane_loop:
  for (int i = 0; i < BLOCK_SIZE_2D; i++) {
    #pragma HLS UNROLL factor=16

    block_buf.data[i] = ublock[i];
  }

  feed_block_bitplane_loop:
  for (size_t block_id = 0; block_id < total_blocks; block_id++) {
    #pragma HLS PIPELINE II=1
    // #pragma HLS UNROLL factor=4

    block_buf.id = block_id;
    uint fifo_idx = FIFO_INDEX(block_id);
    block[fifo_idx].write(block_buf);
    out_bemax[fifo_idx].write(bemax);
    out_maxprec[fifo_idx].write(prec);
  }
}

void write_outputs_bitplane(
  size_t total_blocks,
  stream &s, 
  ptrdiff_t *stream_idx, 
  hls::stream<bit_t> &write_fsm_finished)
{
  await_fsm(write_fsm_finished);
  *stream_idx = s.idx;
}

extern "C" {
  void bitplane(
    const uint32 *ublock, size_t total_blocks, stream_word *out_data, ptrdiff_t *stream_idx)
  {
#pragma HLS INTERFACE mode=m_axi port=ublock offset=slave bundle=gmem1
#pragma HLS INTERFACE mode=m_axi port=out_data offset=slave bundle=gmem2
#pragma HLS INTERFACE mode=m_axi port=stream_idx offset=slave bundle=gmem0

#pragma HLS DATAFLOW
    uint bits = 1 + EBITS;
    size_t max_bytes = 20 * (1 << 20); //* 20 MiB
    stream s(out_data, max_bytes);

    hls::stream<write_request_t, 4> write_queue[FIFO_WIDTH];
    hls::stream<bit_t> write_fsm_finished;
    // drain_write_queue_fsm(total_blocks, s, write_queue, write_fsm_finished);

    zfp_output output;
    double tolerance = 1e-3;
    double error = set_zfp_output_accuracy(output, tolerance);

    int emax = 1;
    uint prec = get_precision(emax, output.maxprec, output.minexp, 2);
    uint biased_emax = prec ? (uint)(emax + EBIAS) : 0;

    hls::stream<uint, 4> bemax_relay[FIFO_WIDTH];
    hls::stream<uint, 4> maxprec_relay[FIFO_WIDTH];
    hls::stream<ublock_2d_t, 4> block[FIFO_WIDTH];
    feed_block_bitplane(total_blocks, ublock, block, biased_emax, prec, bemax_relay, maxprec_relay);

    // encode_bitplanes_2d(total_blocks, bemax_relay, maxprec_relay, block, output, write_queue);
    encode_bitplanes_2d_par(total_blocks, bemax_relay, maxprec_relay, block, output, write_queue);

    // drain_write_queue_fsm(total_blocks, s, write_queue, write_fsm_finished);
    drain_write_queue_fsm_par(total_blocks, s, write_queue, write_fsm_finished);
    
    write_outputs_bitplane(total_blocks, s, stream_idx, write_fsm_finished);
  }
}