#include "encode.hpp"
#include "stream.hpp"
#include "io.hpp"


void read_block_bitplane(
  size_t total_blocks,
  const uint32 *ublock, hls::stream<ublock_2d_t> &block, uint bemax, uint prec, 
  hls::stream<uint> &out_bemax, hls::stream<uint> &out_maxprec)
{
  for (size_t block_id = 0; block_id < total_blocks; block_id++) {
    ublock_2d_t block_buf;
    block_buf.id = block_id;
    for (int i = 0; i < BLOCK_SIZE_2D; i++) {
      block_buf.data[i] = ublock[i];
    }
    block.write(block_buf);
    out_bemax.write(bemax);
    out_maxprec.write(prec);
  }
}

void write_outputs_bitplane(
  size_t total_blocks,
  stream &s, ptrdiff_t *stream_idx, hls::stream<bit_t> &write_fsm_finished)
{
  await_fsm(write_fsm_finished);
  *stream_idx = s.idx;
}

extern "C" {
  void bitplane(
    const uint32 *ublock, size_t total_blocks, double *max_error, uint *maxprec, bool *exceeded,
    stream_word *out_data, uint *encoded_bits, ptrdiff_t *stream_idx)
  {
#pragma HLS INTERFACE mode=m_axi port=ublock offset=slave bundle=gmem1
#pragma HLS INTERFACE mode=m_axi port=max_error offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=maxprec offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=exceeded offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=out_data offset=slave bundle=gmem2
#pragma HLS INTERFACE mode=m_axi port=encoded_bits offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=stream_idx offset=slave bundle=gmem0

#pragma HLS DATAFLOW
    uint bits = 1 + EBITS;
    size_t max_bytes = 1000; //* Does not matter for this test, just a bound.
    stream s(out_data, max_bytes);

    hls::stream<write_request_t, 2048> write_queue;
    hls::stream<bit_t> write_fsm_finished;
    // drain_write_queue_fsm(total_blocks, s, write_queue, write_fsm_finished);

    zfp_output output;
    double tolerance = 1e-3;
    double error = set_zfp_output_accuracy(output, tolerance);

    int emax = 1;
    uint prec = get_precision(emax, output.maxprec, output.minexp, 2);
    uint biased_emax = prec ? (uint)(emax + EBIAS) : 0;

    hls::stream<uint, 2048> bemax_relay;
    hls::stream<uint, 2048> maxprec_relay;
    hls::stream<ublock_2d_t, 2048> block;
    read_block_bitplane(total_blocks, ublock, block, biased_emax, prec, bemax_relay, maxprec_relay);

    encode_bitplanes_2d(total_blocks, bemax_relay, maxprec_relay, block, output, write_queue);

    drain_write_queue_fsm(total_blocks, s, write_queue, write_fsm_finished);
    
    write_outputs_bitplane(total_blocks, s, stream_idx, write_fsm_finished);
  }
}