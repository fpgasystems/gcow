#include "encode.hpp"
#include "stream.hpp"
#include "io.hpp"


void read_block_bitplane(const uint32 *ublock, hls::stream<ublock_2d_t> &block)
{
  ublock_2d_t block_buf;
  block_buf.id = 0;
  for (int i = 0; i < BLOCK_SIZE_2D; i++) {
    block_buf.data[i] = ublock[i];
  }
  block.write(block_buf);
}

void prepare_input_bitplane(hls::stream<double> &error, hls::stream<uint> &prec );

void enqueue_emax(int emax, uint bits, hls::stream<write_request_t> &write_queue)
{
  write_queue.write(write_request_t(0, bits, (uint64)(2 * emax + 1), false));
}

void read_outputs_bitplane(
  stream &s, ptrdiff_t *stream_idx,
  hls::stream<uint> &encoded_bits, hls::stream<ap_uint<1>> &write_fsm_finished, uint *out_encoded_bits)
{
  write_fsm_finished.read();
  // encoded_bits.read();
  *out_encoded_bits = encoded_bits.read();
  *stream_idx = s.idx;
}

extern "C" {
  void bitplane(
    const uint32 *ublock, double *max_error, uint *maxprec, bool *exceeded,
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
    int emax = 1;
    uint bits = 1 + EBITS;
    size_t max_bytes = 1000; //* Does not matter for this test, just a bound.
    stream s(out_data, max_bytes);

    hls::stream<write_request_t, 32> write_queue;
    hls::stream<ap_uint<1>> write_fsm_finished;
    drain_write_queue_fsm(s, 1, write_queue, write_fsm_finished);

    // stream_write_bits(s, 2 * emax + 1, bits);
    // write_queue.write(write_request_t(0, bits, (uint64)(2 * emax + 1), false));
    // enqueue_emax(emax, bits, write_queue);

    zfp_output output;
    double tolerance = 1e-3;
    double error = set_zfp_output_accuracy(output, tolerance);

    //* Start encoding.
    uint prec = get_precision(emax, output.maxprec, output.minexp, 2);
    // *exceeded = exceeded_maxbits(output.maxbits, *maxprec, BLOCK_SIZE_2D);

    hls::stream<ublock_2d_t, 512> block;
    read_block_bitplane(ublock, block);

    hls::stream<uint, 32> encoded_bits_relay;
    encode_bitplanes_2d(block, output.minbits - MIN(output.minbits, bits), output.maxbits - bits, 
        prec, write_queue, encoded_bits_relay);
    // *encoded_bits = encode_bitplanes(s, ublock, *maxprec, BLOCK_SIZE_2D);

    read_outputs_bitplane(s, stream_idx, encoded_bits_relay, write_fsm_finished, encoded_bits);
    //* Close the stream.
    // stream_flush(s);
    // *stream_idx = s.idx;
  }
}