#include "encode.hpp"
#include "stream.hpp"
#include "io.hpp"


void enqueue_write_request(
  hls::stream<write_request_t> &write_queue)
{
  write_queue.write(write_request_t(0, 0, 1, false));
  //* Actual data 1.
  write_queue.write(write_request_t(0, 64, 7455816852505100291UL, false));
  write_queue.write(write_request_t(0, 0, 2, false));
  write_queue.write(write_request_t(0, 0, 3, false));
  //* Actual data 2.
  write_queue.write(write_request_t(0, 9, 432UL, false));
  //* Indicate the end of the block.
  write_queue.write(write_request_t(0, 0, 4, true));
}

void dequeue_write_request(
  stream &s, hls::stream<ap_uint<1>> &write_fsm_finished, ptrdiff_t *stream_idx)
{
  write_fsm_finished.read();
  *stream_idx = s.idx;
}

extern "C" {
  void writes(stream_word *out_data, ptrdiff_t *stream_idx)
  {
#pragma HLS INTERFACE mode=m_axi port=out_data offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=stream_idx offset=slave bundle=gmem1

#pragma HLS DATAFLOW

    size_t max_bytes = 1000; //* Does not matter for this test, just a bound.
    stream s(out_data, max_bytes);

    hls::stream<write_request_t, 32> write_queue;
    hls::stream<ap_uint<1>> write_fsm_finished;

    //* Launch the write FSMs.
    drain_write_queue_fsm(s, 1, write_queue, write_fsm_finished);

    //* Write some bits.
    enqueue_write_request(write_queue);
    // write_queue.write(write_request_t(0, 64, 7455816852505100291UL, false));
    // write_queue.write(write_request_t(0, 9, 432UL, true));
    // stream_write_bits(s, 7455816852505100291UL, 64);
    // stream_write_bits(s, 432UL, 9);

    //* Close the stream.
    dequeue_write_request(s, write_fsm_finished, stream_idx);
  }
}