#include "encode.hpp"
#include "stream.hpp"

extern "C" {
  void writes(stream_word *out_data, ptrdiff_t *stream_idx)
  {
#pragma HLS INTERFACE mode=m_axi port=out_data offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=stream_idx offset=slave bundle=gmem0

    size_t max_bytes = 1000; //* Does not matter for this test, just a bound.
    stream s(out_data, max_bytes);

    //* Write some bits.
    stream_write_bits(s, 7455816852505100291UL, 64);
    stream_write_bits(s, 432UL, 9);
    //* Close the stream.
    stream_flush(s);
    *stream_idx = s.idx;
  }
}