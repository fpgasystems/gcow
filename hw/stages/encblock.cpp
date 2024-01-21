#include "encode.hpp"
#include "stream.hpp"

extern "C" {
  //! Avoid having the kernel name being the same as one of the arguments, e.g., "block."
  void encblock(
    volatile int32 *iblock, double *max_error, uint *maxprec, bool *exceeded,
    stream_word *out_data, uint *encoded_bits, ptrdiff_t *stream_idx)
  {
#pragma HLS INTERFACE mode=m_axi port=iblock offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=max_error offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=maxprec offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=exceeded offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=out_data offset=slave bundle=gmem1
#pragma HLS INTERFACE mode=m_axi port=encoded_bits offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=stream_idx offset=slave bundle=gmem0

    int e = 9;
    uint bits = 1 + EBITS;
    size_t max_bytes = 1000; //* Does not matter for this test, just a bound.
    stream s(out_data, max_bytes);

    stream_write_bits(s, 2 * e + 1, bits);

    zfp_output output;
    double tolerance = 1e-3;
    *max_error = set_zfp_output_accuracy(output, tolerance);

    //* Start encoding.
    *maxprec = get_precision(e, output.maxprec, output.minexp, 2);
    *exceeded = exceeded_maxbits(output.maxbits, *maxprec, BLOCK_SIZE_2D);
    // *encoded_bits = encode_iblock(s, output.minbits, output.maxbits, *maxprec,
                                  // iblock, 2);
    //* Close the stream.
    stream_flush(s);
    *stream_idx = s.idx;
  }
}