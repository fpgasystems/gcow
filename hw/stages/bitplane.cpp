#include "encode.hpp"
#include "stream.hpp"

extern "C" {
  uint encode_bitplanes(stream &s, volatile const uint32 *const ublock,
                        uint maxprec, uint block_size)
  {
    /* make a copy of bit stream to avoid aliasing */
    //! CHANGE: No copy is made here!
    // stream s = *out_data;
    uint64 offset = stream_woffset(s);
    uint intprec = (uint)(CHAR_BIT * sizeof(uint32));
    //* `kmin` is the cutoff of the least significant bit plane to encode.
    //* E.g., if `intprec` is 32 and `maxprec` is 17, only [31:16] bit planes are encoded.
    uint kmin = intprec > maxprec ? intprec - maxprec : 0;
    uint k = intprec;
    uint n = 0;

    /* encode one bit plane at a time from MSB to LSB */
LOOP_ENCODE_ALL_BITPLANES:
    for (k = intprec, n = 0; k-- > kmin;) {
      //^ Step 1: extract bit plane #k of every block to x.
      stream_word x(0);
LOOP_ENCODE_ALL_BITPLANES_TRANSPOSE:
      for (uint i = 0; i < block_size; i++) {
        // x += (uint64)((ublock[i] >> k) & 1u) << i;
        x += ((stream_word(ublock[i]) >> k) & stream_word(1)) << i;
      }

      // //^ Step 2: encode first n bits of bit plane.
      // x = stream_write_bits(s, x, n);

      //^ Step 3: unary run-length encode remainder of bit plane.
LOOP_ENCODE_ALL_BITPLANES_EMBED:
      for (; n < block_size; x >>= 1, n++) {
        if (!stream_write_bit(s, !!x)) {
          //^ Negative group test (x == 0) -> Done with all bit planes.
          break;
        }
        for (; n < block_size - 1; x >>= 1, n++) {
          //* Continue writing 0's until a 1 bit is found.
          //& `x & 1u` is used to extract the least significant (right-most) bit of `x`.
          if (stream_write_bit(s, x & stream_word(1))) {
            //* After writing a 1 bit, break out for another group test
            //* (to see whether the bitplane code `x` turns 0 after encoding `n` of its bits).
            //* I.e., for every 1 bit encoded, do a group test on the rest.
            break;
          }
        }
      }
    }
    // *out_data = s;
    //* Returns the number of bits written.
    return (uint)(stream_woffset(s) - offset);
  }


  void bitplane(
    const uint32 *ublock, double *max_error, uint *maxprec, bool *exceeded,
    stream_word *out_data, uint *encoded_bits, ptrdiff_t *stream_idx)
  {
#pragma HLS INTERFACE mode=m_axi port=ublock offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=max_error offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=maxprec offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=exceeded offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=out_data offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=encoded_bits offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=stream_idx offset=slave bundle=gmem0

    int emax = 1;
    uint bits = 1 + EBITS;
    size_t max_bytes = 1000; //* Does not matter for this test, just a bound.
    stream s(out_data, max_bytes);

    stream_write_bits(s, 2 * emax + 1, bits);

    zfp_output output;
    double tolerance = 1e-3;
    *max_error = set_zfp_output_accuracy(output, tolerance);

    //* Start encoding.
    *maxprec = get_precision(emax, output.maxprec, output.minexp, 2);
    *exceeded = exceeded_maxbits(output.maxbits, *maxprec, BLOCK_SIZE_2D);
    *encoded_bits = encode_bitplanes(s, ublock, *maxprec, BLOCK_SIZE_2D);
    //* Close the stream.
    stream_flush(s);
    *stream_idx = s.idx;
  }
}