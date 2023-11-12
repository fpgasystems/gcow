// #include <ap_int.h>
#include <hls_stream.h>
#include <stdio.h>
#include <vector>

#include "constants.hpp"
#include "gcow.hpp"
#include "encode.hpp"
#include "stream.hpp"
#include "zfp.hpp"

void gcow(
  const zfp_input input_specs,
  const zfp_output output_specs,
  const float *in_fp_gradients,
  stream_word *out_zfp_gradients
)
{
//* Seperate input and output to the different memory banks.
#pragma HLS INTERFACE m_axi port=in_fp_gradients offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=out_zfp_gradients offset=slave bundle=gmem1

  //* Read IO specs from the global memory.
  zfp_input input = input_specs;
  zfp_output output = output_specs;
  size_t input_size = get_input_size(input);

  //* Read input gradients from the global memory.
  float in_fp_buffer[input_size];
  for (int i = 0; i < input_size; i++) {
    in_fp_buffer[i] = in_fp_gradients[i];
  }

  size_t out_bytes = get_max_output_bytes(output, input);
  size_t out_words = out_bytes / sizeof(stream_word);
  for (int i = 0; i < out_words; i++) {
    out_zfp_gradients[i] = (stream_word)i;
  }
}