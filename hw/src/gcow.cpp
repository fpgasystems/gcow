#include <hls_stream.h>
#include <stdio.h>
#include <vector>

#include "gcow.hpp"
#include "encode.hpp"
#include "stream.hpp"
#include "zfp.hpp"

void gcow(
  const size_t in_dim,
  const size_t *in_shape,
  const float *in_fp_gradients,
  stream_word *out_zfp_gradients,
  size_t *out_bytes)
{
//* Seperate input and output to the different memory banks for now.
#pragma HLS INTERFACE mode=m_axi port=in_shape offset=slave bundle=gmem0
#pragma HLS INTERFACE mode=m_axi port=in_fp_gradients offset=slave bundle=gmem0 max_read_burst_length=256
#pragma HLS INTERFACE mode=m_axi port=out_zfp_gradients offset=slave bundle=gmem1 // max_write_burst_length=256
#pragma HLS INTERFACE mode=m_axi port=out_bytes offset=slave bundle=gmem1
// #pragma HLS INTERFACE s_axilite port=in_dim bundle=control
// #pragma HLS INTERFACE s_axilite port=in_shape bundle=control
// #pragma HLS INTERFACE s_axilite port=in_fp_gradients bundle=control
// #pragma HLS INTERFACE s_axilite port=out_zfp_gradients bundle=control
// #pragma HLS INTERFACE s_axilite port=out_bytes bundle=control
// #pragma HLS INTERFACE s_axilite port=return bundle=control

//& AXILite somehow doesn't work with Vitis HLS for transfering scalars.
// #pragma HLS INTERFACE mode=s_axilite port=out_bytes

  //* Read input shape from the global memory.
  size_t input_shape[DIM_MAX];
LOOP_READ_SHAPE:
  for (int i = 0; i < in_dim; i++) {
    input_shape[i] = in_shape[i];
  }

  //* Initialize input.
  zfp_input input(dtype_float, input_shape, in_dim);
  input.data = in_fp_gradients;

  //* Initialize output.
  zfp_output output;
  double tolerance = 1e-3;
  double max_error = set_zfp_output_accuracy(output, tolerance);
  size_t max_output_bytes = get_max_output_bytes(output, input);
  stream output_stream(out_zfp_gradients, max_output_bytes);
  output.data = output_stream;

  *out_bytes = zfp_compress(output, input);

}