#include <stdio.h>

#include "types.hpp"
#include "stream.hpp"
#include "encode.hpp"
#include "zfp.hpp"
#include "io.hpp"


size_t zfp_compress(zfp_output &output, const zfp_input &input)
{
  switch (get_input_dimension(input)) {
    case 2:
      zfp_compress_2d(output, input);
      break;
    case 3:
      //TODO
      break;
    case 4:
      //TODO
      break;
    default:
      break;
  }

  // stream_flush(output.data);
  return stream_size_bytes(output.data);
}


void zfp_compress_2d(zfp_output &output, const zfp_input &input)
{
  #pragma HLS DATAFLOW
  
  size_t total_blocks = get_input_num_blocks(input);
  // hls::stream<write_request_t> write_queues[NUM_WRITE_QUEUES];
  hls::stream<write_request_t> write_queues[FIFO_WIDTH];
  hls::stream<bit_t> write_fsm_finished;

  //^ Step 0: Launch the write FSM and wait until it finishes.
  // drain_write_queue(total_blocks, output.data, write_queue, write_fsm_finished);
  await(write_fsm_finished);

  //^ Step 1: Partition input data into 4x4 blocks.
  hls::stream<fblock_2d_t, 16> fblock[FIFO_WIDTH];
  // chunk_blocks_2d_par(total_blocks, fblock, input);
  chunk_blocks_2d_seq(fblock, input);

  //^ Step 2: Block floating-point transform.
  hls::stream<int, 8> emax[FIFO_WIDTH];
  hls::stream<uint, 8> bemax[FIFO_WIDTH];
  hls::stream<uint, 8> maxprec[FIFO_WIDTH];
  hls::stream<fblock_2d_t, 16> fblock_relay[FIFO_WIDTH];
  compute_block_emax_2d(total_blocks, fblock, output, emax, bemax, maxprec, fblock_relay);

  //^ Step 3: Cast floats to integers.
  hls::stream<iblock_2d_t, 16> iblock[FIFO_WIDTH];
  hls::stream<uint, 8> bemax_relay[FIFO_WIDTH];
  fwd_float2int_2d_par(total_blocks, emax, bemax, fblock_relay, iblock, bemax_relay);

  //^ Step 4: Perform forward decorrelation transform.
  hls::stream<iblock_2d_t, 16> iblock_relay[FIFO_WIDTH];
  hls::stream<uint, 8> bemax_relay2[FIFO_WIDTH];
  fwd_decorrelate_2d_par(total_blocks, bemax_relay, iblock, iblock_relay, bemax_relay2);

  //^ Step 5: Perform forward block reordering transform and convert to unsigned integers.
  hls::stream<ublock_2d_t, 16> ublock[FIFO_WIDTH];
  hls::stream<uint, 8> bemax_relay3[FIFO_WIDTH];
  fwd_reorder_int2uint_2d_par(total_blocks, bemax_relay2, iblock_relay, ublock, bemax_relay3);

  //^ Step 6: Bit plane encoding.
  encode_bitplanes_2d_par(total_blocks, bemax_relay3, maxprec, ublock, output, write_queues);
  hls::stream<outputbuf, 4> outputbufs[FIFO_WIDTH]; 
  aggregate_write_queues(total_blocks, write_queues, outputbufs);
  burst_write_encodings(total_blocks, outputbufs, output.data.begin, write_fsm_finished);
}