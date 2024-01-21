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

  // size_t MAX_NUM_BLOCKS_2D = 200000000; //* 200M blocks (3B parameters)
  // hls::stream<write_request_t> write_queues[MAX_NUM_BLOCKS_2D];
  size_t total_blocks = get_input_num_blocks(input);
  hls::stream<write_request_t> write_queue;
  hls::stream<ap_uint<1>> write_fsm_finished;

  //^ Step 0: Launch the write FSMs.
  drain_write_queue_fsm(output.data, total_blocks, write_queue, write_fsm_finished);
  // drain_write_queues_fsm(output, MAX_NUM_BLOCKS_2D, write_queues);
  // hls::stream<write_request_t, 1024*4> write_queue; /* write queue for all blocks */
  // #pragma HLS STREAM variable=write_queue type=pipo 
  // hls::stream<write_request_t, 1024*4> write_requests; 
  // hls::stream<write_request_t, 1024*4> write_backlog; 
  // pull_write_requests_fsm(output, write_requests, write_backlog);
  // fill_write_requests_fsm(write_queue, write_requests, write_backlog);

  //^ Step 1: Partition input data into 4x4 blocks.
  hls::stream<fblock_2d_t, 512> fblock;
  chunk_blocks_2d(fblock, input);

  //^ Step 2: Block floating-point transform.
  uint dim = 2;
  hls::stream<int, 32> emax;
  hls::stream<uint, 32> maxprec;
  hls::stream<fblock_2d_t, 512> fblock_relay;
  get_block_exponent(total_blocks, fblock, output, emax, maxprec, fblock_relay);

  steps_loop: for (int block_id = 0; block_id < total_blocks; block_id++) {
    //* Blocking reads.
    int e = emax.read();
    uint prec = maxprec.read();
    
    uint bits = 1;
    uint minbits = output.minbits;
    hls::stream<uint, 32> encoded_bits;
    //* Encode block only if biased exponent is nonzero
    if (e) {
      //* Encode block exponent
      bits += EBITS;
      // uint64 tmp;
      // stream_write_bits(output.data, (2 * e + 1), bits, &tmp);
      //TODO: Parallel I/O and compute.
      write_queue.write(
        write_request_t(block_id, bits, (uint64)(2 * e + 1), false));

      //^ Step 3: Cast floats to integers.
      hls::stream<iblock_2d_t, 512> iblock;
      fwd_float2int_2d(fblock_relay, e, iblock);

      //^ Step 4: Perform forward decorrelation transform.
      hls::stream<iblock_2d_t, 512> iblock_relay;
      fwd_decorrelate_2d(iblock, iblock_relay);

      //^ Step 5: Perform forward block reordering transform and convert to unsigned integers.
      hls::stream<ublock_2d_t, 512> ublock;
      fwd_reorder_int2uint_2d(iblock_relay, ublock);

      //^ Step 6: Bit plane encoding.
      minbits = output.minbits - MIN(bits, output.minbits);
      encode_bitplanes_2d(
        ublock, minbits, output.maxbits - bits, prec, write_queue, encoded_bits);

      //& Temporary ///////////////////////////////
      // bits += encoded_bits.read();
      // ublock.read();
      //& Temporary ///////////////////////////////
    } 
    else {
      /* write single zero-bit to indicate that all values are zero */
      //* Compress a block of all zeros and add padding if it's fixed-rate.
      // write_queue.write(
      //   write_request_t(block_id, 1, (uint64)0, true /* last bit */));
      encoded_bits.write(0);
      // stream_write_bit(output.data, 0);
    }

    hls::stream<uint, 32> encoded_bits_relay;
    //* Padding to the minimum number of bits required.
    pad_bits(encoded_bits, bits, minbits, block_id, write_queue, encoded_bits_relay);
    //* Unused for now.
    encoded_bits_relay.read();
    // if (bits < minbits) {
    //   // stream_pad(output.data, minbits - bits);
    //   write_queue.write(
    //     write_request_t(block_id, minbits - bits, (uint64)0, true));
    //   bits = minbits;
    // }
  }
  //* Blocking until the write FSM finishes.
  write_fsm_finished.read();


  //& Temporary ///////////////////////////////
  // for (int k = 0; k < total_blocks; k++) {
  //   int e = emax_relay.read();
  // }
  // iblock.read();
  // for (int k = 0; k < total_blocks*BLOCK_SIZE_2D; k++) {
  //   int32 x = integer.read();
  // }
  //& Temporary ///////////////////////////////
}