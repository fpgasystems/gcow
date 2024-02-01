#include "types.hpp"
#include "stream.hpp"


// hls::stream<write_request_t, 1024*4> g_write_queue; /* write request for all blocks */
// hls::stream<write_request_t, 1024*4> g_write_backlog; 
size_t g_write_block_id = 0; /* block id of the current write request */
size_t g_write_index = 0; /* index of the current write request in the block */

void await_fsm(hls::stream<bit_t> &finished)
{
  finished.read();
}

void drain_write_queue_fsm(
  size_t in_total_blocks,
  stream &s,
  hls::stream<write_request_t> &write_queue,
  hls::stream<bit_t> &write_fsm_finished)
{
  write_queue_loop: for (size_t block_id = 0; block_id < in_total_blocks; block_id++) {
    write_request_t request_buf;
    uint index = 0;
    do {
      //* Blocking read (every stage has ≥1 write request).
      request_buf = write_queue.read();
      //* Using the block id from the request to determine the current block.
      //& assert(request_buf.block_id == block_id);
      block_id = request_buf.block_id;
      if (request_buf.index != index++) {
        //* Verify the order of the write requests within a block for debugging (validated).
        continue;
      }

      if (request_buf.nbits > 1) {
        if (request_buf.value > 0) {
          stream_write_bits(s, request_buf.value, request_buf.nbits);
        } else {
          //* Zero paddings.
          stream_pad(s, request_buf.nbits);
        }
      } else if (request_buf.nbits == 1) {
        stream_write_bit(s, request_buf.value);
      } else {
        //* Empty request signals the end of an encoding stage, 
        //* expecting the `last` flag to be set.
      }
    } while (!request_buf.last);
  }
  //& assert(block_id == in_total_blocks);
  stream_flush(s);
  write_fsm_finished.write(1);
}

// void drain_write_queue_fsm(
//   stream &s, 
//   size_t total_blocks,
//   hls::stream<write_request_t> &write_queue, 
//   hls::stream<bit_t> &write_fsm_finished)
// {
//   //! Assuming requests are in order.
//   size_t block_id = 0;
//   write_queue_loop: for (; block_id < total_blocks; ) {
//     //* Blocking read.
//     write_request_t request_buf = write_queue.read();
//     // if (request_buf.block_id != block_id) {
//     //   //! Something is wrong.
//     //   break;
//     // }
//     if (request_buf.nbits > 1) {
//       if (request_buf.value > 0) {
//         stream_write_bits(s, request_buf.value, request_buf.nbits);
//       } else {
//         //* Zero paddings.
//         stream_pad(s, request_buf.nbits);
//       }
//     } else if (request_buf.nbits == 1) {
//       stream_write_bit(s, request_buf.value);
//     } else {
//       //* Empty request signals the end of the block, expecting the `last` flag to be set.
//     }
//     if (request_buf.last) {
//       block_id++;
//     }
//   }
//   if (block_id == total_blocks) {
//     stream_flush(s);
//     write_fsm_finished.write(1);
//   }
// }

void relay_scalers_2d(
  hls::stream<fblock_2d_t> &in_block, 
  hls::stream<float> &out_float)
{
  while (true) {
    fblock_2d_t block_buf = in_block.read();
    relay_block2d_loop: for (int i = 0; i < BLOCK_SIZE_2D; i++) {
      #pragma HLS PIPELINE II=1
      out_float << block_buf.data[i];
    }
  }
}

void write_bits(
  write_request_t &in_wrequest, 
  hls::stream<write_request_t> &write_queue, 
  hls::stream<uint64> &out_residual)
{
  //* Enqueue the write request.
  write_queue.write(in_wrequest);
  //* Return the updated value immediately as if it was written already.
  out_residual.write(in_wrequest.value >> in_wrequest.nbits);
}

void pull_write_requests_fsm(
  zfp_output &output,
  hls::stream<write_request_t> &write_requests,
  hls::stream<write_request_t> &write_backlog) 
{
  while (true) {
    //* Blocking read.
    write_request_t request_buf = write_requests.read();
    //* Pull the write request from the queue until the desired next block is reached.
    //TODO: Could be more efficient
    while (!(request_buf.block_id == g_write_block_id)) {
      //* Enqueue the current write request again.
      write_backlog << request_buf;
      //* Pull out another write request.
      write_requests >> request_buf;
    }
    //* Write the requested bits to the output stream.
    // stream_write_bits(output.data, request_buf.value, request_buf.nbits);
    //* Update the block id and index.
    if (request_buf.last) {
      g_write_block_id++;
      g_write_index = 0;
    } else {
      g_write_index++;
    }
  }
}

void fill_write_requests_fsm(
  hls::stream<write_request_t> &write_queue,
  hls::stream<write_request_t> &write_requests,
  hls::stream<write_request_t> &write_backlog) 
{
  while (true) {
    //* Non-blocking reads.
    if (!write_queue.empty()) {
      write_requests << write_queue.read();
    }
    if (!write_backlog.empty()) {
      //* Push the write request from the backlog back to the queue.
      write_requests <<  write_backlog.read();
    }
  }
}