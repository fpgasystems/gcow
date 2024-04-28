#include "types.hpp"
#include "stream.hpp"
#include <bitset>


// hls::stream<write_request_t, 1024*4> g_write_queue; /* write request for all blocks */
// hls::stream<write_request_t, 1024*4> g_write_backlog; 
size_t g_write_block_id = 0; /* block id of the current write request */
size_t g_write_index = 0; /* index of the current write request in the block */

void await(hls::stream<bit_t> &finished)
{
  finished.read();
}

/* Append n zero-bits to stream (n >= 0) */
void pad(outputbuf &o, uint64 n)
{
#pragma HLS INLINE
  o.buffered_bits += (size_t)n;
}

/* Write single bit (must be 0 or 1) */
void write_bit(outputbuf &o, ap_uint<1> bit)
{
#pragma HLS INLINE

  // std::cout << "write_bit: " << bit << std::endl;

  //?! Why can't we cast `bit` to `buffer_t` first?
  o.buffer += buffer_t(bit) << o.buffered_bits;
  ++o.buffered_bits;

  // std::bitset<200> val1(o.buffer);
  // std::cout << "Buffer: " << val1 << std::endl;
  // std::cout << "Buffered bits: " << o.buffered_bits << std::endl;
}

/* Buffer/write 0 <= n <= 64 low bits of value and return remaining bits */
void write_bits(outputbuf &o, uint64 value, size_t n)
{
#pragma HLS INLINE

  // std::bitset<64> val2(value);
  // std::cout << "write_bits: " << val2 << " (" << n << " bits)" << std::endl;

  //* The `value` is shifted left by the number of buffered bits.
  //* For example, if the buffer is 0b0101 and the value is 0b11, then the buffer becomes 0b110101.
  buffer_t val = value;
  o.buffer += val << o.buffered_bits;
  //& assert(0 <= s.buffered_bits < wsize); // The number of bits written in the buffer)
  o.buffered_bits += n;
  
  //! Clean up the higher bits.
  uint margin = BUFFER_SIZE - o.buffered_bits;
  o.buffer <<= margin;
  o.buffer >>= margin;
  //& assert(outbuf.buffered_bits <= BUFFER_SIZE);

  // std::bitset<200> val1(o.buffer);
  // std::cout << "Buffer: " << val1 << std::endl;
  // std::cout << "Buffered bits: " << o.buffered_bits << std::endl;
}

void queue_aggregator(
  uint pe_idx,
  size_t total_blocks,
  hls::stream<write_request_t> &write_queue,
  hls::stream<outputbuf> &outbuf)
{
#pragma HLS INLINE off

  agg_block_loop: 
  for (size_t block_id = pe_idx; block_id < total_blocks; block_id+=FIFO_WIDTH) {

    write_request_t request_buf = write_request_t();
    outputbuf obuf = outputbuf();
    //! The initializers are NOT translated to the hardware!!!
    //! => Dangling signals and bits in the buffer.
    request_buf.last = false; 
    obuf.buffered_bits = 0;
    obuf.buffer = buffer_t(0);

    drain_request_loop: 
    for (uint index = 0; !request_buf.last; index++) {
      request_buf = write_queue.read();

      //* Using the block id from the request to determine the current block.
      //& assert(request_buf.block_id == block_id);
      // block_id = request_buf.block_id;
      //* Verify the order of the write requests within a block for debugging (validated).
      // if (request_buf.index != index) {
      //   continue;
      // }

      if (request_buf.nbits > 1) {
        if (request_buf.value > 0) {
          write_bits(obuf, request_buf.value, request_buf.nbits);
        } else {
          //* Zero paddings.
          pad(obuf, request_buf.nbits);
          // std::cout << "pad: " << 0 << " (" << request_buf.nbits << " bits)" << std::endl;
        }
      } else if (request_buf.nbits == 1) {
        //! Must extract the least significant bit because the value is a 64-bit integer.
        //! E.g., value = 48(0b110000) -> bit = 0, otherwise, `stream_write_bit` handles it wrong.
        ap_uint<1> bit = request_buf.value & stream_word(1);
        // uint bit = request_buf.value & stream_word(1);
        write_bit(obuf, bit);
      } else {
        //* Empty request signals the end of an encoding stage, 
        //* expecting the `last` flag to be set.
      }
    }
    outbuf.write(obuf);

    // std::cout << "Block " << block_id << " encodings:" << std::endl;
    // for (int i = 0; i < BUFFER_SIZE; i+=SWORD_BITS) {
    //   std::bitset<64> val(obuf.buffer.range(i+SWORD_BITS-1, i));
    //   std::cout << "Drain encoding [" << i << "]:" << val << std::endl;
    //   if (i > obuf.buffered_bits) {
    //     break;
    //   }
    // }
  } 
  // fsm_finished.write(1);
}

/**
 ** Aggregates results from the write queues to the output buffers 
 ** in parallel using different PEs by unrolling the pulling loop.
*/
void aggregate_write_queues(
  size_t in_total_blocks,
  hls::stream<write_request_t> write_queues[FIFO_WIDTH],
  hls::stream<outputbuf> outbufs[FIFO_WIDTH])
{
#pragma HLS PIPELINE II=1

  size_t total_blocks = in_total_blocks;

  agg_pulling_loop:
  for (uint pe_idx = 0; pe_idx < FIFO_WIDTH; pe_idx++) {
    #pragma HLS UNROLL
    
    hls::stream<write_request_t> &write_queue = write_queues[pe_idx];
    #pragma HLS DEPENDENCE variable=write_queues class=array inter false
    hls::stream<outputbuf> &outbuf = outbufs[pe_idx];
    #pragma HLS DEPENDENCE variable=outbufs class=array inter false

    queue_aggregator(pe_idx, total_blocks, write_queue, outbuf);
  }
}

void burst_write(
  hls::stream<stream_word> &words,
  hls::stream<uint> &counts,
  stream_word *output_data,
  hls::stream<bit_t> &write_fsm_finished)
{
  uint write_idx = 0;
  while (true) {
    uint count = counts.read();
    if (count == BURST_SIZE) {
      burst_write_loop:
      for (uint i = 0; i < BURST_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        output_data[write_idx++] = words.read();
      }
    } else {
      burst_write_flush_loop:
      for (uint i = 0; i < count; i++) {
        #pragma HLS PIPELINE II=1
        output_data[write_idx++] = words.read();
      }
      break;
    }
  }
  write_fsm_finished.write(1);
}

/**
 ** Write out the encodings in the output buffers sequentially but in bursts.
*/
void burst_write_encodings(
  size_t in_total_blocks,
  hls::stream<outputbuf> outbufs[FIFO_WIDTH],
  // hls::stream<stream_word> &words,
  // hls::stream<uint> &counts,
  volatile stream_word *output_data,
  hls::stream<bit_t> &write_fsm_finished)
{
  size_t block_id = 0;
  size_t write_idx = 0;
  size_t total_blocks = in_total_blocks;
  residual_t residual_buf = 0;
  size_t residual_bits = 0;
  //* Cache the residuals for burst writing.
  stream_word burst_cache[BURST_SIZE];
  uint cache_idx = 0;
  // uint word_count = 0;

  write_encodings_block_loop: 
  for (; block_id < total_blocks; ) {
    
    //* Sequentially pulling from the output buffers of different PEs in order.
    write_encodings_loop:
    for (uint pe_idx = 0; pe_idx < FIFO_WIDTH && block_id < total_blocks; 
        pe_idx++, block_id++) {

      outputbuf obuf = outbufs[pe_idx].read();
      #pragma HLS DEPENDENCE variable=outbufs class=array inter false

      //* First, handle the residual bits from the previous block.
      //! I don't think we need to handle the below case.
      //& assert(residual_bits < SWORD_BITS);
      // if (residual_bits > 0) {
      //   //! WRONG: Append means to add the bits to the high significant bits, i.e., 00101 -> append 11 -> 1100101.
      //   int shift = SWORD_BITS - residual_bits;
      //   //* Narrower -> wider unsigned values: zero-padded.
      //   //! Casting before shifting.
      //   residual_t val = residual_t(obuf.buffer.range(shift-1, 0)) << residual_bits;
      //   residual_buf += val;
      //   output_data[write_idx++] = residual_buf;

      //   residual_buf = 0;
      //   residual_bits = 0;
      //   obuf.buffer >>= shift;
      //   obuf.buffered_bits -= shift;
      // }

      write_loop:
      for (uint i = 0; i < BUFFER_SIZE; i += SWORD_BITS) {
        // #pragma HLS UNROLL
        if (obuf.buffered_bits >= SWORD_BITS) {
          // std::bitset<64> val(obuf.buffer.range(SWORD_BITS-1, 0));
          // std::cout << "Output bits: " << val << " (" << SWORD_BITS << " bits)" << std::endl;
          // std::cout << "Buffered bits: " << obuf.buffered_bits << std::endl;

          // output_data[write_idx++] = obuf.buffer.range(SWORD_BITS-1, 0);
          burst_cache[cache_idx++] = obuf.buffer.range(SWORD_BITS-1, 0);
          // words.write(obuf.buffer.range(SWORD_BITS-1, 0));
          // word_count++;
          obuf.buffer >>= SWORD_BITS;
          obuf.buffered_bits -= SWORD_BITS;
        } else {
          // std::cout << "Buffer residual bits" << std::endl;
          // std::bitset<200> val1(residual_buf.range(SWORD_BITS-1, 0));
          // std::cout << "Before append: " << val1 << " (" << residual_bits << " bits)" << std::endl;

          //! Casting before shifting.
          residual_t val = residual_t(obuf.buffer.range(obuf.buffered_bits-1, 0)) << residual_bits;
          residual_buf += val;
          //& assert (residual_bits < 512)
          residual_bits += obuf.buffered_bits;

          // std::bitset<200> val2(residual_buf.range(SWORD_BITS-1, 0));
          // std::cout << "After append: " << val2 << " (" << residual_bits << " bits)" << std::endl;
          obuf.buffer = 0;
          obuf.buffered_bits = 0;

          //TODO: Need a while loop to write until exhausting the bits?
          if (residual_bits >= SWORD_BITS) {
            // output_data[write_idx++] = residual_buf.range(SWORD_BITS-1, 0);
            burst_cache[cache_idx++] = residual_buf.range(SWORD_BITS-1, 0);
            // words.write(residual_buf.range(SWORD_BITS-1, 0));
            // word_count++;
            // std::bitset<200> val(residual_buf.range(SWORD_BITS-1, 0));
            // std::cout << "Output bits: " << val << " (" << SWORD_BITS << " bits)" << std::endl;

            residual_buf >>= SWORD_BITS;
            residual_bits -= SWORD_BITS;
            // std::cout << "Residual bits: " << residual_bits << std::endl;
          }
          // break;
        }
        // if (word_count == BURST_SIZE) {
        //   counts.write(BURST_SIZE);
        //   word_count = 0;
        // }
        if (cache_idx == BURST_SIZE) {
          burst_write_loop:
          for (uint j = 0; j < BURST_SIZE; j++) {
            #pragma HLS PIPELINE II=1
            output_data[write_idx++] = burst_cache[j];
          }
          cache_idx = 0;
        }
      } // write_loop
    } // batch_write_loop
  } // batch_write_block_loop

  //* Flush residual buffer.
  while (residual_bits > 0) {
    // std::bitset<64> val1(residual_buf.range(SWORD_BITS-1, 0));
    // std::cout << "Before flush: " << val1 << std::endl;
    // std::cout << "residual_bits: " << residual_bits << std::endl;
    //! No need for shifting, since the writing is from LSB to MSB.
    output_data[write_idx++] = residual_buf.range(SWORD_BITS-1, 0);
    // words.write(residual_buf.range(SWORD_BITS-1, 0));
    // word_count++;
    residual_buf >>= SWORD_BITS;
    //! Cap subtraction, otherwise it will cause undefined behavior on `size_t` type in HW.
    residual_bits -= MIN(SWORD_BITS, residual_bits);
    // std::bitset<64> val2(residual_buf.range(SWORD_BITS-1, 0));
    // std::cout << "Flush: " << val2 << std::endl;
  }
  //* Flush the cached residuals.
  // if (word_count > 0) {
  //   counts.write(word_count);
  // }
  if (cache_idx > 0) {
    burst_flush_loop:
    for (uint j = 0; j < cache_idx; j++) {
      #pragma HLS PIPELINE II=1
      output_data[write_idx++] = burst_cache[j];
    }
  }
  write_fsm_finished.write(1);
}

void drain_write_queue(
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
          // std::bitset<64> val(request_buf.value);
          // std::cout << "write_bits: " << val << " (" << request_buf.nbits << " bits)" << std::endl;
        } else {
          //* Zero paddings.
          stream_pad(s, request_buf.nbits);
          // std::cout << "pad: " << 0 << " (" << request_buf.nbits << " bits)" << std::endl;
        }
      } else if (request_buf.nbits == 1) {
        //! Must extract the least significant bit because the value is a 64-bit integer.
        //! E.g., value = 48(0b110000) -> bit = 0, otherwise, `stream_write_bit` handles it wrong.
        uint bit = request_buf.value & stream_word(1);
        stream_write_bit(s, bit);
        // std::cout << "write_bit: " << bit << std::endl;
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
 
void drain_write_queue_fsm_index(
  size_t in_total_blocks,
  stream &s,
  hls::stream<write_request_t> write_queues[FIFO_WIDTH],
  hls::stream<bit_t> &write_fsm_finished)
{
  drain_queue_loop: 
  for (size_t block_id = 0; block_id < in_total_blocks; block_id++) {
    #pragma HLS PIPELINE II=1

    uint fifo_idx = FIFO_INDEX(block_id);
    write_request_t request_buf = write_queues[fifo_idx].read();
    #pragma HLS DEPENDENCE variable=write_queues class=array inter false
    uint index = 0;
    bool start_flag = true;
    write_request_loop: 
    for (; start_flag || !request_buf.last; 
        start_flag = false, request_buf = write_queues[fifo_idx].read()) {
      //* Blocking read (every stage has ≥1 write request).
      // request_buf = write_queues[fifo_idx].read();

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
          // std::bitset<64> val(request_buf.value);
          // std::cout << "write_bits: " << val << " (" << request_buf.nbits << " bits)" << std::endl;
        } else {
          //* Zero paddings.
          stream_pad(s, request_buf.nbits);
          // std::cout << "pad: " << 0 << " (" << request_buf.nbits << " bits)" << std::endl;
        }
      } else if (request_buf.nbits == 1) {
        //! Must extract the least significant bit because the value is a 64-bit integer.
        //! E.g., value = 48(0b110000) -> bit = 0, otherwise, `stream_write_bit` handles it wrong.
        uint bit = request_buf.value & stream_word(1);
        stream_write_bit(s, bit);
        // std::cout << "write_bit: " << bit << std::endl;
      } else {
        //* Empty request signals the end of an encoding stage, 
        //* expecting the `last` flag to be set.
      }
    }
  }
  //& assert(block_id == in_total_blocks);
  stream_flush(s);
  write_fsm_finished.write(1);
}

void drain_write_queue_fsm_par(
  size_t in_total_blocks,
  stream &s,
  hls::stream<write_request_t> write_queues[FIFO_WIDTH],
  hls::stream<bit_t> &write_fsm_finished)
{
  size_t block_id = 0;
  size_t total_blocks = in_total_blocks;

  drain_queue_loop: 
  for (; block_id < total_blocks; ) {
    #pragma HLS PIPELINE II=1

    drain_dispatch_loop:
    for (uint pe_idx = 0; 
        pe_idx < FIFO_WIDTH && block_id < total_blocks; 
        pe_idx++, block_id++) {

      uint index = 0;
      bool start_flag = true;
      write_request_t request_buf = write_queues[pe_idx].read();
      #pragma HLS DEPENDENCE variable=write_queues class=array inter false

      drain_request_loop: 
      for (; start_flag || !request_buf.last; 
          start_flag = false, request_buf = write_queues[pe_idx].read()) {
        //* Blocking read (every stage has ≥1 write request).
        // request_buf = write_queues[fifo_idx].read();

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
            // std::bitset<64> val(request_buf.value);
            // std::cout << "write_bits: " << val << " (" << request_buf.nbits << " bits)" << std::endl;
          } else {
            //* Zero paddings.
            stream_pad(s, request_buf.nbits);
            // std::cout << "pad: " << 0 << " (" << request_buf.nbits << " bits)" << std::endl;
          }
        } else if (request_buf.nbits == 1) {
          //! Must extract the least significant bit because the value is a 64-bit integer.
          //! E.g., value = 48(0b110000) -> bit = 0, otherwise, `stream_write_bit` handles it wrong.
          uint bit = request_buf.value & stream_word(1);
          stream_write_bit(s, bit);
          // std::cout << "write_bit: " << bit << std::endl;
        } else {
          //* Empty request signals the end of an encoding stage, 
          //* expecting the `last` flag to be set.
        }
      }
    }
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
