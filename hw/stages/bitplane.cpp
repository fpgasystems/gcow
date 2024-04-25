#include "encode.hpp"
#include "stream.hpp"
#include "io.hpp"

void block_dispatcher(
  uint pe_idx,
  size_t total_blocks,
  const ublock_2d_t &block_buf,
  uint bemax, 
  uint prec,
  hls::stream<ublock_2d_t> &block_out,
  hls::stream<uint> &bemax_out,
  hls::stream<uint> &maxprec_out)
{
#pragma HLS INLINE off

  for (size_t block_id = pe_idx; block_id < total_blocks; block_id+=FIFO_WIDTH) {
    ublock_2d_t block_local = block_buf;
    block_local.id = block_id;

    block_out.write(block_local);
    bemax_out.write(bemax);
    maxprec_out.write(prec);
  }
}

void feed_block_bitplane(
  size_t total_blocks,
  const uint32 *ublock, 
  hls::stream<ublock_2d_t> block[FIFO_WIDTH], 
  uint bemax, 
  uint prec, 
  hls::stream<uint> out_bemax[FIFO_WIDTH], 
  hls::stream<uint> out_maxprec[FIFO_WIDTH])
{
  ublock_2d_t block_buf;

  read_block_bitplane_loop:
  for (int i = 0; i < BLOCK_SIZE_2D; i++) {
    #pragma HLS UNROLL factor=16
    block_buf.data[i] = ublock[i];
  }

  feed_block_bitplane_loop:
  for (uint pe_idx = 0; pe_idx < FIFO_WIDTH; pe_idx++) {
    #pragma HLS UNROLL
    
    hls::stream<ublock_2d_t> &block_out = block[pe_idx];
    #pragma HLS DEPENDENCE variable=block class=array inter false
    hls::stream<uint> &bemax_out = out_bemax[pe_idx];
    #pragma HLS DEPENDENCE variable=out_bemax class=array inter false
    hls::stream<uint> &maxprec_out = out_maxprec[pe_idx];
    #pragma HLS DEPENDENCE variable=out_maxprec class=array inter false

    block_dispatcher(pe_idx, total_blocks, block_buf, bemax, prec, block_out, bemax_out, maxprec_out);
  }
}

void write_outputs_bitplane(
  size_t total_blocks,
  stream &s, 
  ptrdiff_t *stream_idx, 
  hls::stream<bit_t> &write_fsm_finished)
{
  await(write_fsm_finished);
  *stream_idx = 4;
}

////////////////////////////////////////////////////////////////////////

void outputbuf_consumer(
  uint pe_idx,
  size_t total_blocks,
  hls::stream<outputbuf> &outbuf)
{
#pragma HLS INLINE off

  for (size_t block_id = pe_idx; block_id < total_blocks; block_id+=FIFO_WIDTH) {
    outbuf.read();
  }
} 

void write_outputs_bitplane_new(
  size_t total_blocks,
  stream &s, 
  ptrdiff_t *stream_idx, 
  hls::stream<outputbuf> outbufs[FIFO_WIDTH],
  hls::stream<bit_t> write_fsm_finished[FIFO_WIDTH])
{
#pragma HLS PIPELINE II=1

  for (uint pe_idx = 0; pe_idx < FIFO_WIDTH; pe_idx++) {
    //! Should be Sequential reads.
    #pragma HLS UNROLL
    
    hls::stream<outputbuf> &outbuf = outbufs[pe_idx];
    #pragma HLS DEPENDENCE variable=outbufs class=array inter false

    outputbuf_consumer(pe_idx, total_blocks, outbuf);
  }

  // for (uint pe_idx = 0; pe_idx < FIFO_WIDTH; pe_idx++) {
  //   #pragma HLS UNROLL
  //   write_fsm_finished[pe_idx].read();
  // }
  *stream_idx = 4;
}

void aggregator(
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

void aggregate_queues(
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

    aggregator(pe_idx, total_blocks, write_queue, outbuf);
  }
}


extern "C" {
  void bitplane(
    const uint32 *ublock, size_t total_blocks, stream_word *out_data, ptrdiff_t *stream_idx)
  {
#pragma HLS INTERFACE mode=m_axi port=ublock offset=slave bundle=gmem1
#pragma HLS INTERFACE mode=m_axi port=out_data offset=slave bundle=gmem2 max_write_burst_length=128
#pragma HLS INTERFACE mode=m_axi port=stream_idx offset=slave bundle=gmem0

#pragma HLS DATAFLOW
    uint bits = 1 + EBITS;
    size_t max_bytes = 20 * (1 << 20); //* 20 MiB
    stream s(out_data, max_bytes);

    hls::stream<write_request_t, 16> write_queues[FIFO_WIDTH];
    // #pragma HLS BIND_STORAGE variable=write_queues type=fifo impl=LUTRAM
    hls::stream<bit_t> write_fsm_finished;
    // hls::stream<bit_t> write_fsm_finished[FIFO_WIDTH];
    // drain_write_queue_fsm(total_blocks, s, write_queues, write_fsm_finished);

    zfp_output output;
    double tolerance = 1e-3;
    double error = set_zfp_output_accuracy(output, tolerance);

    int emax = 1;
    uint prec = get_precision(emax, output.maxprec, output.minexp, 2);
    uint biased_emax = prec ? (uint)(emax + EBIAS) : 0;

    hls::stream<uint, 16> bemax_relay[FIFO_WIDTH];
    hls::stream<uint, 16> maxprec_relay[FIFO_WIDTH];
    hls::stream<ublock_2d_t, 16> block[FIFO_WIDTH];
    // #pragma HLS BIND_STORAGE variable=block type=fifo impl=LUTRAM

    feed_block_bitplane(total_blocks, ublock, block, biased_emax, prec, bemax_relay, maxprec_relay);

    // encode_bitplanes_2d(total_blocks, bemax_relay, maxprec_relay, block, output, write_queues);
    encode_bitplanes_2d_par(total_blocks, bemax_relay, maxprec_relay, block, output, write_queues);

    // drain_write_queue_fsm(total_blocks, s, write_queues, write_fsm_finished);
    // drain_write_queue_fsm_par(total_blocks, s, write_queues, write_fsm_finished);

    hls::stream<outputbuf, 4> outputbufs[FIFO_WIDTH]; 
    // aggregate_write_queues(total_blocks, write_queues, outputbufs);
    aggregate_queues(total_blocks, write_queues, outputbufs);

    // hls::stream<stream_word, BURST_SIZE> words;
    // hls::stream<uint, 8> counts;
    // write_encodings(total_blocks, outputbufs, words, counts, out_data);
    // burst_write(words, counts, out_data, write_fsm_finished);
    burst_write_encodings(total_blocks, outputbufs, out_data, write_fsm_finished);
     
    write_outputs_bitplane(total_blocks, s, stream_idx, write_fsm_finished);
    // write_outputs_bitplane_new(total_blocks, s, stream_idx, outputbufs, write_fsm_finished);
  }
}