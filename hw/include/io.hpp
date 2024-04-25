#ifndef IO_HPP
#define IO_HPP

#include "types.hpp"

void pad(outputbuf &o, uint64 n);

void write_bit(outputbuf &o, ap_uint<1> bit);

void write_bits(outputbuf &o, uint64 value, size_t n);

void await(hls::stream<bit_t> &finished);

void aggregate_write_queues(
  size_t in_total_blocks,
  hls::stream<write_request_t> write_queues[FIFO_WIDTH],
  hls::stream<outputbuf> outbufs[FIFO_WIDTH]);

void burst_write(
  hls::stream<stream_word> &words,
  hls::stream<uint> &counts,
  stream_word *output_data,
  hls::stream<bit_t> &write_fsm_finished);

void burst_write_encodings(
  size_t in_total_blocks,
  hls::stream<outputbuf> outbufs[FIFO_WIDTH],
  // hls::stream<stream_word> &words,
  // hls::stream<uint> &counts,
  stream_word *output_data,
  hls::stream<bit_t> &write_fsm_finished);

void drain_write_queue(
  size_t in_total_blocks,
  stream &s,
  hls::stream<write_request_t> &write_queue,
  hls::stream<bit_t> &write_fsm_finished);

void drain_write_queue_fsm_par(
  size_t in_total_blocks,
  stream &s,
  hls::stream<write_request_t> write_queues[FIFO_WIDTH],
  hls::stream<bit_t> &write_fsm_finished);

void relay_scalers_2d(
  hls::stream<fblock_2d_t> &in_block, 
  hls::stream<float> &out_float);

// void drain_write_queue_fsm(
//   stream &s, 
//   size_t total_blocks,
//   hls::stream<write_request_t> &write_queue, 
//   hls::stream<bit_t> &write_fsm_finished);

void write_bits(
  write_request_t &in_wrequest, 
  hls::stream<write_request_t> &write_queue, 
  hls::stream<uint64> &out_residual);

void pull_write_requests_fsm(
  zfp_output &output,
  hls::stream<write_request_t> &write_requests,
  hls::stream<write_request_t> &write_backlog);

void fill_write_requests_fsm(
  hls::stream<write_request_t> &write_queue,
  hls::stream<write_request_t> &write_requests,
  hls::stream<write_request_t> &write_backlog);

#endif /* IO_HPP */