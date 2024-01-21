#ifndef IO_HPP
#define IO_HPP

#include "types.hpp"


void pad_bits(
  hls::stream<uint> &in_encoded_bits, 
  uint bits,
  uint minbits,
  size_t block_id,
  hls::stream<write_request_t> &write_queue,
  hls::stream<uint> &out_encoded_bits);

void drain_write_queue_fsm(zfp_output &output, hls::stream<write_request_t> &write_queue, size_t total_blocks);

void relay_scalers_2d(
  hls::stream<fblock_2d_t> &in_block, 
  hls::stream<float> &out_float);

void drain_write_queue_fsm(
  stream &s, 
  size_t total_blocks,
  hls::stream<write_request_t> &write_queue, 
  hls::stream<ap_uint<1>> &write_fsm_finished);

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