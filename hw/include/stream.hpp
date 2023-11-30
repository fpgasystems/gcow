#ifndef STREAM_HPP
#define STREAM_HPP

#include "types.hpp"

void stream_pad(stream &s, uint64 n);
stream_word stream_read_word(stream &s);
void stream_write_word(stream &s, stream_word value);
uint64 stream_read_bits(stream &s, size_t n);
uint64 stream_write_bits(stream &s, uint64 value, size_t n);
uint stream_write_bit(stream &s, uint bit);
uint64 stream_woffset(stream &s);
void stream_rewind(stream &s);
size_t stream_size_bytes(const stream &s);
// stream &stream_init(void* buffer, size_t bytes);
size_t stream_flush(stream &s);

#endif /* STREAM_HPP */