#include "types.h"

struct stream {
  size_t buffered_bits;        /* number of buffered bits (0 <= bits < WORD_BITS) */
  word buffer; /* incoming/outgoing bits (buffer < 2^bits) */
  word* ptr;   /* pointer to next word to be read/written */
  word* begin; /* beginning of stream */
  word* end;   /* end of stream (not enforced) */
// #ifdef BIT_STREAM_STRIDED
//   size_t mask;           /* one less the block size in number of words */
//   ptrdiff_t delta;       /* number of words between consecutive blocks */
// #endif
};

void stream_pad(stream* s, uint64 n);
word stream_read_word(stream* s);
void stream_write_word(stream* s, word value);
uint64 stream_read_bits(stream* s, size_t n);
uint64 stream_write_bits(stream* s, uint64 value, size_t n);
uint stream_write_bit(stream* s, uint bit);
uint64 stream_woffset(stream* s);