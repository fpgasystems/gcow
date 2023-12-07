#include "types.h"
#include "stream.h"


/* Read a single word from memory */
stream_word stream_read_word(stream* s)
{
  // stream_word w = *s->ptr++;
  stream_word w = s->begin[s->idx++];
// #ifdef BIT_STREAM_STRIDED
//   if (!((s->ptr - s->begin) & s->mask))
//     s->ptr += s->delta;
// #endif
  return w;
}

/* Write a single word to memory */
void stream_write_word(stream* s, stream_word value)
{
  // *s->ptr++ = value;
  s->begin[s->idx++] = value;
// #ifdef BIT_STREAM_STRIDED
//   if (!((s->ptr - s->begin) & s->mask))
//     s->ptr += s->delta;
// #endif
}

/* read 0 <= n <= 64 bits */
uint64 stream_read_bits(stream* s, size_t n)
{
  uint64 value = s->buffer;
  if (s->buffered_bits < n) {
    /* keep fetching wsize bits until enough bits are buffered */
    do {
      /* assert: 0 <= s->buffered_bits < n <= 64 */
      s->buffer = stream_read_word(s);
      value += (uint64)s->buffer << s->buffered_bits;
      s->buffered_bits += SWORD_BITS;
    } while (sizeof(s->buffer) < sizeof(value) && s->buffered_bits < n);
    /* assert: 1 <= n <= s->buffered_bits < n + wsize */
    s->buffered_bits -= n;
    if (!s->buffered_bits) {
      /* value holds exactly n bits; no need for masking */
      s->buffer = 0;
    } else {
      /* assert: 1 <= s->buffered_bits < wsize */
      s->buffer >>= SWORD_BITS - s->buffered_bits;
      /* assert: 1 <= n <= 64 */
      value &= ((uint64)2 << (n - 1)) - 1;
    }
  } else {
    /* assert: 0 <= n <= s->buffered_bits < wsize <= 64 */
    s->buffered_bits -= n;
    s->buffer >>= n;
    value &= ((uint64)1 << n) - 1;
  }
  return value;
}

/* write 0 <= n <= 64 low bits of value and return remaining bits */
uint64 stream_write_bits(stream* s, uint64 value, size_t n)
{
  /* append bit string to buffer */
  //? Should be prepending (not appending) the `value` to the buffered bits.
  //* The `value` is shifted left by the number of buffered bits.
  //* For example, if the buffer is 0b0101 and the value is 0b11, then the buffer becomes 0b110101.
  s->buffer += (stream_word)(value << s->buffered_bits);
  /* assert: 0 <= s->buffered_bits < wsize (the number of bits written in the buffer) */
  s->buffered_bits += n;
  /* is buffer full? */
  if (s->buffered_bits >= SWORD_BITS) {
    /* 1 <= n <= 64; decrement n to ensure valid right shifts below */
    value >>= 1;
    n--;
    /* assert: 0 <= n < 64; wsize <= s->buffered_bits <= wsize + n */
    do {
      /* output wsize bits while buffer is full */
      s->buffered_bits -= SWORD_BITS;
      //* Only write back the buffered bits to the stream when the buffer is full (> SWORD_BITS).
      //* I.e., writing one word at a time.
      /* assert: 0 <= s->buffered_bits <= n */
      stream_write_word(s, s->buffer);
      /* assert: 0 <= n - s->buffered_bits < 64 */
      s->buffer = (stream_word)(value >> (n - s->buffered_bits));
    } while (sizeof(s->buffer) < sizeof(value) && s->buffered_bits >= SWORD_BITS);
  }
  /* assert: 0 <= s->buffered_bits < wsize */
  s->buffer &= ((stream_word)1 << s->buffered_bits) - 1;
  /* assert: 0 <= n < 64 */
  return value >> n;
}

/* Read a single bit */
uint stream_read_bit(stream *s)
{
  uint bit;
  if (!s->buffered_bits) {
    s->buffer = stream_read_word(s);
    s->buffered_bits = SWORD_BITS;
  }
  s->buffered_bits--;
  bit = (uint)s->buffer & 1u;
  s->buffer >>= 1;
  return bit;
}

/* Write single bit (must be 0 or 1) */
uint stream_write_bit(stream* s, uint bit)
{
  s->buffer += (stream_word)bit << s->buffered_bits;
  if (++s->buffered_bits == SWORD_BITS) {
    stream_write_word(s, s->buffer);
    s->buffer = 0;
    s->buffered_bits = 0;
  }
  return bit;
}

/* Append n zero-bits to stream (n >= 0) */
void stream_pad(stream* s, uint64 n)
{
  uint64 bits = s->buffered_bits;
  for (bits += n; bits >= SWORD_BITS; bits -= SWORD_BITS) {
    stream_write_word(s, s->buffer);
    s->buffer = 0;
  }
  s->buffered_bits = (size_t)bits;
}

/* Write any remaining buffered bits and align stream on next word boundary */
size_t stream_flush(stream* s)
{
  size_t bits = (SWORD_BITS - s->buffered_bits) % SWORD_BITS;
  if (bits)
    stream_pad(s, bits);
  return bits;
}

/* Return bit offset to next bit to be written */
uint64 stream_woffset(stream* s)
{
  return (s->idx) * SWORD_BITS + s->buffered_bits;
}

/* Return bit offset to next bit to be read */
uint64 stream_roffset(stream* s)
{
  return (s->idx) * SWORD_BITS - s->buffered_bits;
}

/* Position stream for reading or writing at beginning */
void stream_rewind(stream *s)
{
  s->idx = 0;
  s->buffer = 0;
  s->buffered_bits = 0;
}

stream *stream_init(void *buffer, size_t bytes)
{
  stream *s = (stream*)malloc(sizeof(stream));
  if (s) {
    s->begin = (stream_word*)buffer;
    s->end = bytes / sizeof(stream_word);
    stream_rewind(s);
  }
  return s;
}

size_t stream_capacity_bytes(const stream *s)
{
  return (size_t)(s->end) * sizeof(stream_word);
}

size_t stream_size_bytes(const stream *s)
{
  return (size_t)(s->idx) * sizeof(stream_word);
}

/* Position stream for reading at given bit offset */
void stream_rseek(stream* s, uint64 offset)
{
  //* Compute the number of bits into the next word the stream should be positioned.
  size_t n = (size_t)(offset % SWORD_BITS);
  s->idx = (size_t)(offset / SWORD_BITS);
  if (n) {
    //* The stream needs to be positioned within the word.
    //* Buffer the remaining bits for subsequent read operations
    s->buffer = stream_read_word(s) >> n;
    s->buffered_bits = SWORD_BITS - n;
  } else {
    //* The desired offset is at the beginning of a word boundary.
    s->buffer = 0;
    s->buffered_bits = 0;
  }
}

/* Skip over the next n bits (n >= 0) */
void stream_skip(stream *s, uint64 n)
{
  stream_rseek(s, stream_roffset(s) + n);
}

/* Align stream on next word boundary */
size_t stream_algin_next_word(stream *s)
{
  size_t  bits = s->buffered_bits;
  if (bits)
    stream_skip(s, bits);
  return bits;
}