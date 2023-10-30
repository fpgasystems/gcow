#include "types.h"
#include "stream.h"


/* Read a single word from memory */
word stream_read_word(stream* s)
{
  word w = *s->ptr++;
// #ifdef BIT_STREAM_STRIDED
//   if (!((s->ptr - s->begin) & s->mask))
//     s->ptr += s->delta;
// #endif
  return w;
}

/* Write a single word to memory */
void stream_write_word(stream* s, word value)
{
  *s->ptr++ = value;
// #ifdef BIT_STREAM_STRIDED
//   if (!((s->ptr - s->begin) & s->mask))
//     s->ptr += s->delta;
// #endif
}

/* read 0 <= n <= 64 bits */
inline uint64 stream_read_bits(stream* s, size_t n)
{
  uint64 value = s->buffer;
  if (s->buffered_bits < n) {
    /* keep fetching wsize bits until enough bits are buffered */
    do {
      /* assert: 0 <= s->buffered_bits < n <= 64 */
      s->buffer = stream_read_word(s);
      value += (uint64)s->buffer << s->buffered_bits;
      s->buffered_bits += WORD_BITS;
    } while (sizeof(s->buffer) < sizeof(value) && s->buffered_bits < n);
    /* assert: 1 <= n <= s->buffered_bits < n + wsize */
    s->buffered_bits -= n;
    if (!s->buffered_bits) {
      /* value holds exactly n bits; no need for masking */
      s->buffer = 0;
    } else {
      /* assert: 1 <= s->buffered_bits < wsize */
      s->buffer >>= WORD_BITS - s->buffered_bits;
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
  //? I think this is prepending the `value` to the buffered bits.
  //* The `value` is shifted left by the number of buffered bits.
  //* For example, if the buffer is 0b0101 and the value is 0b11, then the buffer becomes 0b110101.
  s->buffer += (word)(value << s->buffered_bits);
  /* assert: 0 <= s->buffered_bits < wsize (the number of bits written in the buffer) */
  s->buffered_bits += n;
  /* is buffer full? */
  if (s->buffered_bits >= WORD_BITS) {
    /* 1 <= n <= 64; decrement n to ensure valid right shifts below */
    value >>= 1;
    n--;
    /* assert: 0 <= n < 64; wsize <= s->buffered_bits <= wsize + n */
    do {
      /* output wsize bits while buffer is full */
      s->buffered_bits -= WORD_BITS;
      //* Only write back the buffered bits to the stream when the buffer is full (> WORD_BITS).
      //* I.e., writing one word at a time.
      /* assert: 0 <= s->buffered_bits <= n */
      stream_write_word(s, s->buffer);
      /* assert: 0 <= n - s->buffered_bits < 64 */
      s->buffer = (word)(value >> (n - s->buffered_bits));
    } while (sizeof(s->buffer) < sizeof(value) && s->buffered_bits >= WORD_BITS);
  }
  /* assert: 0 <= s->buffered_bits < wsize */
  s->buffer &= ((word)1 << s->buffered_bits) - 1;
  /* assert: 0 <= n < 64 */
  return value >> n;
}

/* Write single bit (must be 0 or 1) */
uint stream_write_bit(stream* s, uint bit)
{
  s->buffer += (word)bit << s->buffered_bits;
  if (++s->buffered_bits == WORD_BITS) {
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
  for (bits += n; bits >= WORD_BITS; bits -= WORD_BITS) {
    stream_write_word(s, s->buffer);
    s->buffer = 0;
  }
  s->buffered_bits = (size_t)bits;
}

/* Return bit offset to next bit to be written */
uint64 stream_woffset(stream* s)
{
  return (s->ptr - s->begin) * WORD_BITS + s->buffered_bits;
}

size_t stream_capacity_bytes(const stream* s)
{
  return (size_t)(s->end - s->begin) * sizeof(word);
}