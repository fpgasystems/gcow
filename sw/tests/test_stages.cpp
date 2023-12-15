#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <iostream>

#include "gtest/gtest.h"

#include "encode.h"
#include "stream.h"


void transpose_bitplanes(stream *const s, const uint32 *const ublock,
                         uint maxprec, uint block_size)
{
  uint64 offset = stream_woffset(s);
  uint intprec = (uint)(CHAR_BIT * sizeof(uint32));
  uint kmin = intprec > maxprec ? intprec - maxprec : 0;
  uint i, k, n;

  /* encode one bit plane at a time from MSB to LSB */
  for (k = intprec, n = 0; k-- > kmin;) {
    //^ Step 1: extract bit plane #k to x.
    uint64 x = 0;
    for (i = 0; i < block_size; i++) {
      x += (uint64)((ublock[i] >> k) & 1u) << i;
    }

    printf("%lu, ", x);

    stream_write_bits(s, x, block_size);
  }
  printf("\n");
}

uint encode_bitplanes(stream *const s, const uint32 *const ublock,
                      uint maxprec, uint block_size)
{
  /* make a copy of bit stream to avoid aliasing */
  // stream s = *out_data;
  uint64 offset = stream_woffset(s);
  uint intprec = (uint)(CHAR_BIT * sizeof(uint32));
  uint kmin = intprec > maxprec ? intprec - maxprec : 0;
  uint k = intprec;
  uint n = 0;

  /* encode one bit plane at a time from MSB to LSB */
  while (k-- > kmin) {
    // for (k = intprec, n = 0; k-- > kmin;) {
    //^ Step 1: extract bit plane #k to x.
    uint64 x = 0;
    for (uint i = 0; i < block_size; i++) {
      x += (uint64)((ublock[i] >> k) & 1u) << i;
    }

    // //^ Step 2: encode first n bits of bit plane.
    // x = stream_write_bits(s, x, n);

    //^ Step 3: unary run-length encode remainder of bit plane.
    for (; n < block_size; x >>= 1, n++) {
      if (!stream_write_bit(s, !!x)) {
        //^ Negative group test (x == 0) -> Done with all bit planes.
        break;
      }
      for (; n < block_size - 1; x >>= 1, n++) {
        //* Continue writing 0's until a 1 bit is found.
        //& `x & 1u` is used to extract the least significant (right-most) bit of `x`.
        if (stream_write_bit(s, x & 1u)) {
          //* After writing a 1 bit, break out for another group test
          //* (to see whether the bitplane code `x` turns 0 after encoding `n` of its bits).
          break;
        }
      }
    }
  }

  // *out_data = s;
  //* Returns the number of bits written.
  return (uint)(stream_woffset(s) - offset);
}

uint embedded_encoding(stream *const s, const uint32 *const ublock,
                       uint maxprec, uint block_size)
{
  uint total = 17;
  uint64 inputs[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 9, 15, 6920, 6918};
  // uint64 inputs[] = {
  //   2002158, 6736799, 2736739, 462686,
  //   28905, 28910, 90992, 28371,
  //   116400, 116490, 388, 514420,
  //   114423, 1375, 1195, 6066,
  //   1992158, 1736739, 1736739, 462686,
  //   28905, 28910, 28371, 28371,
  //   116490, 116490, 288, 114420,
  //   114423, 1175, 1175, 5095
  // };
  uint64 offset = stream_woffset(s);
  uint n = 0;
  uint i = 0;

  /* encode one bit plane at a time from MSB to LSB */
  while (i < total) {
    uint64 x = inputs[i++];
    stream_write_bits(s, x, n);
    //^ Step 3: unary run-length encode remainder of bit plane.
    for (; n < block_size; x >>= 1, n++) {
      if (!stream_write_bit(s, !!x)) {
        //^ Negative group test (x == 0) -> Done with all bit planes.
        break;
      }
      for (; n < block_size - 1; x >>= 1, n++) {
        //* Continue writing 0's until a 1 bit is found.
        //& `x & 1u` is used to extract the least significant (right-most) bit of `x`.
        if (stream_write_bit(s, x & 1u)) {
          //* After writing a 1 bit, break out for another group test
          //* (to see whether the bitplane code `x` turns 0 after encoding `n` of its bits).
          break;
        }
      }
    }
  }

  // *out_data = s;
  //* Returns the number of bits written.
  return (uint)(stream_woffset(s) - offset);
}


TEST(STAGES, GATHER_2D)
{
  uint dim = 2;
  size_t nx = 5;
  size_t ny = 7;
  float raw[ny][nx];
  //* Initialize array to be compressed from 1 to nx*ny:
  for (size_t y = 0; y < ny; y++)
    for (size_t x = 0; x < nx; x++)
      raw[y][x] = (float)(x + nx * y + 1);

  size_t block_size = BLOCK_SIZE(dim);
  const float* data = (const float*)raw;
  ptrdiff_t sx = 1;
  ptrdiff_t sy = (ptrdiff_t)nx;

  uint num_blocks = ceil((float)nx / 4) * ceil((float)ny / 4);
  printf("num_blocks: %u\n", num_blocks);

  int expected[num_blocks*BLOCK_SIZE_2D] = {
    1, 2, 3, 4,
    6, 7, 8, 9,
    11, 12, 13, 14,
    16, 17, 18, 19,

    5, 5, 5, 5,
    10, 10, 10, 10,
    15, 15, 15, 15,
    20, 20, 20, 20,

    21, 22, 23, 24,
    26, 27, 28, 29,
    31, 32, 33, 34,
    21, 22, 23, 24,

    25, 25, 25, 25,
    30, 30, 30, 30,
    35, 35, 35, 35,
    25, 25, 25, 25,
  };

  size_t i = 0;
  int b = 1;
  //* Compress array one block of 4x4 values at a time
  for (size_t y = 0; y < ny; y += 4) {
    // printf("Encoding block [%ld, *]\n", y);
    for (size_t x = 0; x < nx; x += 4) {
      const float *raw = data + sx * (ptrdiff_t)x + sy * (ptrdiff_t)y;
      float fblock[block_size];

      if (nx - x < 4 || ny - y < 4) {
        gather_partial_2d_block(fblock, raw, MIN(nx - x, 4u), MIN(ny - y, 4u), sx, sy);
      } else {
        gather_2d_block(fblock, raw, sx, sy);
      }

      //* Verify that the block was gathered correctly.
      for (size_t j = 0; j < block_size; j++) {
        EXPECT_EQ((int)fblock[j], expected[i++]);
      }

      //* Print the fblock with newlines for each row.
      printf("fblock[%d]: \n", b++);
      for (size_t k = 0; k < block_size; k++) {
        printf("%d, ", (int)fblock[k]);
        if ((k + 1) % 4 == 0) {
          printf("\n");
        }
      }
    }
  }
}

TEST(STAGES, EMAX)
{
  float raw[BLOCK_SIZE_2D]; //* 4x4 source array
  size_t nx = 4;
  size_t ny = 4;

  for (size_t j = 0; j < ny; j++)
    for (size_t i = 0; i < nx; i++)
      raw[i + nx * j] = (float)(i + 4 * j + 1);

  EXPECT_EQ(get_block_exponent((const float*)raw, BLOCK_SIZE_2D), 5);

  for (size_t j = 0; j < ny; j++)
    for (size_t i = 0; i < nx; i++) {
      double x = 2.0 * i / nx;
      double y = 2.0 * j / ny;
      raw[i + nx * j] = (float)exp(-(x * x + y *
                                     y)); // (float)(x + 100 * y + 3.1415926);
    }

  EXPECT_EQ(get_block_exponent((const float*)raw, BLOCK_SIZE_2D), 1);
}

TEST(STAGES, CAST)
{
  float fblock[BLOCK_SIZE_2D]; //* 4x4 source array
  int32 iblock[BLOCK_SIZE_2D];

  size_t nx = 4;
  size_t ny = 4;

  for (size_t j = 0; j < ny; j++)
    for (size_t i = 0; i < nx; i++) {
      double x = 2.0 * i / nx;
      double y = 2.0 * j / ny;
      fblock[i + nx * j] = (float)exp(-(x * x + y *
                                        y)); // (float)(x + 100 * y + 3.1415926);
    }

  fwd_cast_block(iblock, (const float*)fblock, BLOCK_SIZE_2D, 9);
  for (int i = 0; i < BLOCK_SIZE_2D; i++) {
    printf("%d, ", iblock[i]);
  }
  printf("\n");

  int32 expected[BLOCK_SIZE_2D] = {
    2097152, 1633263, 771499, 221038,
    1633263, 1271987, 600844, 172144,
    771499, 600844, 283818, 81315,
    221038, 172144, 81315, 23297
  };

  for (int i = 0; i < BLOCK_SIZE_2D; i++) {
    EXPECT_EQ(iblock[i], expected[i]);
  }
}

TEST(STAGES, DECORRELATE)
{
  int32 iblock[BLOCK_SIZE_2D] = {
    2097152, 1633263, 771499, 221038,
    1633263, 1271987, 600844, 172144,
    771499, 600844, 283818, 81315,
    221038, 172144, 81315, 23297
  };

  fwd_decorrelate_2d_block(iblock);

  int32 expected[BLOCK_SIZE_2D] = {
    664778, 360415, 12185, 49910,
    360415, 195402, 6607, 27060,
    12186, 6607, 224, 915,
    49910, 27059, 915, 3747
  };

  for (int i = 0; i < BLOCK_SIZE_2D; i++) {
    printf("%d, ", iblock[i]);
  }
  printf("\n");

  for (int i = 0; i < BLOCK_SIZE_2D; i++) {
    EXPECT_EQ(iblock[i], expected[i]);
  }
}

TEST(STAGES, REORDER)
{
  int32 iblock[BLOCK_SIZE_2D] = {
    664778, 360415, 12185, 49910,
    360415, 195402, 6607, 27060,
    12186, 6607, 224, 915,
    49910, 27059, 915, 3747
  };

  // int32 iblock[BLOCK_SIZE_2D] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
  //                                11, 12, 13, 14, 15, 16};

  uint32 ublock[BLOCK_SIZE_2D];

  fwd_reorder_int2uint(ublock, iblock, PERM_2D, BLOCK_SIZE_2D);

  uint32 expected[BLOCK_SIZE_2D] = {
    1992158, 1736739, 1736739, 462686,
    28905, 28910, 28371, 28371,
    116490, 116490, 288, 114420,
    114423, 1175, 1175, 5095
  };
  // uint32 expected[BLOCK_SIZE_2D] = {1, 6, 5, 26, 7, 25, 27, 30, 4, 29, 31, 24, 18, 28, 19, 16};

  for (int i = 0; i < BLOCK_SIZE_2D; i++) {
    printf("%u, ", ublock[i]);
  }
  printf("\n");

  for (int i = 0; i < BLOCK_SIZE_2D; i++) {
    EXPECT_EQ(ublock[i], expected[i]);
  }
}

TEST(STAGES, STREAM_WRITE)
{
  int e = 1;
  uint bits = 1 + EBITS;
  size_t output_bytes = 1000;
  void *buffer = malloc(output_bytes);
  stream *s = stream_init(buffer, output_bytes);

  // stream_write_bits(s, 2 * e + 1, bits);
  stream_write_bits(s, 7455816852505100291UL, 64);
  // EXPECT_EQ(s->begin[0], 0U);
  // EXPECT_EQ(s->buffer, 3UL);
  // EXPECT_EQ(stream_size_bytes(s), 0U);

  printf("stream: ");
  for (int i = 0; i < s->idx; i++) {
    printf("%lu, ", s->begin[i]);
  }
  printf("\nstream.buffer: %lu\n", s->buffer);
  printf("stream.idx: %ld\n", s->idx);
  printf("stream_size_bytes: %lu\n\n", stream_size_bytes(s));

  stream_write_bits(s, 432UL, 9);

  printf("stream: ");
  for (int i = 0; i < s->idx; i++) {
    printf("%lu, ", s->begin[i]);
  }
  printf("\nstream.buffer: %lu\n", s->buffer);
  printf("stream.idx: %ld\n", s->idx);
  printf("stream_size_bytes: %lu\n\n", stream_size_bytes(s));


  stream_flush(s);
  printf("Flushed stream.\n\n");
  // EXPECT_EQ(s->begin[0], 3UL);
  // EXPECT_EQ(s->buffer, 0U);
  // EXPECT_EQ(stream_size_bytes(s), sizeof(uint64));

  printf("stream: ");
  for (int i = 0; i < s->idx; i++) {
    printf("%lu, ", s->begin[i]);
  }
  printf("\nstream.buffer: %lu\n", s->buffer);
  printf("stream.idx: %ld\n", s->idx);
  printf("stream_size_bytes: %lu\n\n", stream_size_bytes(s));
}

TEST(STAGES, ENCODE_ALL_BITPLANES)
{
  int emax = 1;
  uint bits = 1 + EBITS;
  size_t output_bytes = 1000; //* Does not matter for this test, just a bound.
  void *buffer = malloc(output_bytes);
  stream *s = stream_init(buffer, output_bytes);

  stream_write_bits(s, 2 * emax + 1, bits);

  uint32 ublock[BLOCK_SIZE_2D] = {
    1992158, 1736739, 1736739, 462686,
    28905, 28910, 28371, 28371,
    116490, 116490, 288, 114420,
    114423, 1175, 1175, 5095
  };

  //  uint32 ublock[BLOCK_SIZE_2D] = {
  //   2002158, 6736799, 2736739, 462686,
  //   28905, 28910, 90992, 28371,
  //   116400, 116490, 388, 514420,
  //   114423, 1375, 1195, 6066
  // };

  // uint32 ublock[BLOCK_SIZE_2D] = {
  //   1992158, 1736739, 1736739, 462686,
  //   38905, 28910, 28371, 28371,
  //   116490, 116490, 288, 114420,
  //   114423, 1175, 1175, 5095
  // };

  zfp_output *output = alloc_zfp_output();
  double tolerance = 1e-3;
  double max_error = set_zfp_output_accuracy(output, tolerance);
  printf("Maximum error:\t\t%f\n", max_error);

  uint maxprec = get_precision(emax, output->maxprec, output->minexp, 2);
  printf("Max precision:\t\t%u\n", maxprec);
  bool exceeded = exceeded_maxbits(output->maxbits, maxprec, BLOCK_SIZE_2D);
  printf("Maxbits:\t\t%u\n", output->maxbits);
  printf("Exceeded maxbits:\t%s\n", exceeded ? "true" : "false");
  EXPECT_FALSE(exceeded);


  //* Only test encoding without bit limits.
  uint encoded_bits = encode_all_bitplanes(s, ublock, maxprec, BLOCK_SIZE_2D);
  printf("Encoded bits:\t\t%u\n\n", encoded_bits);

  uint64 expected[2] = {
    7455816852505100291UL,
    432UL
  };

  // uint64 expected[2] = {
  //   2318511421321904131,
  //   113368486UL
  // };

  // uint64 expected[2] = {
  //   7455816852505100291UL,
  //   433UL
  // };

  printf("stream.idx: %ld\n", s->idx);
  printf("stream: ");
  for (int i = 0; i < s->idx; i++) {
    printf("%lu, ", s->begin[i]);
  }
  printf("\nstream.buffer: %lu\n", s->buffer);
  printf("stream_size_bytes: %lu\n\n", stream_size_bytes(s));

  EXPECT_EQ(s->begin[0], expected[0]);
  EXPECT_EQ(s->buffer, expected[1]);
  EXPECT_EQ(stream_size_bytes(s), sizeof(uint64));

  stream_flush(s);

  EXPECT_EQ(s->begin[1], expected[1]);
  EXPECT_EQ(s->buffer, 0U);
  EXPECT_EQ(stream_size_bytes(s), 2*sizeof(uint64));

  printf("stream.idx: %ld\n", s->idx);
  printf("stream: ");
  for (int i = 0; i < s->idx; i++) {
    printf("%lu, ", s->begin[i]);
  }
  printf("\nstream.buffer: %lu\n", s->buffer);
  printf("stream_size_bytes: %lu\n", stream_size_bytes(s));
}

TEST(STAGES, ENCODE_BITPLANES)
{
  int emax = 1;
  uint bits = 1 + EBITS;
  size_t output_bytes = 1000; //* Does not matter for this test, just a bound.
  void *buffer = malloc(output_bytes);
  stream *s = stream_init(buffer, output_bytes);

  stream_write_bits(s, 2 * emax + 1, bits);

  // uint32 ublock[BLOCK_SIZE_2D] = {
  //   1992158, 1736739, 1736739, 462686,
  //   28905, 28910, 28371, 28371,
  //   116490, 116490, 288, 114420,
  //   114423, 1175, 1175, 5095
  // };

  //  uint32 ublock[BLOCK_SIZE_2D] = {
  //   2002158, 6736799, 2736739, 462686,
  //   28905, 28910, 90992, 28371,
  //   116400, 116490, 388, 514420,
  //   114423, 1375, 1195, 6066
  // };

  uint32 ublock[BLOCK_SIZE_2D] = {
    4294967236, 329467215, 4294967214, 1104967293,
    1294967212, 4294967281, 4294967240, 3294967209,
    4294967208, 4294967277, 22967206, 4294967205,
    1294967234, 494967203, 4294967202, 294967251
  };

  zfp_output *output = alloc_zfp_output();
  double tolerance = 1e-3;
  double max_error = set_zfp_output_accuracy(output, tolerance);
  printf("Maximum error:\t\t%f\n", max_error);

  uint maxprec = get_precision(emax, output->maxprec, output->minexp, 2);
  printf("Max precision:\t\t%u\n", maxprec);
  bool exceeded = exceeded_maxbits(output->maxbits, maxprec, BLOCK_SIZE_2D);
  printf("Maxbits:\t\t%u\n", output->maxbits);
  printf("Exceeded maxbits:\t%s\n", exceeded ? "true" : "false");
  EXPECT_FALSE(exceeded);


  //* Only test encoding without bit limits.
  uint encoded_bits = encode_bitplanes(s, ublock, maxprec, BLOCK_SIZE_2D);
  printf("Encoded bits:\t\t%u\n\n", encoded_bits);

  uint64 expected[2] = {
    7455816852505100291UL,
    432UL
  };

  // uint64 expected[2] = {
  //   2318511421321904131,
  //   113368486UL
  // };

  // uint64 expected[2] = {
  //   7455816852505100291UL,
  //   433UL
  // };

  printf("stream.idx: %ld\n", s->idx);
  printf("stream: ");
  for (int i = 0; i < s->idx; i++) {
    printf("%lu, ", s->begin[i]);
  }
  printf("\nstream.buffer: %lu\n", s->buffer);
  printf("stream_size_bytes: %lu\n\n", stream_size_bytes(s));

  // EXPECT_EQ(s->begin[0], expected[0]);
  // EXPECT_EQ(s->buffer, expected[1]);
  // EXPECT_EQ(stream_size_bytes(s), sizeof(uint64));

  stream_flush(s);

  // EXPECT_EQ(s->begin[1], expected[1]);
  // EXPECT_EQ(s->buffer, 0U);
  // EXPECT_EQ(stream_size_bytes(s), 2*sizeof(uint64));

  printf("stream.idx: %ld\n", s->idx);
  printf("stream: ");
  for (int i = 0; i < s->idx; i++) {
    printf("%lu, ", s->begin[i]);
  }
  printf("\nstream.buffer: %lu\n", s->buffer);
  printf("stream_size_bytes: %lu\n", stream_size_bytes(s));
}

TEST(STAGES, TRANSPOSE)
{
  int emax = 1;
  uint bits = 1 + EBITS;
  size_t output_bytes = 1000; //* Does not matter for this test, just a bound.
  void *buffer = malloc(output_bytes);
  stream *s = stream_init(buffer, output_bytes);

  stream_write_bits(s, 2 * emax + 1, bits);

  uint32 ublock[BLOCK_SIZE_2D] = {
    1992158, 1736739, 1736739, 462686,
    28905, 28910, 28371, 28371,
    116490, 116490, 288, 114420,
    114423, 1175, 1175, 5095
  };

  //  uint32 ublock[BLOCK_SIZE_2D] = {
  //   2002158, 6736799, 2736739, 462686,
  //   28905, 28910, 90992, 28371,
  //   116400, 116490, 388, 514420,
  //   114423, 1375, 1195, 6066
  // };

  zfp_output *output = alloc_zfp_output();
  double tolerance = 1e-3;
  double max_error = set_zfp_output_accuracy(output, tolerance);
  printf("Maximum error:\t\t%f\n", max_error);

  uint maxprec = get_precision(emax, output->maxprec, output->minexp, 2);
  printf("Max precision:\t\t%u\n", maxprec);
  bool exceeded = exceeded_maxbits(output->maxbits, maxprec, BLOCK_SIZE_2D);
  printf("Maxbits:\t\t%u\n", output->maxbits);
  printf("Exceeded maxbits:\t%s\n", exceeded ? "true" : "false");
  EXPECT_FALSE(exceeded);


  //* Only test encoding without bit limits.
  transpose_bitplanes(s, ublock, maxprec, BLOCK_SIZE_2D);

  uint64 expected[5] = {
    3, 0, 1008806316530991104, 1152954490257673728, 3542070
    // 3, 0, 144128382282498048, 10957282151736805888, 3542582,
  };

  printf("stream.idx: %ld\n", s->idx);
  printf("stream: ");
  for (int i = 0; i < s->idx; i++) {
    printf("%lu, ", s->begin[i]);
  }
  printf("\nstream.buffer: %lu\n", s->buffer);
  printf("stream_size_bytes: %lu\n\n", stream_size_bytes(s));

  // EXPECT_EQ(s->begin[0], expected[0]);
  // EXPECT_EQ(s->buffer, expected[1]);
  // EXPECT_EQ(stream_size_bytes(s), sizeof(uint64));

  stream_flush(s);

  // EXPECT_EQ(s->begin[1], expected[1]);
  // EXPECT_EQ(s->buffer, 0U);
  // EXPECT_EQ(stream_size_bytes(s), 2*sizeof(uint64));

  printf("stream.idx: %ld\n", s->idx);
  printf("stream: ");
  for (int i = 0; i < s->idx; i++) {
    printf("%lu, ", s->begin[i]);
  }
  printf("\nstream.buffer: %lu\n", s->buffer);
  printf("stream_size_bytes: %lu\n", stream_size_bytes(s));
}

TEST(STAGES, EMBEDDED_ENCODING)
{
  int emax = 1;
  uint bits = 1 + EBITS;
  size_t output_bytes = 1000; //* Does not matter for this test, just a bound.
  void *buffer = malloc(output_bytes);
  stream *s = stream_init(buffer, output_bytes);

  stream_write_bits(s, 2 * emax + 1, bits);

  // uint32 ublock[BLOCK_SIZE_2D] = {
  //   1992158, 1736739, 1736739, 462686,
  //   28905, 28910, 28371, 28371,
  //   116490, 116490, 288, 114420,
  //   114423, 1175, 1175, 5095
  // };

  uint32 ublock[BLOCK_SIZE_2D] = {
    2002158, 6736799, 2736739, 462686,
    28905, 28910, 90992, 28371,
    116400, 116490, 388, 514420,
    114423, 1375, 1195, 6066
  };

  zfp_output *output = alloc_zfp_output();
  double tolerance = 1e-3;
  double max_error = set_zfp_output_accuracy(output, tolerance);
  printf("Maximum error:\t\t%f\n", max_error);

  uint maxprec = get_precision(emax, output->maxprec, output->minexp, 2);
  printf("Max precision:\t\t%u\n", maxprec);
  bool exceeded = exceeded_maxbits(output->maxbits, maxprec, BLOCK_SIZE_2D);
  printf("Maxbits:\t\t%u\n", output->maxbits);
  printf("Exceeded maxbits:\t%s\n", exceeded ? "true" : "false");
  EXPECT_FALSE(exceeded);


  //* Only test encoding without bit limits.
  uint encoded_bits = embedded_encoding(s, ublock, maxprec, BLOCK_SIZE_2D);
  printf("Encoded bits:\t\t%u\n\n", encoded_bits);

  uint64 expected[] = {
    // 3, 0, 1008806316530991104, 1152954490257673728, 3542070
    // 3, 0, 144128382282498048, 10957282151736805888, 3542582,
    17311055420935110659, 1901733698298127, 979066
  };

  printf("stream.idx: %ld\n", s->idx);
  printf("stream: ");
  for (int i = 0; i < s->idx; i++) {
    printf("%lu, ", s->begin[i]);
  }
  printf("\nstream.buffer: %lu\n", s->buffer);
  printf("stream_size_bytes: %lu\n\n", stream_size_bytes(s));

  // EXPECT_EQ(s->begin[0], expected[0]);
  // EXPECT_EQ(s->buffer, expected[1]);
  // EXPECT_EQ(stream_size_bytes(s), sizeof(uint64));

  stream_flush(s);

  // EXPECT_EQ(s->begin[1], expected[1]);
  // EXPECT_EQ(s->buffer, 0U);
  // EXPECT_EQ(stream_size_bytes(s), 2*sizeof(uint64));

  printf("stream.idx: %ld\n", s->idx);
  printf("stream: ");
  for (int i = 0; i < s->idx; i++) {
    printf("%lu, ", s->begin[i]);
  }
  printf("\nstream.buffer: %lu\n", s->buffer);
  printf("stream_size_bytes: %lu\n", stream_size_bytes(s));
}

TEST(STAGES, ENCODE_IBLOCK)
{
  int32 iblock[BLOCK_SIZE_2D] = {
    6588397, 8685549, 10782701, 12879853,
    216303600, 218400752, 220497904, 222595056,
    426018784, 428115936, 430213088, 432310240,
    635734016, 637831168, 639928320, 642025472
  };

  int e = 9;
  uint bits = 1 + EBITS;
  size_t output_bytes = 1000; //* Does not matter for this test, just a bound.
  void *buffer = malloc(output_bytes);
  stream *s = stream_init(buffer, output_bytes);

  stream_write_bits(s, 2 * e + 1, bits);

  zfp_output *output = alloc_zfp_output();
  size_t dim = 2;
  double tolerance = 1e-3;
  double max_error = set_zfp_output_accuracy(output, tolerance);
  uint maxprec = get_precision(e, output->maxprec, output->minexp, dim);
  printf("Maximum error:\t\t%f\n", max_error);
  printf("Max precision:\t\t%u\n", maxprec);

  uint encoded_bits = encode_iblock(
                        s, output->minbits, output->maxbits, maxprec, iblock, dim);
  printf("Encoded bits:\t\t%u\n", encoded_bits);

  uint64 expected[2] = {
    72375632423897107UL,
    1114129UL,
  };
  printf("stream.idx: %ld\n", s->idx);
  printf("stream: ");
  for (int i = 0; i < s->idx; i++) {
    printf("%lu, ", s->begin[i]);
  }
  printf("\nstream.buffer: %lu\n", s->buffer);
  printf("stream_size_bytes: %lu\n", stream_size_bytes(s));

  EXPECT_EQ(s->begin[0], expected[0]);
  EXPECT_EQ(s->buffer, expected[1]);
  EXPECT_EQ(stream_size_bytes(s), sizeof(uint64));

  stream_flush(s);

  EXPECT_EQ(s->begin[1], expected[1]);
  EXPECT_EQ(s->buffer, 0U);
  EXPECT_EQ(stream_size_bytes(s), 2*sizeof(uint64));

  printf("stream.idx: %ld\n", s->idx);
  printf("stream: ");
  for (int i = 0; i < s->idx; i++) {
    printf("%lu, ", s->begin[i]);
  }
  printf("\nstream.buffer: %lu\n", s->buffer);
  printf("stream_size_bytes: %lu\n", stream_size_bytes(s));
}

int main(int argc, char** argv)
{
  printf("\nStages Tests: \n");
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}