#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <iostream>

#include "gtest/gtest.h"

#include "encode.h"
#include "stream.h"


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
  float raw[4][4]; //* 4x4 source array

  for (int y = 0; y < 4; y++)
    for (int x = 0; x < 4; x++)
      raw[y][x] = (float)(x + 4 * y + 1);

  EXPECT_EQ(get_block_exponent((const float*)raw, BLOCK_SIZE_2D), 5);

  for (int y = 0; y < 4; y++)
    for (int x = 0; x < 4; x++)
      raw[y][x] = (float)(x + 100 * y + 3.1415926);

  EXPECT_EQ(get_block_exponent((const float*)raw, BLOCK_SIZE_2D), 9);
}

TEST(STAGES, CAST)
{
  float fblock[4][4]; //* 4x4 source array
  int32 iblock[BLOCK_SIZE_2D];

  for (int y = 0; y < 4; y++)
    for (int x = 0; x < 4; x++)
      fblock[y][x] = (float)(x + 100 * y + 3.1415926);

  fwd_cast_block(iblock, (const float*)fblock, BLOCK_SIZE_2D, 9);

  int32 expected[BLOCK_SIZE_2D] = {
    6588397, 8685549, 10782701, 12879853,
    216303600, 218400752, 220497904, 222595056,
    426018784, 428115936, 430213088, 432310240,
    635734016, 637831168, 639928320, 642025472
  };

  for (int i = 0; i < BLOCK_SIZE_2D; i++) {
    EXPECT_EQ(iblock[i], expected[i]);
  }
}

TEST(STAGES, DECORRELATE)
{
  int32 iblock[BLOCK_SIZE_2D] = {
    6588397, 8685549, 10782701, 12879853,
    216303600, 218400752, 220497904, 222595056,
    426018784, 428115936, 430213088, 432310240,
    635734016, 637831168, 639928320, 642025472
  };

  fwd_decorrelate_2d_block(iblock);

  int32 expected[BLOCK_SIZE_2D] = {
    324306927, -2097152, 0, 0, -209715205,
    0, 0, 0, -7, 0, 0, 0, 8, 0, 0, 0
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
    324306927, -2097152, 0, 0, -209715205,
    0, 0, 0, -7, 0, 0, 0, 8, 0, 0, 0
  };

  uint32 ublock[BLOCK_SIZE_2D];

  fwd_reorder_int2uint(ublock, iblock, PERM_2D, BLOCK_SIZE_2D);

  uint32 expected[BLOCK_SIZE_2D] = {
    391485491, 2097152, 880803855, 0, 0, 9, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0
  };

  for (int i = 0; i < BLOCK_SIZE_2D; i++) {
    EXPECT_EQ(ublock[i], expected[i]);
  }
}

TEST(STAGES, STREAM_WRITE)
{
  int e = 9;
  uint bits = 1 + EBITS;
  size_t output_bytes = 1000;
  void *buffer = malloc(output_bytes);
  stream *s = stream_init(buffer, output_bytes);

  stream_write_bits(s, 2 * e + 1, bits);
  EXPECT_EQ(s->begin[0], 0U);
  EXPECT_EQ(s->buffer, 19UL);
  EXPECT_EQ(stream_size_bytes(s), 0U);

  printf("stream: ");
  for (int i = 0; i < s->idx; i++) {
    printf("%lu, ", s->begin[i]);
  }
  printf("\nstream.buffer: %lu\n", s->buffer);
  printf("stream_size_bytes: %lu\n", stream_size_bytes(s));

  stream_flush(s);
  EXPECT_EQ(s->begin[0], 19UL);
  EXPECT_EQ(s->buffer, 0U);
  EXPECT_EQ(stream_size_bytes(s), sizeof(uint64));

  printf("stream: ");
  for (int i = 0; i < s->idx; i++) {
    printf("%lu, ", s->begin[i]);
  }
  printf("\nstream.buffer: %lu\n", s->buffer);
  printf("stream_size_bytes: %lu\n", stream_size_bytes(s));
}

TEST(STAGES, ENCODE_ALL_BITPLANES)
{
  int e = 9;
  uint bits = 1 + EBITS;
  size_t output_bytes = 1000; //* Does not matter for this test, just a bound.
  void *buffer = malloc(output_bytes);
  stream *s = stream_init(buffer, output_bytes);

  stream_write_bits(s, 2 * e + 1, bits);

  uint32 ublock[BLOCK_SIZE_2D] = {
    391485491, 2097152, 880803855, 0, 0, 9, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0
  };

  zfp_output *output = alloc_zfp_output();
  double tolerance = 1e-3;
  double max_error = set_zfp_output_accuracy(output, tolerance);
  printf("Maximum error:\t\t%f\n", max_error);

  uint maxprec = get_precision(e, output->maxprec, output->minexp, 2);
  printf("Max precision:\t\t%u\n", maxprec);
  bool exceeded = exceeded_maxbits(output->maxbits, maxprec, BLOCK_SIZE_2D);
  printf("Maxbits:\t\t%u\n", output->maxbits);
  printf("Exceeded maxbits:\t%s\n", exceeded ? "true" : "false");
  EXPECT_FALSE(exceeded);


  //* Only test encoding without bit limits.
  uint encoded_bits = encode_all_bitplanes(s, ublock, maxprec, BLOCK_SIZE_2D);
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