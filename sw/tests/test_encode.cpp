#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <iostream>

#include "gtest/gtest.h"

// extern "C" {
//  // * Avoid name mangling for C functions.
#include "encode.h"
#include "stream.h"
// #include "common.hpp"
// }


void test_encode_2d_block(uint32 *encoded, const float *fblock,
                          const zfp_input *input)
{
  for (int i = 0; i < BLOCK_SIZE_2D; i++)
    encoded[i] = (uint32)fblock[i];
}

void test_encode_4d_block(uint32 *encoded, const float *block,
                          const zfp_input *input)
{
  //TODO: Implement 4d encoding
  for (int i = 0; i < BLOCK_SIZE_4D; i++)
    encoded[i] = (uint32)block[i];
}

void test_encode_strided_2d_block(uint32 *encoded, const float *raw,
                                  ptrdiff_t sx, ptrdiff_t sy, const zfp_input *input)
{
  float block[BLOCK_SIZE_2D];

  //TODO: Cache alignment?
  gather_2d_block(block, raw, sx, sy);
  test_encode_2d_block(encoded, block, input);
}

void test_encode_strided_partial_2d_block(uint32 *encoded, const float *raw,
    size_t nx, size_t ny, ptrdiff_t sx, ptrdiff_t sy,
    const zfp_input *input)
{
  float block[BLOCK_SIZE_2D];

  //TODO: Cache alignment?
  gather_partial_2d_block(block, raw, nx, ny, sx, sy);
  test_encode_2d_block(encoded, block, input);
}

void test_encode_strided_4d_block(uint32 *encoded, const float *raw,
                                  ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw,
                                  const zfp_input *input)
{
  float block[BLOCK_SIZE_4D];

  //TODO: Cache alignment?
  gather_4d_block(block, raw, sx, sy, sz, sw);
  test_encode_4d_block(encoded, block, input);
}

void test_encode_strided_partial_4d_block(uint32 *encoded, const float *raw,
    size_t nx, size_t ny, size_t nz, size_t nw,
    ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw,
    const zfp_input *input)
{
  float block[BLOCK_SIZE_4D];

  //TODO: Cache alignment?
  gather_partial_4d_block(block, raw, nx, ny, nz, nw, sx, sy, sz, sw);
  test_encode_4d_block(encoded, block, input);
}

void test_2d(uint32 *compressed, const zfp_input* input)
{
  const float* data = (const float*)input->data;
  size_t nx = input->nx;
  size_t ny = input->ny;
  ptrdiff_t sx = input->sx ? input->sx : 1;
  ptrdiff_t sy = input->sy ? input->sy : (ptrdiff_t)nx;

  //* Compress array one block of 4x4 values at a time
  size_t iblock = 0;
  for (size_t y = 0; y < ny; y += 4) {
    for (size_t x = 0; x < nx; x += 4, iblock++) {
      //! Writing sequentially to the output stream for testing.
      uint32 *encoded = compressed + BLOCK_SIZE_2D * iblock;
      const float *raw = data + sx * (ptrdiff_t)x + sy * (ptrdiff_t)y;
      printf("Encoding block (%ld, %ld)\n", y, x);
      if (nx - x < 4 || ny - y < 4) {
        test_encode_strided_partial_2d_block(encoded, raw, MIN(nx - x, 4u), MIN(ny - y,
                                             4u),
                                             sx, sy, input);
      } else {
        test_encode_strided_2d_block(encoded, raw, sx, sy, input);
      }
    }
  }
}

/* ------------------ Test Cases ------------------ */

TEST(encode, gather_2d_block)
{
  float raw[4][4]; //* 4x4 source array
  float block[BLOCK_SIZE_2D];

  //* Initialize raw with some data (e.g., sequence from 1 to 16)
  for (int y = 0; y < 4; y++)
    for (int x = 0; x < 4; x++)
      raw[y][x] = (float)(x + 4 * y + 1);

  //* Test gather_2d_block
  gather_2d_block(block, (const float*)raw, 1, 4);

  //* Check correctness of gathered values
  for (int i = 0; i < BLOCK_SIZE_2D; i++) {
    EXPECT_EQ(block[i], (float)(i + 1));
  }
}


//* Test function for gather_4d_block
TEST(encode, gather_4d_block)
{
  float raw[4][4][4][4]; //* 4x4x4x4 source array
  float block[BLOCK_SIZE_4D];

  //* Initialize raw with some data (e.g., sequence from 1 to 256)
  for (int w = 0; w < 4; w++)
    for (int z = 0; z < 4; z++)
      for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
          raw[w][z][y][x] = (float)(x + 4 * y + 16 * z + 64 * w + 1);

  //* Test gather_4d_block
  gather_4d_block(block, (const float*)raw, 1, 4, 16, 64);

  //* Check correctness of gathered values
  for (int i = 0; i < BLOCK_SIZE_4D; i++) {
    EXPECT_EQ(block[i], (float)(i + 1));
  }
}

TEST(encode, test_encode_strided_2d_block)
{
  zfp_input specs;
  float raw[4][4]; //* 4x4 source array
  uint32 encoded[BLOCK_SIZE_2D];

  //* Initialize raw with some data (e.g., sequence from 1 to 16)
  for (int y = 0; y < 4; y++)
    for (int x = 0; x < 4; x++)
      raw[y][x] = (float)(x + 4 * y + 1);

  //* Test encode_strided_2d_block
  test_encode_strided_2d_block(encoded, (const float*)raw, 1, 4, &specs);

  //* Check correctness of encoded values
  for (int i = 0; i < BLOCK_SIZE_2D; i++) {
    EXPECT_EQ(encoded[i], (uint32)(i + 1));
  }
}

//* Test function for test_encode_strided_4d_block
TEST(encode, test_encode_strided_4d_block)
{
  zfp_input specs;
  float raw[4][4][4][4]; //* 4x4x4x4 source array
  uint32 encoded[BLOCK_SIZE_4D];

  //* Initialize raw with some data (e.g., sequence from 1 to 256)
  for (int w = 0; w < 4; w++)
    for (int z = 0; z < 4; z++)
      for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
          raw[w][z][y][x] = (float)(x + 4 * y + 16 * z + 64 * w + 1);

  //* Test test_encode_strided_4d_block
  test_encode_strided_4d_block(encoded, (const float*)raw, 1, 4, 16, 64, &specs);

  //* Check correctness of encoded values
  for (int i = 0; i < BLOCK_SIZE_4D; i++) {
    EXPECT_EQ(encoded[i], (uint32)(i + 1));
  }
}

TEST(encode, test_encode_strided_partial_2d_block)
{
  zfp_input specs;
  const int ROWS = 2;
  const int COLS = 3;
  float raw[ROWS][COLS]; //* 2x3 partial source array
  uint32 encoded[BLOCK_SIZE_2D];

  //* Initialize raw with some data (e.g., sequence from 1 to 6)
  for (int y = 0; y < ROWS; y++)
    for (int x = 0; x < COLS; x++)
      raw[y][x] = (float)(x + COLS * y + 1);

  //* Test test_encode_strided_partial_2d_block
  test_encode_strided_partial_2d_block(
    encoded, (const float*)raw, COLS, ROWS, 1, COLS, &specs);

  //* Check correctness of the encoding size.
  EXPECT_EQ(sizeof(encoded), sizeof(uint32) * BLOCK_SIZE_2D);

  printf("Raw values:\n");
  for (int i = 0; i < ROWS; i++) {
    for (int j = 0; j < COLS; ++j)
      printf("%.3f ", raw[i][j]);
    printf("\n");
  }
  printf("Encoded values:\n");
  for (int i = 0; i < BLOCK_SIZE_2D; i++) {
    printf("%d ", encoded[i]);
    if ((i + 1) % 4 == 0)
      printf("\n");
  }

  //* Check correctness of encoded values
  for (int i = 0; i < 4; i++) {
    //* The first and last rows should always be the same.
    EXPECT_EQ(encoded[i], encoded[i + 3*4]) \
        << "[" << i << "]=" << encoded[i] << " " << "[" << i + 3*4 << "]=" << encoded[i
            + 3*4] << "\n";
    if (ROWS < 3) {
      //* The second and third rows should always be the same.
      EXPECT_EQ(encoded[i + 1*4], encoded[i + 2*4]) \
          << "[" << i + 1*4 << "]=" << encoded[i + 1*4] << " " << "[" << i + 2*4 << "]="
          << encoded[i + 2*4] << "\n";
    }
  }
}

TEST(encode, test_2d)
{

  zfp_input specs;
  specs.nx = 10;
  specs.ny = 11;
  specs.nz = 0;
  specs.nw = 0;
  specs.sx = 1;
  specs.sy = specs.nx;
  specs.sz = 0;
  specs.sw = 0;

  float data[specs.nx*specs.ny];
  specs.data = (void*)data;
  //TODO: Compression params

  size_t encoded_x = ceil(specs.nx / 4.0) * 4;
  size_t encoded_y = ceil(specs.ny / 4.0) * 4;
  size_t num_outputs = encoded_x * encoded_y;
  printf("Total outputs: %zu\n", num_outputs);
  uint32 compressed[num_outputs];
  std::fill_n(compressed, num_outputs, 0.0);

  //* Initialize data with some data (e.g., sequence from 1 to 110)
  for (size_t y = 0; y < specs.ny; y++)
    for (size_t x = 0; x < specs.nx; x++)
      data[y * specs.nx + x] = (float)(y * specs.nx + x + 1);

  printf("Raw values:\n");
  // print_2d<float>(data, specs.nx, specs.ny);

  //* Test test_2d
  test_2d(compressed, &specs);

  printf("Encoded values:\n");
  // print_2d<uint32>(compressed, encoded_x, encoded_y, 3);

  EXPECT_EQ(sizeof(compressed), sizeof(uint32) * num_outputs);
  //TODO: Block-wise checks.
}

int main(int argc, char** argv)
{
  printf("\nEncoder Tests: \n");
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

