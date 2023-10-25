#include <stdio.h>
#include <stdbool.h>

#include "gtest/gtest.h"

// extern "C" {
//  // * Avoid name mangling for C functions.
#include "encode.h"
#include "types.h"
// }


TEST(encode, gather_2d_block)
{
  double raw[4][4]; //* 4x4 source array
  double block[BLOCK_SIZE_2D];

  //* Initialize raw with some data (e.g., sequence from 1 to 16)
  for (int y = 0; y < 4; y++)
    for (int x = 0; x < 4; x++)
      raw[y][x] = (double)(x + 4 * y + 1);

  //* Test gather_2d_block
  gather_2d_block(block, (const double*)raw, 1, 4);

  //* Check correctness of gathered values
  for (int i = 0; i < BLOCK_SIZE_2D; i++) {
    EXPECT_EQ(block[i], (double)(i + 1));
  }
}


//* Test function for gather_4d_block
TEST(encode, gather_4d_block)
{
  double raw[4][4][4][4]; //* 4x4x4x4 source array
  double block[BLOCK_SIZE_4D];

  //* Initialize raw with some data (e.g., sequence from 1 to 256)
  for (int w = 0; w < 4; w++)
    for (int z = 0; z < 4; z++)
      for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
          raw[w][z][y][x] = (double)(x + 4 * y + 16 * z + 64 * w + 1);

  //* Test gather_4d_block
  gather_4d_block(block, (const double*)raw, 1, 4, 16, 64);

  //* Check correctness of gathered values
  for (int i = 0; i < BLOCK_SIZE_4D; i++) {
    EXPECT_EQ(block[i], (double)(i + 1));
  }
}

TEST(encode, encode_strided_2d_block)
{
  double raw[4][4]; //* 4x4 source array
  uint64 encoded[BLOCK_SIZE_2D];

  //* Initialize raw with some data (e.g., sequence from 1 to 16)
  for (int y = 0; y < 4; y++)
    for (int x = 0; x < 4; x++)
      raw[y][x] = (double)(x + 4 * y + 1);

  //* Test encode_strided_2d_block
  encode_strided_2d_block(encoded, (const double*)raw, 1, 4);

  //* Check correctness of encoded values
  for (int i = 0; i < BLOCK_SIZE_2D; i++) {
    EXPECT_EQ(encoded[i], (uint64)(i + 1));
  }
}

//* Test function for encode_strided_4d_block
TEST(encode, encode_strided_4d_block)
{
  double raw[4][4][4][4]; //* 4x4x4x4 source array
  uint64 encoded[BLOCK_SIZE_4D];

  //* Initialize raw with some data (e.g., sequence from 1 to 256)
  for (int w = 0; w < 4; w++)
    for (int z = 0; z < 4; z++)
      for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
          raw[w][z][y][x] = (double)(x + 4 * y + 16 * z + 64 * w + 1);

  //* Test encode_strided_4d_block
  encode_strided_4d_block(encoded, (const double*)raw, 1, 4, 16, 64);

  //* Check correctness of encoded values
  for (int i = 0; i < BLOCK_SIZE_4D; i++) {
    EXPECT_EQ(encoded[i], (uint64)(i + 1));
  }
}

TEST(encode, encode_strided_partial_2d_block)
{
  const int ROWS = 2;
  const int COLS = 3;
  double raw[ROWS][COLS]; //* 2x3 partial source array
  uint64 encoded[BLOCK_SIZE_2D];

  //* Initialize raw with some data (e.g., sequence from 1 to 6)
  for (int y = 0; y < ROWS; y++)
    for (int x = 0; x < COLS; x++)
      raw[y][x] = (double)(x + COLS * y + 1);

  //* Test encode_strided_partial_2d_block
  encode_strided_partial_2d_block(encoded, (const double*)raw, COLS, ROWS, 1,
                                  COLS);

  //* Check correctness of the encoding size.
  EXPECT_EQ(sizeof(encoded), sizeof(uint64) * BLOCK_SIZE_2D);

  printf("Raw values:\n");
  for (int i = 0; i < ROWS; i++) {
    for (int j = 0; j < COLS; ++j)
      printf("%.3f ", raw[i][j]);
    printf("\n");
  }
  printf("Encoded values:\n");
  for (int i = 0; i < BLOCK_SIZE_2D; i++) {
    printf("%lld ", encoded[i]);
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

TEST(encode, compress_2d)
{

  zfp_specs specs;
  specs.nx = 10;
  specs.ny = 11;
  specs.nz = 0;
  specs.nw = 0;
  specs.sx = 1;
  specs.sy = 11;
  specs.sz = 0;
  specs.sw = 0;

  double data[specs.nx][specs.ny];
  specs.data = (void*)data;
  //TODO: Compression params
  uint64 compressed[specs.nx*specs.ny];

  //* Initialize data with some data (e.g., sequence from 1 to 110)
  for (size_t y = 0; y < specs.ny; y++)
    for (size_t x = 0; x < specs.nx; x++)
      data[y][x] = (double)(y * specs.nx + x + 1);

  //* Test compress_2d
  compress_2d(compressed, &specs);

  //* Check correctness of the encoding size.
  EXPECT_EQ(sizeof(compressed), sizeof(uint64) * specs.nx * specs.ny);
  printf("Passed size check.\n");

  //* Check correctness of (full-block) encoded values.
  double* p = (double*)specs.data;
  for (int y = 0; y < specs.ny; y++, p += specs.sy) {
    for (int x = 0; x < specs.nx; x++, p += specs.sx) {
      EXPECT_EQ(compressed[y * specs.nx + x], (uint64)(specs.nx * y + x + 1));
    }
  }
}

int main(int argc, char** argv)
{
  printf("\n");
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

