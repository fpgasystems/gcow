#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <iostream>

#include "gtest/gtest.h"

#include "encode.h"
#include "stream.h"


TEST(stages, emax)
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

TEST(stages, fwd_cast)
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

TEST(stages, fwd_decorrelate)
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

  for (int i = 0; i < BLOCK_SIZE_2D; i++) {
    EXPECT_EQ(iblock[i], expected[i]);
  }
}

int main(int argc, char** argv)
{
  printf("\nStages Tests: \n");
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}