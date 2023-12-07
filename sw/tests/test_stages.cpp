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

int main(int argc, char** argv)
{
  printf("\nStages Tests: \n");
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}