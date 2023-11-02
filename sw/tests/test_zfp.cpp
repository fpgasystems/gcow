#include <stdio.h>
#include <stdbool.h>
#include <math.h>

#include "gtest/gtest.h"
#include "encode.h"
#include "stream.h"
#include "zfp.h"

void get_input_2d(float *input_data, size_t n)
{
  size_t nx = n;
  size_t ny = n;
  /* initialize array to be compressed */
  size_t i, j;
  for (j = 0; j < ny; j++)
    for (i = 0; i < nx; i++) {
      double x = 2.0 * i / nx;
      double y = 2.0 * j / ny;
      input_data[i + nx * j] = (float)exp(-(x * x + y * y));
    }
}

TEST(zfp, compress)
{
  size_t n = 100;
  float *input_data = (float*)malloc(n * n * sizeof(float));
  get_input_2d(input_data, n);

  zfp_input *input = init_zfp_input(input_data, dtype_float, 2, n, n);
  zfp_output *output = init_zfp_output(input);

  double tolerance = 1e-3;
  double max_error = set_zfp_output_accuracy(output, tolerance);
  printf("Maximum error: %f\n", max_error);

  cleanup(input, output);
}

int main(int argc, char** argv)
{
  printf("\nZFP Tests: \n");
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}