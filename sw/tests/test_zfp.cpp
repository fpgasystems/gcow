#include <stdio.h>
#include <stdbool.h>
#include <math.h>

#include "gtest/gtest.h"
#include "encode.h"
#include "stream.h"
#include "zfp.h"


class TestZfp2D : public ::testing::TestWithParam<std::tuple<int>> {};

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

void compare_two_files(const char *file1, const char *file2)
{
  FILE *fp1 = fopen(file1, "rb");
  FILE *fp2 = fopen(file2, "rb");

  if (!fp1 || !fp2) {
    printf("Failed to open file for reading.\n");
    exit(1);
  }

  fseek(fp1, 0, SEEK_END);
  fseek(fp2, 0, SEEK_END);

  long size1 = ftell(fp1);
  long size2 = ftell(fp2);

  EXPECT_EQ(size1, size2) << "File sizes are not the same.\n";

  rewind(fp1);
  rewind(fp2);

  char *buffer1 = (char*)malloc(size1);
  char *buffer2 = (char*)malloc(size2);

  fread(buffer1, 1, size1, fp1);
  fread(buffer2, 1, size2, fp2);

  EXPECT_EQ(memcmp(buffer1, buffer2, size1),
            0) << "File contents are not the same\n.";

  fclose(fp1);
  fclose(fp2);
}

TEST_P(TestZfp2D, compress)
{
  size_t n = std::get<0>(GetParam());
  printf("Testing size: %ldx%ld\n", n, n);

  float *input_data = (float*)malloc(n * n * sizeof(float));
  get_input_2d(input_data, n);

  zfp_input *input = init_zfp_input(input_data, dtype_float, 2, n, n);
  zfp_output *output = init_zfp_output(input);

  double tolerance = 1e-3;
  double max_error = set_zfp_output_accuracy(output, tolerance);
  printf("Maximum error:\t\t%f\n", max_error);

  size_t output_size = zfp_compress(output, input);
  printf("Raw data size:\t\t%ld bytes\n", n * n * sizeof(float));
  printf("Compressed size:\t%ld bytes\n", output_size);

  std::stringstream zfpf;
  zfpf << "tests/data/compressed_2d_" << n << ".zfp";
  std::stringstream gcowf;
  gcowf << "tests/data/compressed_2d_" << n << ".gcow";

  FILE *fp = fopen(gcowf.str().c_str(), "wb");
  if (!fp) {
    printf("Failed to open file for writing.\n");
    exit(1);
  } else {
    fwrite(output->data->begin, 1, output_size, fp);
    fclose(fp);
  }

  compare_two_files(gcowf.str().c_str(), zfpf.str().c_str());
  cleanup(input, output);
}

INSTANTIATE_TEST_SUITE_P(zfp, TestZfp2D, ::testing::Values(
                           3, 8, 123, 210, 354, 505, 510, 7654
                         ));

int main(int argc, char** argv)
{
  printf("\nZFP Tests: \n");
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}