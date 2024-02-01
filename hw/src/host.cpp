#include <chrono>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "host.hpp"
#include "types.hpp"


void get_input_2d(float *input_data, size_t n);

int main(int argc, char** argv)
{
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << " <dim>" << std::endl;
    return EXIT_FAILURE;
  }
  cl_int err;
  std::cout << std::endl << "--------------------------" << std::endl;

  //* Initialize input.
  size_t dim = std::stoi(argv[2]);
  size_t in_dim = 2;
  size_t shape[DIM_MAX] = {dim, dim};
  zfp_input in_specs(dtype_float, shape, in_dim);
  assert(in_dim == get_input_dimension(in_specs));
  std::vector<size_t, aligned_allocator<size_t>> in_shape(shape, shape + in_dim);

  std::cout << "Input specs" << std::endl;
  std::cout << "dtype:\t\t" << in_specs.dtype << std::endl;
  std::cout << "nx:\t\t" << in_specs.nx << std::endl;
  std::cout << "ny:\t\t" << in_specs.ny << std::endl;
  std::cout << "nz:\t\t" << in_specs.nz << std::endl;
  std::cout << "nw:\t\t" << in_specs.nw << std::endl;

  size_t input_size = get_input_size(in_specs);
  std::cout << "input size:\t" << input_size << std::endl;
  std::cout << "input dim:\t" << in_dim << std::endl;

  std::vector<float, aligned_allocator<float>> in_fp_gradients(input_size, 0.0);
  get_input_2d(in_fp_gradients.data(), dim);
  // //* Print input data.
  // for (size_t i = 0; i < input_size; i++) {
  //   std::cout << in_fp_gradients[i] << " ";
  // }
  std::cout << "input floats:\t" << in_fp_gradients.size() << std::endl;

  //* Initialize output.
  zfp_output out_specs;
  size_t max_output_bytes = get_max_output_bytes(out_specs, in_specs);
  std::cout << "Max output bytes:\t" << max_output_bytes << std::endl;

  std::vector<stream_word, aligned_allocator<stream_word>> out_zfp_gradients(
        max_output_bytes / sizeof(stream_word), 0);
  std::cout << "Output buffer size:\t" <<
            out_zfp_gradients.size() << std::endl;

  /* OPENCL HOST CODE AREA START */
  std::cout << std::endl << "--------------------------" << std::endl;
  std::vector<cl::Device> devices = get_devices();
  cl::Device device = devices[0];
  std::string device_name = device.getInfo<CL_DEVICE_NAME>();
  std::cout << "Found Device: " << device_name.c_str() << std::endl;

  //* Creating Context and Command Queue for selected device
  cl::Context context(device);
  cl::CommandQueue q(context, device);

  //* Import XCLBIN
  xclbin_file_name = argv[1];
  cl::Program::Binaries gcow_bins = import_binary_file();
  std::cout << "Finished importing binary\n\n";

  //* Program and Kernel
  devices.resize(1);
  cl::Program program(context, devices, gcow_bins);
  std::cout << "\nCreated program from the bitstream\n";
  cl::Kernel kernel_gcow(program, "gcow");
  std::cout << "Created a kernel using the loaded program object\n";

  // When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the hood user ptr
  // is used if it is properly aligned. when not aligned, runtime had no choice but to create
  // its own host side buffer. So it is recommended to use this allocator if user wish to
  // create buffer using CL_MEM_USE_HOST_PTR to align user buffer to page boundary. It will
  // ensure that user buffer is used when user create Buffer/Mem object with CL_MEM_USE_HOST_PTR

  //* Allocate buffers in Global Memory
  //& Read-only buffers for inputs.
  OCL_CHECK(err,
            cl::Buffer buffer_in_shape(
              context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
              in_dim*sizeof(size_t),
              in_shape.data(),
              &err));
  OCL_CHECK(err,
            cl::Buffer buffer_in_gradients(
              context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
              input_size*sizeof(float),
              in_fp_gradients.data(),
              &err));
  OCL_CHECK(err,
            cl::Buffer buffer_out_gradients(
              context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
              max_output_bytes,
              out_zfp_gradients.data(),
              &err));

  //& Read-write/write-only buffers for outputs.
  size_t out_bytes[] = {0};
  OCL_CHECK(err,
            cl::Buffer buffer_out_bytes(
              context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
              sizeof(size_t),
              out_bytes,
              &err));

  std::cout << "Finished allocating buffers\n";

  //* Set the Kernel Arguments
  int arg_counter = 0;
  OCL_CHECK(err,
            err = kernel_gcow.setArg(arg_counter++, in_dim));
  OCL_CHECK(err,
            err = kernel_gcow.setArg(arg_counter++, buffer_in_shape));
  OCL_CHECK(err,
            err = kernel_gcow.setArg(arg_counter++, buffer_in_gradients));
  OCL_CHECK(err,
            err = kernel_gcow.setArg(arg_counter++, buffer_out_gradients));
  OCL_CHECK(err,
            err = kernel_gcow.setArg(arg_counter++, buffer_out_bytes));

  //* Copy input data to device global memory
  OCL_CHECK(err,
  err = q.enqueueMigrateMemObjects({
    //* Input data objects
    buffer_in_shape,
    buffer_in_gradients,
  }, 0 /* 0: from host to device */));

  //* Launch the Kernel
  std::cout << "Launching kernel ...\n";
  auto start = std::chrono::high_resolution_clock::now();
  OCL_CHECK(err,
            err = q.enqueueTask(kernel_gcow));

  std::cout << "Kernel running ...\n";
  //* Copy Result from Device Global Memory to Host Local Memory
  OCL_CHECK(err,
  err = q.enqueueMigrateMemObjects({
    buffer_out_gradients,
    buffer_out_bytes,
  }, CL_MIGRATE_MEM_OBJECT_HOST /* 1: from device to host */));
  OCL_CHECK(err, err = q.finish());

  auto end = std::chrono::high_resolution_clock::now();
  double duration = (std::chrono::duration_cast<std::chrono::milliseconds>
                     (end-start).count());

  std::cout << "Duration (including memcpy out): " << duration << " ms"
            << std::endl;
  std::cout << "Overall grad values per second = " << input_size / duration
            << std::endl;
  std::cout << "Compressed size: " << *out_bytes << " bytes" << std::endl;

  // return EXIT_SUCCESS;
  
  //* Validate against software implementation.
  std::stringstream zfpf;
  if (dim > 1e4)
    zfpf << "tests/data/compressed_2d_" << dim << "_large.zfp";
  else
    zfpf << "tests/data/compressed_2d_" << dim << ".zfp";

  std::stringstream gcowf;
  if (dim > 1e4)
    gcowf << "tests/data/compressed_2d_" << dim << "_large.gcow";
  else
    gcowf << "tests/data/compressed_2d_" << dim << ".gcow";

  //* Dump the compressed data to file.
  FILE *fp = fopen(gcowf.str().c_str(), "wb");
  if (!fp) {
    printf("Failed to open file for writing.\n");
    exit(1);
  } else {
    fwrite(out_zfp_gradients.data(), 1, *out_bytes, fp);
    fclose(fp);
    std::cout << "Dumped compressed data to " << gcowf.str() << std::endl;
  }

  std::stringstream diffcmd;
  diffcmd << "diff --brief -w " << gcowf.str() << " " << zfpf.str();
  bool matched = !system(diffcmd.str().c_str());

  std::cout << "TEST " << (matched ? "PASSED" : "FAILED") << std::endl;
  return (matched ? EXIT_SUCCESS : EXIT_FAILURE);
}


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