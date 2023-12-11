#include <chrono>
#include <cassert>
#include <fstream>
#include <iostream>

#include "host.hpp"
#include "types.hpp"


int main(int argc, char** argv)
{
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
    return EXIT_FAILURE;
  }
  cl_int err;

  //* Initialize input.
  uint dim = 2;
  size_t nx = 5;
  size_t ny = 7;
  size_t total = nx * ny;
  uint num_blocks = ceil((float)nx / 4) * ceil((float)ny / 4);
  std::vector<float, aligned_allocator<float>> in_block(total);
  //* Initialize array to be compressed from 1 to nx*ny:
  for (size_t y = 0; y < ny; y++)
    for (size_t x = 0; x < nx; x++)
      in_block.at(x + nx * y) = (float)(x + nx * y + 1);

  std::cout << "input floats:\t" << in_block.size() << std::endl;

  //* Initialize output.
  std::vector<float, aligned_allocator<float>> out_block(
        num_blocks*BLOCK_SIZE_2D);

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
  cl::Kernel kernel(program, "gather");
  std::cout << "Created a kernel using the loaded program object\n";

  // When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the hood user ptr
  // is used if it is properly aligned. when not aligned, runtime had no choice but to create
  // its own host side buffer. So it is recommended to use this allocator if user wish to
  // create buffer using CL_MEM_USE_HOST_PTR to align user buffer to page boundary. It will
  // ensure that user buffer is used when user create Buffer/Mem object with CL_MEM_USE_HOST_PTR

  //* Allocate buffers in Global Memory
  OCL_CHECK(err,
            cl::Buffer buffer_inblock(
              context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
              total*sizeof(float),
              in_block.data(),
              &err));

  OCL_CHECK(err,
            cl::Buffer buffer_outblock(
              context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
              num_blocks*BLOCK_SIZE_2D*sizeof(float),
              out_block.data(),
              &err));

  std::cout << "Finished allocating buffers\n";

  //* Set the Kernel Arguments
  int arg_counter = 0;
  OCL_CHECK(err,
            err = kernel.setArg(arg_counter++, buffer_inblock));
  OCL_CHECK(err,
            err = kernel.setArg(arg_counter++, nx));
  OCL_CHECK(err,
            err = kernel.setArg(arg_counter++, ny));
  OCL_CHECK(err,
            err = kernel.setArg(arg_counter++, buffer_outblock));

  //* Copy input data to device global memory
  OCL_CHECK(err,
  err = q.enqueueMigrateMemObjects({
    //* Input data objects
    buffer_inblock,
  }, 0 /* 0: from host to device */));

  //* Launch the Kernel
  std::cout << "Launching kernel ...\n";
  auto start = std::chrono::high_resolution_clock::now();
  OCL_CHECK(err,
            err = q.enqueueTask(kernel));

  std::cout << "Kernel running ...\n";
  //* Copy Result from Device Global Memory to Host Local Memory
  OCL_CHECK(err,
  err = q.enqueueMigrateMemObjects({
    buffer_outblock,
  }, CL_MIGRATE_MEM_OBJECT_HOST /* 1: from device to host */));
  q.finish();

  auto end = std::chrono::high_resolution_clock::now();
  double duration = (std::chrono::duration_cast<std::chrono::milliseconds>
                     (end-start).count() / 1000.0);

  std::cout << "Duration (including memcpy out): " << duration << " seconds"
            << std::endl;
  std::cout << "Overall grad values per second = " << BLOCK_SIZE_2D / duration
            << std::endl;

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

  //* Validate against software implementation.
  bool matched = true;
  for (int i = 0; i < num_blocks*BLOCK_SIZE_2D; i++) {
    if ((int)out_block.at(i) != expected[i]) {
      std::cout << "out_block[" << i << "] = " << out_block.at(i)
                << " != " << expected[i] << std::endl;
      matched = false;
    }
    printf("%d, ", (int)out_block[i]);
    if ((i + 1) % 4 == 0) {
      printf("\n");
    }
  }

  std::cout << "TEST " << (matched ? "PASSED" : "FAILED") << std::endl;
  return (matched ? EXIT_SUCCESS : EXIT_FAILURE);
}
