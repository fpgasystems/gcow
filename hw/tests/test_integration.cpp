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
  std::vector<float, aligned_allocator<float>> fblock(BLOCK_SIZE_2D, 0.0);
  size_t n = 4;
  size_t nx = n;
  size_t ny = n;

  for (size_t j = 0; j < ny; j++)
    for (size_t i = 0; i < nx; i++) {
      double x = 2.0 * i / nx;
      double y = 2.0 * j / ny;
      fblock[i + nx * j] = (float)exp(-(x * x + y * y));
    }
  std::cout << "input floats:\t" << fblock.size() << std::endl;

  //* Initialize output.
  size_t total_blocks = 1;
  std::vector<uint32, aligned_allocator<uint32>> ublock(BLOCK_SIZE_2D*total_blocks);

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
  cl::Kernel kernel(program, "integration");
  std::cout << "Created a kernel using the loaded program object\n";

  // When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the hood user ptr
  // is used if it is properly aligned. when not aligned, runtime had no choice but to create
  // its own host side buffer. So it is recommended to use this allocator if user wish to
  // create buffer using CL_MEM_USE_HOST_PTR to align user buffer to page boundary. It will
  // ensure that user buffer is used when user create Buffer/Mem object with CL_MEM_USE_HOST_PTR

  //* Allocate buffers in Global Memory
  OCL_CHECK(err,
            cl::Buffer buffer_fblock(
              context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
              BLOCK_SIZE_2D*sizeof(float),
              fblock.data(),
              &err));

  OCL_CHECK(err,
            cl::Buffer buffer_ublock(
              context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
              total_blocks*BLOCK_SIZE_2D*sizeof(uint32),
              ublock.data(),
              &err));

  std::cout << "Finished allocating buffers\n";

  //* Set the Kernel Arguments
  int arg_counter = 0;
  OCL_CHECK(err,
            err = kernel.setArg(arg_counter++, buffer_fblock));
  OCL_CHECK(err,
            err = kernel.setArg(arg_counter++, total_blocks));
  OCL_CHECK(err,
            err = kernel.setArg(arg_counter++, buffer_ublock));

  //* Copy input data to device global memory
  OCL_CHECK(err,
  err = q.enqueueMigrateMemObjects({
    //* Input data objects
    buffer_fblock,
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
    buffer_ublock,
  }, CL_MIGRATE_MEM_OBJECT_HOST /* 1: from device to host */));
  q.finish();

  auto end = std::chrono::high_resolution_clock::now();
  double duration = (std::chrono::duration_cast<std::chrono::milliseconds>
                     (end-start).count() / 1000.0);

  std::cout << "Duration (including memcpy out): " << duration << " seconds"
            << std::endl;
  std::cout << "Overall grad values per second = " << BLOCK_SIZE_2D / duration
            << std::endl;

  // int32 expected[BLOCK_SIZE_2D] = {
  //   536870912, 418115488, 197503776, 56585776, 
  //   418115488, 325628672, 153816096, 44069048, 
  //   197503776, 153816096, 72657576, 20816744, 
  //   56585776, 44069048, 20816744, 5964097
  // };

  // int32 expected[BLOCK_SIZE_2D] = {
  //   170183444, 92266196, 3119492, 12777032, 
  //   92266197, 50022792, 1691257, 6927161, 
  //   3119493, 1691256, 57181, 234206, 
  //   12777032, 6927161, 234205, 959274,
  // };

  //* Expected 3-block output for dim=4x4.
  uint32 expected[BLOCK_SIZE_2D] = {
    509992724, 444605396, 444605397, 118447768, 
    7401092, 7401093, 7263113, 7263112, 
    29821528, 29821528, 73901, 29292361, 
    29292361, 300834, 300845, 1304446
  };

  // //* Expected single-block output for dim=3x3.
  // uint32 expected[BLOCK_SIZE_2D] = {
  //   462604581, 104049822, 461851134, 110467290, 47986086, 230720545, 47262228, 16676584, 27041915, 3113022, 23275410, 24462429, 29334312, 62007498, 7458168, 64915459
  // };

  //* Validate against software implementation.
  bool matched = true;
  for (size_t i = 0; i < total_blocks; i++) {
    for (size_t j = 0; j < BLOCK_SIZE_2D; j++) {
      if (ublock.at(i * BLOCK_SIZE_2D + j) != expected[j]) {
        std::cout << "ublock[" << i << "][" << j << "] = " << ublock.at(i * BLOCK_SIZE_2D + j)
                  << " != " << expected[j] << std::endl;
        matched = false;
      } else {
        std::cout << "ublock[" << i << "][" << j << "] = " << ublock.at(i * BLOCK_SIZE_2D + j) << std::endl;
      }
    }
  }

  std::cout << "TEST " << (matched ? "PASSED" : "FAILED") << std::endl;
  return (matched ? EXIT_SUCCESS : EXIT_FAILURE);
}
