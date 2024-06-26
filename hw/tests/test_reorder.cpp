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
  std::vector<int32, aligned_allocator<int32>> iblock = {
    664778, 360415, 12185, 49910,
    360415, 195402, 6607, 27060,
    12186, 6607, 224, 915,
    49910, 27059, 915, 3747
  };
  // std::vector<int32, aligned_allocator<int32>> iblock = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
  //  11, 12, 13, 14, 15, 16};

  //* Initialize output.
  std::vector<uint32, aligned_allocator<uint32>> ublock(BLOCK_SIZE_2D);

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
  cl::Kernel kernel(program, "reorder");
  std::cout << "Created a kernel using the loaded program object\n";

  // When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the hood user ptr
  // is used if it is properly aligned. when not aligned, runtime had no choice but to create
  // its own host side buffer. So it is recommended to use this allocator if user wish to
  // create buffer using CL_MEM_USE_HOST_PTR to align user buffer to page boundary. It will
  // ensure that user buffer is used when user create Buffer/Mem object with CL_MEM_USE_HOST_PTR

  //* Allocate buffers in Global Memory
  OCL_CHECK(err,
            cl::Buffer buffer_iblock(
              context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
              BLOCK_SIZE_2D*sizeof(int32),
              iblock.data(),
              &err));
  OCL_CHECK(err,
            cl::Buffer buffer_ublock(
              context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
              BLOCK_SIZE_2D*sizeof(uint32),
              ublock.data(),
              &err));

  std::cout << "Finished allocating buffers\n";

  //* Set the Kernel Arguments
  size_t num_blocks = 4198401; // 66; // 4198401;
  int arg_counter = 0;
  OCL_CHECK(err,
            err = kernel.setArg(arg_counter++, buffer_iblock));
  OCL_CHECK(err,
            err = kernel.setArg(arg_counter++, num_blocks));
  OCL_CHECK(err,
            err = kernel.setArg(arg_counter++, buffer_ublock));

  //* Copy input data to device global memory
  OCL_CHECK(err,
  err = q.enqueueMigrateMemObjects({
    //* Input data objects
    buffer_iblock,
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

  return EXIT_SUCCESS;

  // uint32 expected[BLOCK_SIZE_2D] = {
  //   1992158, 1736739, 1736739, 462686,
  //   28905, 28910, 28371, 28371,
  //   116490, 116490, 288, 114420,
  //   114423, 1175, 1175, 5095
  // };

  // // uint32 expected[BLOCK_SIZE_2D] = {1, 6, 5, 26, 7, 25, 27, 30, 4, 29, 31, 24, 18, 28, 19, 16};

  // //* Validate against software implementation.
  // bool matched = true;
  // for (int i = 0; i < BLOCK_SIZE_2D; i++) {
  //   if (ublock.at(i) != expected[i]) {
  //     std::cout << "(ublock[" << i << "] = " << ublock.at(i)
  //               << " != " << expected[i] << ")";
  //     matched = false;
  //   }
  //   //* Print the values with each row on a new line.
  //   std::cout << ublock.at(i) << " ";
  //   if (i % 4 == 0) {
  //     std::cout << std::endl;
  //   }
  // }

  // std::cout << "TEST " << (matched ? "PASSED" : "FAILED") << std::endl;
  // return (matched ? EXIT_SUCCESS : EXIT_FAILURE);
}
