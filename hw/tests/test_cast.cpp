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
  std::vector<float, aligned_allocator<float>> fblock(BLOCK_SIZE_2D);
  size_t nx = 4;
  size_t ny = 4;

  for (size_t j = 0; j < ny; j++)
    for (size_t i = 0; i < nx; i++) {
      double x = 2.0 * i / nx;
      double y = 2.0 * j / ny;
      fblock[i + nx * j] = (float)exp(-(x * x + y *
                                        y)); // (float)(x + 100 * y + 3.1415926);
    }
  std::cout << "input floats:\t" << fblock.size() << std::endl;

  //* Initialize output.
  std::vector<int32, aligned_allocator<int32>> iblock(BLOCK_SIZE_2D);

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
  cl::Kernel kernel(program, "cast");
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
            cl::Buffer buffer_iblock(
              context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
              BLOCK_SIZE_2D*sizeof(int32),
              iblock.data(),
              &err));

  std::cout << "Finished allocating buffers\n";

  //* Set the Kernel Arguments
  int emax = 1;
  int arg_counter = 0;
  OCL_CHECK(err,
            err = kernel.setArg(arg_counter++, buffer_fblock));
  OCL_CHECK(err,
            err = kernel.setArg(arg_counter++, BLOCK_SIZE_2D));
  OCL_CHECK(err,
            err = kernel.setArg(arg_counter++, emax));
  OCL_CHECK(err,
            err = kernel.setArg(arg_counter++, buffer_iblock));

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
    buffer_iblock,
  }, CL_MIGRATE_MEM_OBJECT_HOST /* 1: from device to host */));
  q.finish();

  auto end = std::chrono::high_resolution_clock::now();
  double duration = (std::chrono::duration_cast<std::chrono::milliseconds>
                     (end-start).count() / 1000.0);

  std::cout << "Duration (including memcpy out): " << duration << " seconds"
            << std::endl;
  std::cout << "Overall grad values per second = " << BLOCK_SIZE_2D / duration
            << std::endl;

  int32 expected[BLOCK_SIZE_2D] = {
    536870912, 418115488, 197503776, 56585776, 
    418115488, 325628672, 153816096, 44069048, 
    197503776, 153816096, 72657576, 20816744, 
    56585776, 44069048, 20816744, 5964097
  };

  //* Validate against software implementation.
  bool matched = true;
  for (int i = 0; i < BLOCK_SIZE_2D; i++) {
    if (iblock.at(i) != expected[i]) {
      std::cout << "iblock[" << i << "] = " << iblock.at(i)
                << " != " << expected[i] << std::endl;
      matched = false;
    }
  }

  std::cout << "TEST " << (matched ? "PASSED" : "FAILED") << std::endl;
  return (matched ? EXIT_SUCCESS : EXIT_FAILURE);
}
