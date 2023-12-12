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

  //* Initialize output.
  size_t max_bytes = 1000; //* Does not matter for this test, just a bound.
  std::vector<stream_word, aligned_allocator<stream_word>> out_data(
        max_bytes / sizeof(stream_word));

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
  cl::Kernel kernel(program, "writes");
  std::cout << "Created a kernel using the loaded program object\n";

  // When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the hood user ptr
  // is used if it is properly aligned. when not aligned, runtime had no choice but to create
  // its own host side buffer. So it is recommended to use this allocator if user wish to
  // create buffer using CL_MEM_USE_HOST_PTR to align user buffer to page boundary. It will
  // ensure that user buffer is used when user create Buffer/Mem object with CL_MEM_USE_HOST_PTR

  //* Allocate buffers in Global Memory
  OCL_CHECK(err,
            cl::Buffer buffer_out_data(
              context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
              max_bytes,
              out_data.data(),
              &err));

  ptrdiff_t stream_idx = 0;
  OCL_CHECK(err,
            cl::Buffer buffer_stream_idx(
              context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
              sizeof(ptrdiff_t),
              &stream_idx,
              &err));


  std::cout << "Finished allocating buffers\n";

  //* Set the Kernel Arguments
  int arg_counter = 0;
  OCL_CHECK(err,
            err = kernel.setArg(arg_counter++, buffer_out_data));
  OCL_CHECK(err,
            err = kernel.setArg(arg_counter++, buffer_stream_idx));

  //* Copy input data to device global memory
  // OCL_CHECK(err,
  // err = q.enqueueMigrateMemObjects({
  //   //* Input data objects
  //   buffer_ublock,
  // }, 0 /* 0: from host to device */));

  //* Launch the Kernel
  std::cout << "Launching kernel ...\n";
  auto start = std::chrono::high_resolution_clock::now();
  OCL_CHECK(err,
            err = q.enqueueTask(kernel));

  std::cout << "Kernel running ...\n";
  //* Copy Result from Device Global Memory to Host Local Memory
  OCL_CHECK(err,
  err = q.enqueueMigrateMemObjects({
    buffer_out_data,
    buffer_stream_idx,
  }, CL_MIGRATE_MEM_OBJECT_HOST /* 1: from device to host */));
  q.finish();

  auto end = std::chrono::high_resolution_clock::now();
  double duration = (std::chrono::duration_cast<std::chrono::milliseconds>
                     (end-start).count() / 1000.0);

  std::cout << "Duration (including memcpy out): " << duration << " seconds"
            << std::endl;
  std::cout << "Overall grad values per second = " << BLOCK_SIZE_2D / duration
            << std::endl;

  ptrdiff_t stream_idx_host = 2;
  uint64 expected[stream_idx_host] = {
    7455816852505100291UL,
    432UL
  };

  //* Validate against software implementation.
  assert (stream_idx_host == stream_idx);
  std::cout << "Stream idx: " << stream_idx << std::endl;

  bool matched = true;
  for (int i = 0; i < stream_idx_host; i++) {
    if (out_data.at(i) != expected[i]) {
      std::cout << "out_data[" << i << "] = " << out_data.at(i)
                << " != " << expected[i] << std::endl;
      matched = false;
    } else {
      std::cout << "out_data[" << i << "] = " << out_data.at(i) << std::endl;
    }
  }

  std::cout << "TEST " << (matched ? "PASSED" : "FAILED") << std::endl;
  return (matched ? EXIT_SUCCESS : EXIT_FAILURE);
}
