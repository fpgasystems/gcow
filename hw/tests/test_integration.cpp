#include <chrono>
#include <cassert>
#include <fstream>
#include <iostream>

#include "host.hpp"
#include "types.hpp"
#include <bitset>


int main(int argc, char** argv)
{
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
    return EXIT_FAILURE;
  }
  cl_int err;

  //* Initialize input.
  std::vector<float, aligned_allocator<float>> fblock(BLOCK_SIZE_2D, 0.0);
  size_t n = 3;
  size_t nx = n;
  size_t ny = n;

  for (size_t j = 0; j < ny; j++)
    for (size_t i = 0; i < nx; i++) {
      double x = 2.0 * i / nx;
      double y = 2.0 * j / ny;
      fblock[i + nx * j] = (float)exp(-(x * x + y * y));
    }
  std::cout << "input floats:\t" << nx*ny << std::endl;

  //* Initialize output.
  size_t total_blocks = 1;
  size_t max_bytes = 1000; //* Does not matter for this test, just a bound.
  std::vector<uint32, aligned_allocator<uint32>> ublock(BLOCK_SIZE_2D*total_blocks);
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
            err = kernel.setArg(arg_counter++, buffer_out_data));
  OCL_CHECK(err,
            err = kernel.setArg(arg_counter++, buffer_stream_idx));
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
    buffer_out_data,
    buffer_stream_idx,
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


  // //* Expected 3-block output for dim=4x4.
  // uint32 expected[BLOCK_SIZE_2D] = {
  //   282897489, 33434444, 33434444, 1796011, 156265097, 156265097, 13133998, 13133998, 68099259, 68099256, 131453921, 8376857, 8376856, 38902892, 38902892, 16897137
  // };

  // //* Validate against software implementation.
  // bool matched = true;
  // for (size_t i = 0; i < total_blocks; i++) {
  //   for (size_t j = 0; j < BLOCK_SIZE_2D; j++) {
  //     if (ublock.at(i * BLOCK_SIZE_2D + j) != expected[j]) {
  //       std::cout << "ublock[" << i << "][" << j << "] = " << ublock.at(i * BLOCK_SIZE_2D + j)
  //                 << " != " << expected[j] << std::endl;
  //       matched = false;
  //     } else {
  //       std::cout << "ublock[" << i << "][" << j << "] = " << ublock.at(i * BLOCK_SIZE_2D + j) << std::endl;
  //     }
  //   }
  // }

  ptrdiff_t stream_idx_host = 4;
  uint64 expected[stream_idx_host] = {
    //* Results of 3 blocks (dim=4x4).
    // 12711260835255415041UL, 5058120776611336133UL, 9096252834960252658UL, 
    // 7789501227241241664UL, 10487902231007609841UL, 2274063208740063164UL, 
    // 6559061325237698320UL, 2621975557751902460UL, 280285426033304047UL

    //* Results of 1 blocks (dim=3x3).
    7846959668108800257UL, 9092781915241025288UL, 1168152206201298680UL, 33031166UL
  };

  //* Validate against software implementation.
  bool matched = true;
  for (int i = 0; i < stream_idx_host; i++) {
    std::bitset<64> out(out_data.at(i));
    std::bitset<64> val(expected[i]);
    if (out_data.at(i) != expected[i]) {
      std::cout << "out_data[" << i << "] = " << out
                << " != " << val << std::endl;
      matched = false;
    } else {
      std::cout << "out_data[" << i << "] = " << out << std::endl;
    }
  }

  std::cout << "Output words: " << stream_idx << std::endl;

  for (int i=0; i < stream_idx; i++) {
    std::bitset<64> out(out_data.at(i));
    std::cout << "out_data[" << i << "] = " << out << std::endl;
  }

  std::cout << "TEST " << (matched ? "PASSED" : "FAILED") << std::endl;
  return (matched ? EXIT_SUCCESS : EXIT_FAILURE);
}
