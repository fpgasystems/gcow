#include <stdint.h>
#include <chrono>

#include "host.hpp"
#include "constants.hpp"
//? 2022.1 somehow does not seem to support linking ap_uint.h to host?
// #include "ap_uint.h"


int main(int argc, char** argv)
{
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
    return EXIT_FAILURE;
  }
  cl_int err;

  //* Initialize input.
  std::vector<double, aligned_allocator<double>> in_fp_gradients(GRAD_BLOCK_SIZE);
  for (int i = 0; i < GRAD_BLOCK_SIZE; i++) {
    in_fp_gradients[i] = i + 0.555;
  }
  std::cout << "Initialized double gradient input of size " <<
            in_fp_gradients.size() << std::endl;

  //* Initialize output.
  std::vector<int, aligned_allocator<int>> out_zfp_gradients(GRAD_BLOCK_SIZE);
  std::cout << "Initialized int gradient output of size " <<
            out_zfp_gradients.size() << std::endl;

  /* OPENCL HOST CODE AREA START */

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
  std::cout << "Finish importing binary ...\n";

  //* Program and Kernel
  devices.resize(1);
  cl::Program program(context, devices, gcow_bins);
  std::cout << "Created program from the bitstream\n";
  cl::Kernel kernel_gcow(program, "gcow");
  std::cout <<
            "Created a kernel which uses the program object previously loaded\n";

  // When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the hood user ptr
  // is used if it is properly aligned. when not aligned, runtime had no choice but to create
  // its own host side buffer. So it is recommended to use this allocator if user wish to
  // create buffer using CL_MEM_USE_HOST_PTR to align user buffer to page boundary. It will
  // ensure that user buffer is used when user create Buffer/Mem object with CL_MEM_USE_HOST_PTR

  //* Allocate buffers in Global Memory
  OCL_CHECK(err,
            cl::Buffer buffer_in_gradients(
              context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
              GRAD_BLOCK_SIZE*sizeof(double),
              in_fp_gradients.data(),
              &err));
  OCL_CHECK(err,
            cl::Buffer buffer_out_gradients(
              context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
              GRAD_BLOCK_SIZE*sizeof(int),
              out_zfp_gradients.data(),
              &err));

  std::cout << "Finish allocating buffers ...\n";

  //* Set the Kernel Arguments
  int arg_counter = 0;
  OCL_CHECK(err,
            err = kernel_gcow.setArg(arg_counter++, buffer_in_gradients));
  OCL_CHECK(err,
            err = kernel_gcow.setArg(arg_counter++, buffer_out_gradients));

  //* Copy input data to device global memory
  OCL_CHECK(err,
  err = q.enqueueMigrateMemObjects({
    //* Input data objects
    buffer_in_gradients,
    buffer_out_gradients
  }, 0 /* 0: from host to device */));

  std::cout << "Launching kernel...\n";
  //* Launch the Kernel
  auto start = std::chrono::high_resolution_clock::now();
  OCL_CHECK(err,
            err = q.enqueueTask(kernel_gcow));

  //* Copy Result from Device Global Memory to Host Local Memory
  OCL_CHECK(err,
            err = q.enqueueMigrateMemObjects({buffer_out_gradients},
                  CL_MIGRATE_MEM_OBJECT_HOST /* 1: from device to host */));
  q.finish();

  auto end = std::chrono::high_resolution_clock::now();
  double duration = (std::chrono::duration_cast<std::chrono::milliseconds>
                     (end-start).count() / 1000.0);

  std::cout << "Duration (including memcpy out): " << duration << " seconds" <<
            std::endl;
  std::cout << "Overall grad values per second = " << GRAD_BLOCK_SIZE / duration
            << std::endl;

  //* Validate against software implementation.
  //TODO: implement software ZFP.
  std::vector<int> sw_out_zfp_gradients(GRAD_BLOCK_SIZE);
  for(int i = 0; i < GRAD_BLOCK_SIZE; i++) {
    sw_out_zfp_gradients[i] = (int) in_fp_gradients[i];
  }

  bool match = true;
  for(int i = 0; i < GRAD_BLOCK_SIZE; i++) {
    if (out_zfp_gradients[i] != sw_out_zfp_gradients[i]) {
      std::cout << "ERROR at [" << i << "] HW: " << out_zfp_gradients[i] << " != SW: "
                << sw_out_zfp_gradients[i] << std::endl;
      match = false;
      break;
    }
  }

  std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;
  return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}
