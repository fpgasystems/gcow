#ifndef HOST_HPP
#define HOST_HPP

#include <iostream>
#include <vector>
#include <fstream>

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl2.hpp>
#include <CL/cl_ext_xilinx.h>

//* Error checking in OpenCL code
#define OCL_CHECK(error,call)                                       \
    call;                                                           \
    if (error != CL_SUCCESS) {                                      \
      printf("%s:%d Error calling " #call ", error code is: %d\n",  \
              __FILE__,__LINE__, error);                            \
      exit(EXIT_FAILURE);                                           \
    }

std::string xclbin_file_name;

template <typename T>
struct aligned_allocator
{
    using value_type = T;
    T* allocate(std::size_t num)
    {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, 4096, num*sizeof(T)))
            throw std::bad_alloc();
        return reinterpret_cast<T*>(ptr);
    }
    void deallocate(T* p, std::size_t num)
    {
        free(p);
    }
};


cl::Program::Binaries import_binary_file()
{
    std::cout << "\nLoading: '"<< xclbin_file_name.c_str() << "'\n";
    std::ifstream bin_file(xclbin_file_name.c_str(), std::ifstream::binary);
    //* Move cursor to the end of the bitstream file.
    bin_file.seekg(0, bin_file.end);
    //* Get the position of the current character in the stream.
    unsigned nb = bin_file.tellg();
    //* Move cursor to the beginning of the bitstream file.
    bin_file.seekg(0, bin_file.beg);
    std::cout << "Bitstream size (MiB): " << (nb - bin_file.tellg() + 1) / 1024.0 / 1024 << std::endl;
    //* Create a vector of char to store the content of the file.
    char *buf = new char [nb];
    //* Read the next `nb` bytes from the file into the buffer.
    bin_file.read(buf, nb);

    cl::Program::Binaries bins;
    bins.push_back({buf, nb});
    return bins;
}

std::vector<cl::Device> get_devices()
{

    size_t i;
    cl_int err;
    std::vector<cl::Platform> platforms;
    OCL_CHECK(err, err = cl::Platform::get(&platforms));
    cl::Platform platform;
    for (i  = 0 ; i < platforms.size(); i++) {
        platform = platforms[i];
        OCL_CHECK(err, std::string platformName = platform.getInfo<CL_PLATFORM_NAME>(&err));
        if (platformName == "Xilinx") {
            std::cout << "\nFound Platform" << std::endl;
            std::cout << "\nPlatform Name: " << platformName.c_str() << std::endl;
            break;
        }
    }
    if (i == platforms.size()) {
        std::cout << "Error: Failed to find Xilinx platform" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Getting ACCELERATOR Devices and selecting 1st such device
    std::vector<cl::Device> devices;
    OCL_CHECK(err, err = platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices));
    return devices;
}

#endif // HOST_HPP