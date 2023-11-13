#ifndef GCOW_HPP
#define GCOW_HPP

#include "types.hpp"

//* Since the the host and C++ kernel code are developed and compiled independently,
//* wrap the kernel function declaration with the extern “C” linkage specifier to
//* prevent C++ compiler from performing name mangling.
extern "C" {

  void gcow(
    const size_t in_dim,
    const size_t *in_shape,
    const float *in_fp_gradients,
    volatile stream_word *out_zfp_gradients,
    size_t *out_bytes);

}

#endif // GCOW_HPP