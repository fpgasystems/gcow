#ifndef GCOW_HPP
#define GCOW_HPP


//* Since the the host and C++ kernel code are developed and compiled independently, 
//* wrap the kernel function declaration with the extern “C” linkage specifier to 
//* prevent C++ compiler from performing name mangling.
extern "C" {

  void gcow(
    const double *in_fp_gradients,
    int *out_zfp_gradients
  );

}

#endif // GCOW_HPP