#include "types.h"


size_t zfp_compress(zfp_output *output, const zfp_input *input);
void zfp_compress_2d(zfp_output *output, const zfp_input *input);
size_t zfp_decompress(zfp_output *output, const zfp_input *input);
void zfp_decompress_2d(zfp_output *output, const zfp_input *input);