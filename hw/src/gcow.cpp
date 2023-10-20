#include "constants.hpp"

extern "C" {

void gcow(  
    const double *in_fp_gradients,
    int *out_zfp_gradients
) 
{
//* Bundle input and output to the same bus for now.
#pragma HLS INTERFACE m_axi port=in_fp_gradients offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=out_zfp_gradients offset=slave bundle=gmem0

    for(int i=0; i < GRAD_BLOCK_SIZE; i++) {
        // TODO
        out_zfp_gradients[i] = (int) in_fp_gradients[i];
    }
}

}
