# gCOW: *g*radient *C*ompression *O*n the *W*ire

## Hardware

```bash
make all KERNEL=<gcow|a_test_kernel> TARGET=<hw_emu|hw> PLATFORM=xilinx_u250_gen3x16_xdma_4_1_202210_1
```

## Software

```bash
make clean && make <run_zfp_test|run_encoder_test|...>
``` 
