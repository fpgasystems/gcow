# gCOW: *g*radient *C*ompression *O*n the *W*ire

## Hardware

```bash
make all TARGET=hw_emu/hw PLATFORM=xilinx_u250_gen3x16_xdma_4_1_202210_1
./run_hwemu.sh
```

## Software

```bash
make clean && make test
``` 