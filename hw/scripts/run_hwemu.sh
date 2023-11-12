#!/usr/bin/env bash
# emconfigutil --platform xilinx_u250_gen3x16_xdma_4_1_202210_1
export  XCL_EMULATION_MODE=hw_emu
./gcow_host ./xclbin/gcow.hw_emu.xclbin