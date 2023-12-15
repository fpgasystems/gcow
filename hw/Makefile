SHELL := bash

#* Primary build targets
.PHONY: xclbin host zfp_test run_zfp_test run_emax_test all run help clean cleanall format

# e.g., 
# time make all TARGET=hw PLATFORM=xilinx_u250_gen3x16_xdma_4_1_202210_1 > out 2>&1

help:
	@echo "Makefile Usage:"
	@echo '  make all TARGET=<sw_emu|hw_emu|hw> PLATFORM=<FPGA platform>'
	@echo "      Command to generate the design for specified Target and Device."
	@echo ""
	@echo "  make xclbin TARGET=<sw_emu|hw_emu|hw> PLATFORM=<FPGA platform>"
	@echo "      Command compile just the kernel of the design for specified Target and Device."
	@echo ""
	@echo "  make exe TARGET=<sw_emu|hw_emu|hw> PLATFORM=<FPGA platform>"
	@echo "      Command to generate the host application for specified Target and Device."
	@echo ""
	@echo "  make run TARGET=<sw_emu|hw_emu|hw> PLATFORM=<FPGA platform>"
	@echo "      Command compile just the kernel of the design for specified Target and Device."
	@echo ""
	@echo "  make clean "
	@echo "      Command to remove the generated non-hardware files."
	@echo ""
	@echo "  make cleanall"
	@echo "      Command to remove all the generated files."
	@echo ""

# VPP := $(XILINX_VITIS)/bin/v++
VPP := v++
EMCONFIGUTIL := $(XILINX_VITIS)/bin/emconfigutil
TARGET := hw_emu
PLATFORM := xilinx_u250_gen3x16_xdma_4_1_202210_1

#* Directories
SRC_DIR := src
INCLUDE_DIR := include
XILINX_VITIS_HLS := /tools/Xilinx/Vitis_HLS/2022.1
XCLBIN_DIR := ./xclbin
TEMP_DIR := ./_x.$(TARGET)
TEST_DIR := ./tests

#* Sources
KERNEL_SRC = $(wildcard $(SRC_DIR)/*.cpp stages/*.cpp)
#* (Use immediate expansion `:=` to avoid re-evaluation of `wildcard`.)
KERNEL_SRC := $(filter-out $(SRC_DIR)/host.cpp,$(KERNEL_SRC))

HOST_SRC := $(SRC_DIR)/host.cpp $(SRC_DIR)/common.cpp
TEST_ZFP_SRC := $(TEST_DIR)/test_zfp.cpp $(SRC_DIR)/common.cpp
TEST_EMAX_SRC := $(TEST_DIR)/test_emax.cpp $(KERNEL_SRC)
TEST_CAST_SRC := $(TEST_DIR)/test_cast.cpp $(KERNEL_SRC)
TEST_DECORRELATE_SRC := $(TEST_DIR)/test_decorrelate.cpp $(KERNEL_SRC)
TEST_REORDER_SRC := $(TEST_DIR)/test_reorder.cpp $(KERNEL_SRC)
TEST_BITPLANE_SRC := $(TEST_DIR)/test_bitplane.cpp $(KERNEL_SRC)
TEST_ENCBLOCK_SRC := $(TEST_DIR)/test_encblock.cpp $(KERNEL_SRC)
TEST_GATHER_SRC := $(TEST_DIR)/test_gather.cpp $(KERNEL_SRC)
TEST_WRITES_SRC := $(TEST_DIR)/test_writes.cpp $(KERNEL_SRC)
TEST_TRANSPOSE_SRC := $(TEST_DIR)/test_transpose.cpp $(KERNEL_SRC)
TEST_EMBED_SRC := $(TEST_DIR)/test_embed.cpp $(KERNEL_SRC)

#* Targets
KERNEL := gcow
HOST_EXE := $(KERNEL)_host
TEST_ZFP_EXE := zfp_test
TEST_EMAX_EXE := emax_test
TEST_CAST_EXE := cast_test
TEST_DECORRELATE_EXE := decorrelate_test
TEST_REORDER_EXE := reorder_test
TEST_BITPLANE_EXE := bitplane_test
TEST_ENCBLOCK_EXE := encblock_test
TEST_GATHER_EXE := gather_test
TEST_WRITES_EXE := writes_test
TEST_TRANSPOSE_EXE := transpose_test
TEST_EMBED_EXE := embed_test

XOS := $(XCLBIN_DIR)/$(KERNEL).$(TARGET).xo
XCLBIN := $(XCLBIN_DIR)/$(KERNEL).$(TARGET).xclbin
EMCONFIG_FILE := ./emconfig.json

VPP_COMMON_OPTS := -g -t $(TARGET) --platform $(PLATFORM) -I$(INCLUDE_DIR) --config configs/$(KERNEL)_config.cfg --save-temps 
CFLAGS := -g -std=c++11 -I$(XILINX_XRT)/include -I$(XILINX_VITIS_HLS)/include -I$(INCLUDE_DIR)
LFLAGS := -L$(XILINX_XRT)/lib -lxilinxopencl -pthread -lrt
NUMDEVICES := 1

xclbin: $(XCLBIN)

host: $(HOST_EXE)

all: host xclbin $(EMCONFIG_FILE)

#*  Compile kernels into Xilinx object file.
$(XOS): $(KERNEL_SRC)
# $(RM) $@
	mkdir -p $(XCLBIN_DIR)
	$(VPP) $(VPP_COMMON_OPTS) -c -k $(KERNEL) --temp_dir $(TEMP_DIR) -o $@ $+

#*  Link kernel object files into Xilinx binary file.
$(XCLBIN): $(XOS)
	mkdir -p $(XCLBIN_DIR)
	$(VPP) $(VPP_COMMON_OPTS) -l --temp_dir $(TEMP_DIR) -o $@ $+ # $(VPP_LINK_OPTS)

#*  Compile host application written in C++ into executable.
$(HOST_EXE): $(HOST_SRC)
	rm -f $@
	g++ $(CFLAGS) -o $@ $+ $(LFLAGS)
	@echo "Compiled Host Executable: $(HOST_EXE)"

$(TEST_EMAX_EXE): $(TEST_EMAX_SRC)
	rm -f $@
	g++ $(CFLAGS) -o $@ $+ $(LFLAGS)
	@echo "Compiled Test Executable: $(TEST_EMAX_EXE)"

$(TEST_CAST_EXE): $(TEST_CAST_SRC)
	rm -f $@
	g++ $(CFLAGS) -o $@ $+ $(LFLAGS)
	@echo "Compiled Test Executable: $(TEST_CAST_EXE)"

$(TEST_DECORRELATE_EXE): $(TEST_DECORRELATE_SRC)
	rm -f $@
	g++ $(CFLAGS) -o $@ $+ $(LFLAGS)
	@echo "Compiled Test Executable: $(TEST_DECORRELATE_EXE)"

$(TEST_REORDER_EXE): $(TEST_REORDER_SRC)
	rm -f $@
	g++ $(CFLAGS) -o $@ $+ $(LFLAGS)
	@echo "Compiled Test Executable: $(TEST_REORDER_EXE)"

$(TEST_BITPLANE_EXE): $(TEST_BITPLANE_SRC)
	rm -f $@
	g++ $(CFLAGS) -o $@ $+ $(LFLAGS)
	@echo "Compiled Test Executable: $(TEST_BITPLANE_EXE)"

$(TEST_ENCBLOCK_EXE): $(TEST_ENCBLOCK_SRC)
	rm -f $@
	g++ $(CFLAGS) -o $@ $+ $(LFLAGS)
	@echo "Compiled Test Executable: $(TEST_ENCBLOCK_EXE)"

$(TEST_GATHER_EXE): $(TEST_GATHER_SRC)
	rm -f $@
	g++ $(CFLAGS) -o $@ $+ $(LFLAGS)
	@echo "Compiled Test Executable: $(TEST_GATHER_EXE)"

$(TEST_WRITES_EXE): $(TEST_WRITES_SRC)
	rm -f $@
	g++ $(CFLAGS) -o $@ $+ $(LFLAGS)
	@echo "Compiled Test Executable: $(TEST_WRITES_EXE)"

$(TEST_TRANSPOSE_EXE): $(TEST_TRANSPOSE_SRC)
	rm -f $@
	g++ $(CFLAGS) -o $@ $+ $(LFLAGS)
	@echo "Compiled Test Executable: $(TEST_TRANSPOSE_EXE)"

$(TEST_EMBED_EXE): $(TEST_EMBED_SRC)
	rm -f $@
	g++ $(CFLAGS) -o $@ $+ $(LFLAGS)
	@echo "Compiled Test Executable: $(TEST_EMBED_EXE)"

$(TEST_ZFP_EXE): $(TEST_ZFP_SRC)
	rm -f $@
	g++ $(CFLAGS) -o $@ $+ $(LFLAGS)
	@echo "Compiled Test Executable: $(TEST_ZFP_EXE)"

$(EMCONFIG_FILE):
	$(EMCONFIGUTIL) --nd $(NUMDEVICES) --od . --platform $(PLATFORM)

run_emax_test: $(XCLBIN) $(TEST_EMAX_EXE) $(EMCONFIG_FILE)
#^ With `KERNEL=emax`
	export XCL_EMULATION_MODE=hw_emu; ./$(TEST_EMAX_EXE) $(XCLBIN)

run_cast_test: $(XCLBIN) $(TEST_CAST_EXE) $(EMCONFIG_FILE)
#^ With `KERNEL=cast`
	export XCL_EMULATION_MODE=hw_emu; ./$(TEST_CAST_EXE) $(XCLBIN)

run_decorrelate_test: $(XCLBIN) $(TEST_DECORRELATE_EXE) $(EMCONFIG_FILE)
#^ With `KERNEL=decorrelate`
	export XCL_EMULATION_MODE=hw_emu; ./$(TEST_DECORRELATE_EXE) $(XCLBIN)

run_reorder_test: $(XCLBIN) $(TEST_REORDER_EXE) $(EMCONFIG_FILE)
#^ With `KERNEL=reorder`
	export XCL_EMULATION_MODE=hw_emu; ./$(TEST_REORDER_EXE) $(XCLBIN)

run_bitplane_test: $(XCLBIN) $(TEST_BITPLANE_EXE) $(EMCONFIG_FILE)
#^ With `KERNEL=bitplane`
	export XCL_EMULATION_MODE=hw_emu; ./$(TEST_BITPLANE_EXE) $(XCLBIN)

run_encblock_test: $(XCLBIN) $(TEST_ENCBLOCK_EXE) $(EMCONFIG_FILE)
#^ With `KERNEL=encblock`
	export XCL_EMULATION_MODE=hw_emu; ./$(TEST_ENCBLOCK_EXE) $(XCLBIN)

run_gather_test: $(XCLBIN) $(TEST_GATHER_EXE) $(EMCONFIG_FILE)
#^ With `KERNEL=gather`
	export XCL_EMULATION_MODE=hw_emu; ./$(TEST_GATHER_EXE) $(XCLBIN)

run_writes_test: $(XCLBIN) $(TEST_WRITES_EXE) $(EMCONFIG_FILE)
#^ With `KERNEL=writes`
	export XCL_EMULATION_MODE=hw_emu; ./$(TEST_WRITES_EXE) $(XCLBIN)

run_transpose_test: $(XCLBIN) $(TEST_TRANSPOSE_EXE) $(EMCONFIG_FILE)
#^ With `KERNEL=transpose`
	export XCL_EMULATION_MODE=hw_emu; ./$(TEST_TRANSPOSE_EXE) $(XCLBIN)

run_embed_test: $(XCLBIN) $(TEST_EMBED_EXE) $(EMCONFIG_FILE)
#^ With `KERNEL=embed`
	export XCL_EMULATION_MODE=hw_emu; ./$(TEST_EMBED_EXE) $(XCLBIN)

run_zfp_test: $(XCLBIN) $(TEST_ZFP_EXE) $(EMCONFIG_FILE)
	export XCL_EMULATION_MODE=hw_emu; ./$(TEST_ZFP_EXE) $(XCLBIN)

run: $(XCLBIN) $(HOST_EXE) $(EMCONFIG_FILE)
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	export XCL_EMULATION_MODE=${TARGET}; ./$(HOST_EXE) $(XCLBIN)
else
	./$(HOST_EXE) $(XCLBIN)
endif

clean:
	-$(RM) -rf $(EMCONFIG_FILE) $(HOST_EXE) $(XCLBIN) *.xclbin *.xo $(XOS) *.log *.csv *summary *.json *.xml **/*.orig .nfs*
	
cleanall: clean
	-$(RM) -rf _x.* .Xil .run
	-$(RM) -rf $(XCLBIN_DIR)

format:
	astyle --style=linux --indent=spaces=2 -S --max-code-length=80 src/*.cpp include/*.hpp stages/*.cpp tests/*.cpp