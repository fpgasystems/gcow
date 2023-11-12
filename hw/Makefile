SHELL := bash

#* Primary build targets
.PHONY: xclbin exe all help clean cleanall run

# e.g., 
# time make all TARGET=hw PLATFORM=xilinx_u250_gen3x16_xdma_4_1_202210_1 > out 2>&1

help:
	@echo "Makefile Usage:"
	@echo '  make all TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform>'
	@echo "      Command to generate the design for specified Target and Device."
	@echo ""
	@echo "  make xclbin TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform>"
	@echo "      Command compile just the kernel of the design for specified Target and Device."
	@echo ""
	@echo "  make exe TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform>"
	@echo "      Command to generate the host application for specified Target and Device."
	@echo ""
	@echo "  make run TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform>"
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
SRC_DIR = src
INCLUDE_DIR = include
XCLBIN_DIR := ./xclbin
TEMP_DIR := ./_x.$(TARGET)

#* Sources
KERNEL_SRC := $(SRC_DIR)/gcow.cpp
HOST_SRC := $(SRC_DIR)/host.cpp

#* Targets
KERNEL = gcow
HOST_EXE := $(KERNEL)_host

XOS := $(XCLBIN_DIR)/gcow.$(TARGET).xo
XCLBIN := $(XCLBIN_DIR)/gcow.$(TARGET).xclbin
EMCONFIG_FILE := ./emconfig.json

VPP_COMMON_OPTS := -g -t $(TARGET) --platform $(PLATFORM) -I$(INCLUDE_DIR) --save-temps --config connectivity.cfg
CFLAGS := -g -std=c++11 -I$(XILINX_XRT)/include -I$(INCLUDE_DIR)
LFLAGS := -L$(XILINX_XRT)/lib -lxilinxopencl -pthread -lrt
NUMDEVICES := 1

# run time args
# EXE_OPT := gcow.$(TARGET).xclbin

xclbin: $(XCLBIN)

exe: $(HOST_EXE)

all: exe xclbin $(EMCONFIG_FILE)

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

$(EMCONFIG_FILE):
	$(EMCONFIGUTIL) --nd $(NUMDEVICES) --od . --platform $(PLATFORM)

run: $(XCLBIN) $(HOST_EXE) $(EMCONFIG_FILE)
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	export XCL_EMULATION_MODE=${TARGET}; ./$(HOST_EXE) $(XCLBIN)
else
	./$(HOST_EXE) $(XCLBIN)
endif

clean:
	-$(RM) -rf $(EMCONFIG_FILE) $(HOST_EXE) $(XCLBIN) *.xclbin *.xo $(XOS) *.log *.csv *summary *.json *.xml **/*.orig
	
cleanall: clean
	-$(RM) -rf _x.* .Xil .run
	-$(RM) -rf $(XCLBIN_DIR)
