#$ vitis_hls -f scripts/synthesis.tcl
#* List all the source files
set source_files [glob -nocomplain src/*.cpp]
set header_files [glob -nocomplain include/*.hpp]

#* Exclude specific files
set excluded_sources [list src/host.cpp]
set excluded_headers [list include/host.hpp]

#* Remove excluded files from the lists
foreach file $excluded_sources {
    set idx [lsearch -exact $source_files $file]
    if {$idx >= 0} {
        set source_files [lreplace $source_files $idx $idx]
    }
}

foreach file $excluded_headers {
    set idx [lsearch -exact $header_files $file]
    if {$idx >= 0} {
        set header_files [lreplace $header_files $idx $idx]
    }
}

#* Concatenate the source and header files
set all_files [concat $source_files $header_files]

#* Replace with the appropriate path in the add_files command
set files_to_add [join $all_files " "]

#* Define include directories
set include_dirs "-I./include -I/usr/include -I/local/home/honghe/xrt_2022.1/opt/xilinx/xrt/include -I/tools/Xilinx/Vitis_HLS/2022.1/include -I/tools/Xilinx/Vitis/2022.1/include -I/local/home/honghe/xrt_2022.1/opt/xilinx/xrt/include"

#* Start creating the HLS project and setting up parameters
open_project -reset hls-outputs 
open_solution xcu250-figd2104-2L-e

#* Add all the source and header files to the project
add_files -cflags "-std=c++11 $include_dirs" $files_to_add

#* Set the top module
set_top gcow 

#* Set the target part
set_part xcu250-figd2104-2L-e

#* Set clock period
create_clock -period 140MHz

#* Configure the interface
config_interface -m_axi_addr64

#* Synthesize the design
csynth_design

#* Exit HLS tool
close_project
exit
