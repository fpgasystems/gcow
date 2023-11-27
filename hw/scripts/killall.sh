#! /bin/bash
ps aux | grep xilinx | grep honghe | awk '{print $2}' | xargs -i kill -9 {}  