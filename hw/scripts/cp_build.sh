#!/usr/bin/env bash

rm -rf /scratch/honghe/gcow/hw
mkdir /scratch/honghe/gcow/hw
rsync -av --exclude='.ipcache' --exclude='_x.*' --exclude='.run' --exclude='*.log' --exclude='.Xil' --exclude='xclbin' --exclude='.nfs*' --exclude='*.csv' ./ /scratch/honghe/gcow/hw/