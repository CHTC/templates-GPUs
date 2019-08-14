#!/bin/bash

export PATH=/usr/sbin:$PATH
echo 'Date: ' `date` 
echo 'Host: ' `hostname` 
echo 'System: ' `uname -spo` 
echo 'GPU: ' `lspci | grep NVIDIA`
