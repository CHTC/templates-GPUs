#!/bin/bash

#!/bin/sh
echo 'Date: ' `date` 
echo 'Host: ' `hostname` 
echo 'System: ' `uname -spo` 
echo 'GPU: ' `lspci`
