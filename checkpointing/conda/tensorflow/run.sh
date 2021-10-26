#!/bin/bash

#If failure, exit with non-zero
set -e

#Establish environment name
ENVNAME=tf_checkpointing

#Environment directory
ENVDIR=$ENVNAME

#Set up environment:
export PATH

#We must only set up this directory on the first checkpoint run
if [ ! -d "$ENVDIR" ] 
then
	mkdir $ENVDIR
	tar -xzf $ENVNAME.tar.gz -C $ENVDIR
fi	

. $ENVDIR/bin/activate

#Set up home
export HOME=$_CONDOR_SCRATCH_DIR

#Run the program
python3 tf_checkpointing.py

