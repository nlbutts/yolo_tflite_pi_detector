#!/bin/bash
MLDIR=INSTALL_LOC
cd $MLDIR
export LD_LIBRARY_PATH=$MLDIR/armnn:$MLDIR/armnn/delegate
source $MLDIR/venv/bin/activate && export LD_LIBRARY_PATH=$MLDIR/armnn:$MLDIR/armnn/delegate && python $MLDIR/yolov8_tflite.py --model $MLDIR/yolov8n_int8.tflite --delegate
