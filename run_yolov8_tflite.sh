#!/bin/bash
MLDIR=INSTALL_LOC
export LD_LIBRARY_PATH=$MLDIR/armnn:$MLDIR/armnn/delegate
source $MLDIR/venv/bin/activate && python $MLDIR/yolov8_tflite.py --model $MLDIR/yolov8n_int8.tflite --delegate
