# yolo_tflite_pi_detector

This is an example project to setup a Raspberry Pi to capture images via it's camera
and use Yolov8 TFLite network to detect objects. The bounding box is put
around detections and streamed via a HTTP image server.

This repo is used in conjunction with this repo:
https://github.com/nlbutts/ultralytics

## Setup
Run the **setup_pi.sh** script.

## Manually run
Due to unknown reasons, you can't install picamera2 Python library in a virtual envionment. You also can't install
tflite_runtime without a Virtual environment. Perhaps some days of mysteries of this complex ecosystem will be revealed. 

To run manually, create two terminal windows. In one window run:
`python3 zmqcam.py`

You will see the FPS printed to the screen. There is no configuration to that script. It is hard coded to 10 FPS.

Now source your virtualenv and run the Yolo detector:
```
. venv/bin/activate
export LD_LIBRARY_PATH=`pwd`/armnn:`pwd`/armnn/delegate
python yolov8_tflite.py --delegate
```

On a RP4, this can acheive 2 FPS. Once you hit the HTTP server, it drops to about 1.2-1.3 FPS.

Now with a web browser navigate to `<raspberry pi ip address>:8000/stream.mjpg`

You should see a live stream and detections.

## Run automatically
The install script should setup the systemd services.
