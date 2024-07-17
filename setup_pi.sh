
#!/bin/bash
sudo apt update
sudo apt install -y git build-essential ffmpeg locate gstreamer1.0-plugins-bad gstreamer1.0-libav net-tools pipenv libncurses5-dev
sudo apt install -y libssl-dev nano htop python3-opencv tmux cmake libopencv-dev python3-picamera2 gstreamer1.0-tools libffi-dev libprotobuf-dev
sudo apt install -y fd-find git-lfs python3-pip python3-zmq cpufrequtils

echo "alias ll='ls -al'" >> ~/.profile

# Get fd, because find sucks
sudo ln -s /usr/bin/fdfind /usr/bin/fd

# Setup wpilib
# Do this in a virtualenv
virtualenv -p python3 venv
source venv/bin/activate
pip install numpy==1.26.4 zmq opencv-python tflite_runtime
wget https://github.com/ARM-software/armnn/releases/download/v24.05/ArmNN-linux-aarch64.tar.gz
mkdir armnn
cd armnn
tar -xf ../ArmNN-linux-aarch64.tar.gz

sudo cp systemd/*.service /lib/systemd/system
sudo systemctl daemon-reload
sudo systemctl enable zmqcam
sudo systemctl enable yolov8_tflite


