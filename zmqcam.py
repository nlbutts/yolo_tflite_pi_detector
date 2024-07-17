import zmq
import numpy as np
import cv2
from picamera2 import Picamera2
import time

def capture_image(camera_manager):
    with camera_manager.acquire() as camera:
        frame = camera.get_frame()
        # Convert the frame to an OpenCV image
        image = frame.as_opencv_image()
        return image

def compress_image(image):
    # Convert the image to JPEG format
    _, encoded_image = cv2.imencode(".jpg", image)
    return encoded_image

def main():
    # Initialize ZeroMQ publisher
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://*:5555")

    picam2 = Picamera2()
    # This line isn't used. The camera defaults to 640x480, which is great.
    #config = picam2.create_video_configuration(main={"size": (640, 480)}, lores={"size": (320,240)}, encode="lores")
    # Limit the framerate to 10 FPS
    picam2.video_configuration.controls.FrameRate = 10.0
    picam2.video_configuration.main.size = (640, 480)
    picam2.start('video')

    start = time.time()
    frames = 0

    while True:
        try:
            stop = time.time()
            if stop - start > 1:
                diff = stop - start
                fps = frames / diff
                start = stop
                frames = 0
                print(f'FPS: {fps}')

            frames += 1
            array = picam2.capture_array("main")

            # Publish the compressed image
            publisher.send(array.tobytes())

        except KeyboardInterrupt:
            print("Interrupted")
            publisher.close()

if __name__ == "__main__":
    main()
