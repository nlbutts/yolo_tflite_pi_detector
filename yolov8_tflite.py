# Ultralytics YOLO 🚀, AGPL-3.0 license

import argparse

import cv2
import numpy as np
from tflite_runtime import interpreter as tflite
import time
import zmq
import cv2
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import os
import sys

# Declare as global variables, can be updated based trained model image size
img_width = 320
img_height = 320

#os.environ['LD_LIBRARY_PATH'] = '/usr/lib/armnn:/usr/lib/armnn/delegate'
sys.path.append('/usr/lib/armnn')

class LetterBox:
    def __init__(
        self, new_shape=(img_width, img_height), auto=False, scaleFill=False, scaleup=True, center=True, stride=32
    ):
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left

    def __call__(self, labels=None, image=None):
        """Return updated labels and image with added border."""

        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # for evaluation

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels."""

        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels


class Yolov8TFLite:
    def __init__(self, tflite_model, input_image, confidence_thres, iou_thres, use_delegate, threads):
        """
        Initializes an instance of the Yolov8TFLite class.

        Args:
            tflite_model: Path to the TFLite model.
            input_image: Path to the input image.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """

        self.tflite_model = tflite_model
        self.input_image = input_image
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # Load the class names from the COCO dataset
        #self.classes = yaml_load(check_yaml("coco128.yaml"))["names"]
        self.classes = []
        for i in range(80):
            self.classes.append(i)

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

        self.use_delegate = use_delegate
        self.threads = threads
        self.preload()

    def draw_detections(self, img, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img,
            (int(label_x), int(label_y - label_height)),
            (int(label_x + label_width), int(label_y + label_height)),
            color,
            cv2.FILLED,
        )

        # Draw the label text on the image
        cv2.putText(img, label, (int(label_x), int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self, q, img):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """

        # Read the input image using OpenCV
        self.img = img

        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]

        letterbox = LetterBox(new_shape=[img_width, img_height], auto=False, stride=32)
        image = letterbox(image=self.img)
        image = [image]
        image = np.stack(image)
        image = image[..., ::-1].transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(image)
        # n, h, w, c
        img = img.astype(np.float32) / 255
        return img
        #scale = q[0]
        #zero_point = q[1]
        #img = (img / scale + zero_point).astype('int8')  # de-scale
        #return img

    def postprocess(self, input_image, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """

        input_image = input_image.copy()

        boxes = []
        scores = []
        class_ids = []
        for pred in output:
            pred = np.transpose(pred)
            for box in pred:
                x, y, w, h = box[:4]
                x1 = x - w / 2
                y1 = y - h / 2
                boxes.append([x1, y1, w, h])
                idx = np.argmax(box[4:])
                scores.append(box[idx + 4])
                class_ids.append(idx)

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            gain = min(img_width / self.img_width, img_height / self.img_height)
            pad = (
                round((img_width - self.img_width * gain) / 2 - 0.1),
                round((img_height - self.img_height * gain) / 2 - 0.1),
            )
            box[0] = (box[0] - pad[0]) / gain
            box[1] = (box[1] - pad[1]) / gain
            box[2] = box[2] / gain
            box[3] = box[3] / gain
            score = scores[i]
            class_id = class_ids[i]
            if score > 0.25:
            #if True:
                print(f'score: {score} class: {class_id} box: {box}')
                # Draw the detection on the input image
                self.draw_detections(input_image, box, score, class_id)

        return input_image

    def main(self, img):
        """
        Performs inference using a TFLite model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """

        img_data = self.preprocess(self.q, img)
        # Set the input tensor to the interpreter
        img_data = img_data.transpose((0, 2, 3, 1))

        self.interpreter.set_tensor(self.input_details[0]["index"], img_data)

        # Run inference
        self.interpreter.invoke()

        # Get the output tensor from the self.interpreter
        output = self.interpreter.get_tensor(self.output_details[0]["index"])
        scale, zero_point = self.output_details[0]["quantization"]
        #output = (output.astype(np.float32) - zero_point) * scale

        output[:, [0, 2]] *= img_width
        output[:, [1, 3]] *= img_height
        # Perform post-processing on the outputs to obtain output image.
        return self.postprocess(self.img, output)

    def preload(self):
        # Create an interpreter for the TFLite model
        print("Loaded model")
        if self.use_delegate:
            print('Using Delegate')
            delegate = tflite.load_delegate('armnn/delegate/libarmnnDelegate.so', options={"backends": "CpuAcc, CpuRef ", "logging-severity": "fatal"})
            self.interpreter = tflite.Interpreter(model_path=self.tflite_model,
                                                  num_threads=self.threads,
                                                  experimental_delegates=[delegate])
        else:
            self.interpreter = tflite.Interpreter(model_path=self.tflite_model, num_threads=self.threads)
        self.model = self.interpreter
        self.interpreter.allocate_tensors()

        # Get the model inputs
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Store the shape of the input for later use
        input_shape = self.input_details[0]["shape"]
        self.input_width = input_shape[1]
        self.input_height = input_shape[2]

        # Preprocess the image data
        self.q = self.input_details[0]['quantization']

class MJPEGStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        print("do_get called")
        if self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()

            while True:
                # Capture image from camera
                frame = get_frame()
                if frame is None:
                    break

                # Encode image to JPEG
                compression_params = [cv2.IMWRITE_JPEG_QUALITY, 50]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _, jpeg = cv2.imencode('.jpg', frame, compression_params)
                jpeg_bytes = jpeg.tobytes()

                # Send HTTP response with JPEG image
                self.wfile.write(b'--jpgboundary\r\n')
                self.send_header('Content-type', 'image/jpeg')
                self.send_header('Content-length', str(len(jpeg_bytes)))
                self.end_headers()
                self.wfile.write(jpeg_bytes)
                self.wfile.write(b'\r\n')

                # Add some delay to control the frame rate
                #time.sleep(0.05)

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')


def serve():
    httpd = HTTPServer(('0.0.0.0', 8000), MJPEGStreamHandler)
    print('HTTP server started on port 8000...')
    httpd.serve_forever()

if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="yolov8n_int8.tflite", help="Input your TFLite model."
    )
    parser.add_argument("--img", type=str, default="bus.jpg", help="Path to input image.")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--delegate", action='store_true', help='Use ARMNN Delegate')
    parser.add_argument("--threads", type=int, default=4, help='Number of threads to use')
    args = parser.parse_args()

    # Create an instance of the Yolov8TFLite class with the specified arguments
    detection = Yolov8TFLite(args.model, args.img, args.conf_thres, args.iou_thres, args.delegate, args.threads)

    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect("tcp://0.0.0.0:5555")
    subscriber.setsockopt_string(zmq.SUBSCRIBE, '')

    latest_img = None

    def grabImage():
        while True:
            global latest_img
            latest_img = subscriber.recv()

    output_image = None

    # Start the HTTP server in a separate thread
    server_thread = threading.Thread(target=serve)
    server_thread.start()

    grab_thread = threading.Thread(target=grabImage)
    grab_thread.start()

    start = time.time()
    frames = 0

    while True:
        if latest_img is not None:
            stop = time.time()
            if stop - start > 1:
                diff = stop - start
                fps = frames / diff
                start = stop
                frames = 0
                print(f'FPS: {fps}')

            img = np.frombuffer(latest_img, 'uint8')       
            img = np.reshape(img, (480, 640, 4))
            img = img[:, :, 0:3]
            output_image = detection.main(img)
            frames += 1

            def get_frame():
                # Capture frame from the video capture device
                return output_image

            #cv2.imwrite('input.jpg', img)
            #cv2.imwrite('output.jpg', output_image)

