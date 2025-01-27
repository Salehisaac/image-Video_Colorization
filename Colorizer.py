import numpy as np
import cv2
import os
import time
from os.path import splitext, basename, join

class Colorizer:
    def __init__(self, height=480, width=600):
        # Initialize the height and width for resizing images/videos
        (self.height, self.width) = height, width

        # Load the colorization model architecture and weights using OpenCV DNN module
        self.colorModel = cv2.dnn.readNetFromCaffe(
            "model/colorization_deploy_v2.prototxt", 
            caffeModel="model/colorization_release_v2.caffemodel"
        )

        # Load cluster centers for color quantization and reshape for model compatibility
        clusterCenters = np.load("model/pts_in_hull.npy")
        clusterCenters = clusterCenters.transpose().reshape(2, 313, 1, 1)

        # Assign the cluster centers to the appropriate layer in the model
        self.colorModel.getLayer(self.colorModel.getLayerId('class8_ab')).blobs = [clusterCenters.astype(np.float32)]
        
        # Assign a fixed prior factor for the colorization model
        self.colorModel.getLayer(self.colorModel.getLayerId('conv8_313_rh')).blobs = [np.full((1, 313), 2.606, dtype=np.float32)]

    def processVideo(self, videoName):
        # Open the video file
        cap = cv2.VideoCapture(videoName)
        if not cap.isOpened():
            print(f"Error opening video file: {videoName}")
            return

        # Prepare the output directory and define the output file path
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = join(output_dir, splitext(basename(videoName))[0] + '.avi')

        # Define the codec and output video parameters
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the input video
        frame_size = (self.width * 2, self.height)  # Double the width to show original and colorized side-by-side

        # Initialize the VideoWriter to save the processed video
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        # Track the processing time for frames
        prevFrameTime = time.time()

        while True:
            # Read the next frame from the video
            success, self.img = cap.read()
            if not success:
                print("Finished processing video or error reading frame.")
                break

            # Resize the frame to match the model's input dimensions
            self.img = cv2.resize(self.img, (self.width, self.height))

            # Process the current frame for colorization
            self.processFrame()

            # Combine the original and colorized frames side-by-side
            self.imgFinal = np.hstack((self.img, self.imgOut))

            # Save the processed frame to the output video
            out.write(self.imgFinal)

            # Display the current processed frame in a window
            cv2.imshow("Colorized Video", self.imgFinal)

            # Allow user to interrupt the video processing with 'q' key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Video processing interrupted by user.")
                break

        # Release video resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def processImage(self, imgName):
        # Load the input image
        self.img = cv2.imread(imgName)
        self.img = cv2.resize(self.img, (self.width, self.height))  # Resize to model-compatible dimensions

        # Process the image for colorization
        self.processFrame()

        # Save the processed image in the output directory
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(imgName))
        cv2.imwrite(output_path, self.imgFinal)

        # Display the processed image
        cv2.imshow("Colorized Image", self.imgFinal)
        cv2.waitKey(0)  # Wait for user input to close the window
        cv2.destroyAllWindows()

    def processFrame(self):
        # Normalize the input image to [0, 1] and convert BGR to RGB
        imgNormalized = (self.img[:, :, [2, 1, 0]] * 1.0 / 255).astype(np.float32)

        # Convert the normalized image to LAB color space and extract the L channel
        imgLab = cv2.cvtColor(imgNormalized, cv2.COLOR_RGB2LAB)
        channelL = imgLab[:, :, 0]

        # Resize the LAB image to 224x224 (model input size) and normalize the L channel
        imgLabResized = cv2.cvtColor(cv2.resize(imgNormalized, (224, 224)), cv2.COLOR_RGB2LAB)
        channelLResized = imgLabResized[:, :, 0]
        channelLResized -= 50  # Normalize L channel for model input

        # Prepare the model input blob and forward pass through the model
        self.colorModel.setInput(cv2.dnn.blobFromImage(channelLResized))
        result = self.colorModel.forward()[0, :, :, :].transpose((1, 2, 0))  # Get the predicted AB channels

        # Resize the predicted AB channels to the original image size
        resultResize = cv2.resize(result, (self.width, self.height))

        # Merge the L channel with the predicted AB channels
        self.imgOut = np.concatenate((channelL[:, :, np.newaxis], resultResize), axis=2)
        
        # Convert the LAB image back to BGR color space and clip pixel values to [0, 1]
        self.imgOut = np.clip(cv2.cvtColor(self.imgOut, cv2.COLOR_Lab2BGR), 0, 1)

        # Scale pixel values back to [0, 255] and convert to uint8
        self.imgOut = np.array(self.imgOut * 255, dtype=np.uint8)
        self.imgFinal = self.imgOut
