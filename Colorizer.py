import numpy as np
import cv2
import os
import time
from os.path import splitext, basename, join

class Colorizer:
    def __init__(self, height=480, width=600):
        (self.height, self.width) = height, width

        
        self.colorModel = cv2.dnn.readNetFromCaffe(
            "model/colorization_deploy_v2.prototxt", 
            caffeModel="model/colorization_release_v2.caffemodel"
        )

        
        clusterCenters = np.load("model/pts_in_hull.npy")
        clusterCenters = clusterCenters.transpose().reshape(2, 313, 1, 1)

        
        self.colorModel.getLayer(self.colorModel.getLayerId('class8_ab')).blobs = [clusterCenters.astype(np.float32)]
        self.colorModel.getLayer(self.colorModel.getLayerId('conv8_313_rh')).blobs = [np.full((1, 313), 2.606, dtype=np.float32)]


    def processVideo(self, videoName):
        cap = cv2.VideoCapture(videoName)
        if not cap.isOpened():
            print(f"Error opening video file: {videoName}")
            return

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = join(output_dir, splitext(basename(videoName))[0] + '.avi')

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_size = (self.width * 2, self.height) 

        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        prevFrameTime = time.time()

        while True:
            success, self.img = cap.read()
            if not success:
                print("Finished processing video or error reading frame.")
                break

            
            self.img = cv2.resize(self.img, (self.width, self.height))

           
            self.processFrame()

            
            self.imgFinal = np.hstack((self.img, self.imgOut))


            out.write(self.imgFinal)

            cv2.imshow("Colorized Video", self.imgFinal)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Video processing interrupted by user.")
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()



    def processImage(self, imgName):
       
        self.img = cv2.imread(imgName)
        self.img = cv2.resize(self.img, (self.width, self.height))

        
        self.processFrame()

        
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(imgName))
        cv2.imwrite(output_path, self.imgFinal)

        
        cv2.imshow("Colorized Image", self.imgFinal)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def processFrame(self):
        
        imgNormalized = (self.img[:, :, [2, 1, 0]] * 1.0 / 255).astype(np.float32)

        
        imgLab = cv2.cvtColor(imgNormalized, cv2.COLOR_RGB2LAB)
        channelL = imgLab[:, :, 0]

       
        imgLabResized = cv2.cvtColor(cv2.resize(imgNormalized, (224, 224)), cv2.COLOR_RGB2LAB)
        channelLResized = imgLabResized[:, :, 0]
        channelLResized -= 50

        
        self.colorModel.setInput(cv2.dnn.blobFromImage(channelLResized))
        result = self.colorModel.forward()[0, :, :, :].transpose((1, 2, 0))

        
        resultResize = cv2.resize(result, (self.width, self.height))

        
        self.imgOut = np.concatenate((channelL[:, :, np.newaxis], resultResize), axis=2)
        self.imgOut = np.clip(cv2.cvtColor(self.imgOut, cv2.COLOR_Lab2BGR), 0, 1)

        
        self.imgOut = np.array(self.imgOut * 255, dtype=np.uint8)
        self.imgFinal = self.imgOut
