import cv2
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QPlainTextEdit, QFileDialog

class FacialRecognition_GUI():

 def __init__(self):
    self.app = QApplication(sys.argv)

    self.main_window = QMainWindow()
    self.main_window.setWindowTitle("Facial Recognition")
    self.main_window.setGeometry(400, 400, 400, 200)

    self.photo_filepath = QPlainTextEdit(self.main_window)
    self.photo_filepath.setGeometry(3, 30, 240, 30)
    self.photo_filepath.move(20, 50)

    # Crearea butonului
    self.push_button_upload = QPushButton("Upload Photo", self.main_window)

    # Crearea butonului
    self.push_button_fr = QPushButton("Facial Recognition", self.main_window)
    self.push_button_fr.setGeometry(30, 60, 120, 30)
    self.push_button_fr.move(70, 120)

    # Crearea butonului
    self.push_button_save = QPushButton("Save", self.main_window)
    self.push_button_save.move(230, 120)

    # read the desired image
    self.original_image = ""
    self.output_image = ""

    def FileOpen():
        file = QFileDialog.getOpenFileName(None, 'Open File', "D:\Project_FacialRecognition_uP")
        file_list = list(file)
        self.original_image = Load_Image(file_list[0])
        self.output_image = Load_Image(file_list[0])
        print(self.original_image)
        return file_list[0]


    def FilePath():
        self.photo_filepath.clear()
        self.photo_filepath.insertPlainText(FileOpen())

    def Load_Image(image_path):
        # read the desired image
        print("Loading image...")
        original_image = cv2.imread(image_path)

        return original_image

    def Preprocess_Image(original_image):
        # Resize the image to 300*300 for performing predictions with the trained model
        original_image_resized = cv2.resize(original_image, (300, 300))

        # Perform mean prediction and transforms the image into a 4-dimensional array
        blob = cv2.dnn.blobFromImage(original_image_resized, 1.0, (300, 300), (104.0, 177.0, 123.0))

        return blob

    def Load_PretrainedModel():
        # load the model architecture and weights
        print("Loading model architecture...")

        # https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
        architecture_path = "D:\Project_FacialRecognition_uP/deploy.prototxt.txt"

        # https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
        model_path = "D:\Project_FacialRecognition_uP/res10_300x300_ssd_iter_140000_fp16.caffemodel"

        ResNet_model = cv2.dnn.readNetFromCaffe(architecture_path, model_path)

        return ResNet_model

    # pass the image through the network and obtain the detections and predictions
    def get_ModelDetections(ResNet_model, blob):

        print("Computing object detections...")

        # Set the image into the input of the neural network
        ResNet_model.setInput(blob)

        # Perform predictions and get the result
        detections = ResNet_model.forward()

        return detections

    def extract_PredictionProbability(detections):

        confidence = []

        # loop over the detections
        print("Looping over object detections...")
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence.append(detections[0, 0, i, 2])

        return confidence

    def draw_rectange(original_image, confidence, detections):
        # Draw Rectangle around image if detected
        print("Draw rectangle...")

        output_image = original_image
        h, w = output_image.shape[:2]

        for i in range(0, detections.shape[2]):

            # If confidence (detection probability) is above 70%, then draw the face surrounding box
            if confidence[i] > 0.7:
                # compute the (x, y)-coordinates of the rectangle shape for the face in image and upscale it to original image
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

                # Convert coordinates to integers
                startX, startY, endX, endY = box.astype("int")

                # The bounding box of the face along with the associated probability
                text = "{:.2f}%".format(confidence[i] * 100)

                y = startY - 10 if startY - 10 > 10 else startY + 10

                # Draw the rectangle surrounding the face
                cv2.rectangle(output_image, (startX, startY), (endX, endY), color=(0, 255, 0), thickness=4)

                # Draw the detection probability
                cv2.putText(output_image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color=(0, 0, 255),
                            thickness=2)

        return output_image


    def display_Images(original_image, output_image):
        # show results
        cv2.namedWindow("Original Image", cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Original Image", original_image)

        cv2.namedWindow("Output Image", cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Output Image", output_image)

        cv2.waitKey(0)


    def FaceDetection():

        blob = Preprocess_Image(self.original_image)

        ResNet_model = Load_PretrainedModel()

        detections = get_ModelDetections(ResNet_model, blob)

        confidence = extract_PredictionProbability(detections)

        self.output_image = draw_rectange(self.output_image, confidence, detections)

        display_Images(self.original_image, self.output_image)


    def save_OutputImage():
        # save the image with face detection
        print("Save....")
        cv2.imwrite("D:\Project_FacialRecognition_uP\Face_Detection_Image.jpg", self.output_image)


    self.push_button_upload.clicked.connect(FilePath)
    self.push_button_fr.clicked.connect(FaceDetection)
    self.push_button_save.clicked.connect(save_OutputImage)

    self.main_window.show()


 def getApp(self):
     return self.app


if __name__ == '__main__':

 GUI = FacialRecognition_GUI()
 app = GUI.getApp()

 sys.exit(app.exec_())