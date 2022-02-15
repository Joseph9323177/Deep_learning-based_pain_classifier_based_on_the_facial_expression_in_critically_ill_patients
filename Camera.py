import time
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2.cv2 as cv2
import mediapipe as mp
import time
import math
import numpy as np
from numpy.linalg import inv
import torch
from torchvision import transforms
from PIL import Image
import pyqtgraph as pg

class Camera(QThread):
    Imageupdate = pyqtSignal(QImage)
    Cropped_Imageupdate = pyqtSignal(QImage)
    wave_plot_Imageupdate = pyqtSignal(QImage)
    wave_plot_list = pyqtSignal(list)
    face_capture_rate = pyqtSignal(int)
    def __init__(self, model, device, temp, camera_number):
        super(Camera, self).__init__()
        self.mp_face_mesh = mp.solutions.face_mesh
        no_face_backup = cv2.imread('./no_face.png')
        self.no_face_backup = cv2.cvtColor(no_face_backup, cv2.COLOR_BGR2RGB)
        self.model = model
        self.t0 = time.time()
        self.ThreadActivate = False
        self.device = device
        self.temp = temp
        self.camera_number = camera_number

    def reload_model(self, model, temp):
        self.model = None
        self.model = model
        self.temp = temp
        print('model change success')
        print('temp = ', temp)


    def run_prediction(self):
        cap = cv2.VideoCapture(self.camera_number,cv2.CAP_DSHOW)
        failure = 0
        count = 0
        with self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            while self.ThreadActivate:
                t0 = time.time()
                ret, frame = cap.read()
                # frame = cv2.flip(frame,0)
                cv2.waitKey(1)
                if not ret:
                    cv2.destroyAllWindows()
                    break
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_ = image.copy()
                count += 1
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)
                # Draw the face mesh annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if not results.multi_face_landmarks:
                    failure += 1
                    fps_message = f'FPS = {round((1 / ((time.time()+ 0.0001) - t0)), 2)}'
                    cv2.putText(image_, fps_message, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 1,
                                cv2.LINE_AA)
                    cv2.putText(image_, "No face", (250, 250), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 3, cv2.LINE_AA)
                    show_ = image_.copy()
                    backup = self.no_face_backup
                    ConvertQtformat = QImage(show_.data, show_.shape[1], show_.shape[0], QImage.Format_RGB888)
                    Pic = ConvertQtformat.scaled(640, 480, Qt.KeepAspectRatio)
                    self.Imageupdate.emit(Pic)
                    ConvertQtformat_cropped = QImage(backup.data, backup.shape[1], backup.shape[0],backup.shape[1]*3,
                                                     QImage.Format_RGB888)
                    Pic_crop = ConvertQtformat_cropped.scaled(320, 240, Qt.KeepAspectRatio)
                    self.wave_plot_list.emit([[0, 0, 0]])
                    self.Cropped_Imageupdate.emit(Pic_crop)
                    self.face_capture_rate.emit(int(((count - failure) / count) * 100))
                elif results.multi_face_landmarks:
                    key_point_list = [133, 362]
                    for i in range(468):
                        key_point_list.append(i)
                    relative_key_point_list = []
                    for key_point in key_point_list:
                        x = results.multi_face_landmarks[0].landmark[key_point].x
                        y = results.multi_face_landmarks[0].landmark[key_point].y
                        shape = image.shape
                        relative_x = int(x * shape[1])
                        relative_y = int(y * shape[0])
                        relative_key_point_list.append([relative_x, relative_y])
                    # Get rotation angle
                    cos = (relative_key_point_list[1][0] - relative_key_point_list[0][0]) / (
                            (relative_key_point_list[0][0] - relative_key_point_list[1][0]) ** 2 + (
                            relative_key_point_list[0][1] - relative_key_point_list[1][1]) ** 2) ** 0.5
                    # need to consider that -45 and 45 degree has the same cosine

                    if relative_key_point_list[1][1] - relative_key_point_list[0][1] < 0:
                        angle_rad = math.acos(cos)
                    else:
                        angle_rad = -(math.acos(cos))

                    sin = np.sin(angle_rad)
                    angle_deg = angle_rad * (180 / math.pi)
                    # Get rotation matrix for landmark
                    R = np.array(((cos, -sin), (sin, cos)))
                    # Rotate image
                    rotated_landmark = []
                    for point in relative_key_point_list:
                        rotated_point = np.matmul(R, point)
                        # rotated = cv2.circle(rotated, (int(rotated_point[0]), int(rotated_point[1])), radius=5,
                        #                      color=(0, 255, 0), thickness=-1)
                        rotated_landmark.append([int(rotated_point[0]), int(rotated_point[1])])
                    # get cutting corrner
                    # finding the widest pair and the weidth
                    index = 0
                    index_min = 0
                    index_max = 0
                    max = 0
                    min = 1000
                    while index < len(rotated_landmark):
                        if rotated_landmark[index][0] < min:
                            min = rotated_landmark[index][0]
                            index_min = index
                        if rotated_landmark[index][0] > max:
                            max = rotated_landmark[index][0]
                            index_max = index
                        index += 1
                    W = max - min
                    height = W / 416 * 336
                    cutting_corrner = []
                    cutting_corrner.append(
                        [rotated_landmark[index_min][0], rotated_landmark[0][1] - int(height * 0.6)])
                    cutting_corrner.append(
                        [rotated_landmark[index_max][0], rotated_landmark[0][1] - int(height * 0.6)])
                    cutting_corrner.append(
                        [rotated_landmark[index_min][0], rotated_landmark[0][1] + int(height * 0.4)])
                    cutting_corrner.append(
                        [rotated_landmark[index_max][0], rotated_landmark[0][1] + int(height * 0.4)])
                    new_corrner = []
                    for corrner in cutting_corrner:
                        point = (np.matmul(inv(R), corrner))
                        new_corrner.append([int(point[0]), int(point[1])])
                    retangle = [new_corrner[0], new_corrner[2], new_corrner[3], new_corrner[1], ]
                    cv2.polylines(image, pts=[np.array(retangle)], isClosed=True, color=(255, 0, 0), thickness=3)
                    target_list = np.float32([[0, 0], [416, 0], [0, 336], [416, 336]])
                    M = cv2.getPerspectiveTransform(np.float32(new_corrner), target_list)
                    cropped = cv2.warpPerspective(image_, M, (416, 336))
                    processed = self.image_processing(cropped)
                    prob, result = self.prediction(processed)
                    self.wave_plot_list.emit(prob.tolist())
                    pain_prob = (1 - prob[0][0])
                    easy_prob = prob[0][0]
                    if int(result) != 0:
                        pred_message = f'pain! {round(float(pain_prob * 100), 1)}%'
                        cv2.putText(image, pred_message, (new_corrner[0][0] - 100, new_corrner[0][1] - 40),
                                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2, cv2.LINE_AA)
                    else:
                        pred_message = f'easy {round(float(easy_prob * 100), 1)}%'
                        cv2.putText(image, pred_message, (new_corrner[0][0] - 100, new_corrner[0][1] - 40),
                                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2, cv2.LINE_AA)
                    fps_message = f'FPS = {round((1 / ((time.time()+ 0.0001) - t0)), 2)}'
                    cv2.putText(image, fps_message, (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1,
                                cv2.LINE_AA)
                    ConvertQtformat = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
                    Pic = ConvertQtformat.scaled(640, 480, Qt.KeepAspectRatio)

                    ConvertQtformat_cropped = QImage(cropped.data, cropped.shape[1], cropped.shape[0],
                                                     QImage.Format_RGB888)
                    self.Imageupdate.emit(Pic)
                    Pic_crop = ConvertQtformat_cropped.scaled(320, 240, Qt.KeepAspectRatio)
                    self.Cropped_Imageupdate.emit(Pic_crop)
                    self.face_capture_rate.emit(int(((count - failure) / count) * 100))


    def prediction(self, processed_cropped_image):
        processed_cropped_image = processed_cropped_image.to(self.device)
        pred, _ = self.model(processed_cropped_image.unsqueeze(0), processed_cropped_image.unsqueeze(0))
        prob = torch.softmax(pred / self.temp, dim=1) #  分母是 softmax的溫度係數
        result = torch.argmax(pred, dim=-1)
        return prob, result
    def image_processing(self,cropped_image):
        '''
        :return: return a processed tensor, but the arguments needs to re design
        '''
        img_ = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        resize_img = cv2.resize(img_, (108, 108))
        crop_size = (60, 60* 4 / 3)
        trasform = transforms.Compose([transforms.CenterCrop(crop_size), transforms.ToTensor()])
        return trasform(Image.fromarray(resize_img))

    def stop(self):
        self.ThreadActivate = False
    def activate(self):
        self.ThreadActivate = True
    def stopThread(self):
        self.quit()