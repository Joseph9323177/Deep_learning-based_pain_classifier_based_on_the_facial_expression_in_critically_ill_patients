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
import glob
import os
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QFileDialog
import copy


class Video_processor(QThread):
    single_video_information = pyqtSignal(str)
    Imageupdate = pyqtSignal(QImage)
    Cropped_Imageupdate = pyqtSignal(QImage)
    wave_plot_list = pyqtSignal(list)
    face_capture_rate = pyqtSignal(int)
    def __init__(self,model,device):
        super(Video_processor, self).__init__()
        self.model = None
        no_face_backup = cv2.imread('./no_face.png')
        self.no_face_backup = cv2.cvtColor(no_face_backup, cv2.COLOR_BGR2RGB)
        self.class_number = 2
        self.device = device
        self.model = model
        self.temp = 5
        self.modeltype = ''
        self.Threadactivate = False
    def label_converter(self, video_path, modeltype):
        video_name = os.path.basename(video_path)
        print(video_name)
        print(modeltype)
        print('split = ', video_name.split('_')[3])
        origional_facial_label = int(video_name.split('_')[3])
        if modeltype == '{0},{1,2}':
            return 0 if origional_facial_label == 0 else 1
        elif modeltype == '{0},{2}':
            return -1 if origional_facial_label == 1 else origional_facial_label
        else:
            return origional_facial_label

    def reload_model(self, model, temp, modeltype):
        print('in video processor thread')
        self.model = model
        self.temp = temp
        self.modeltype = modeltype
        print('model type = ', modeltype)
        print('model change success in video processor')
        print('temp = ', self.temp)

    def set_folder_path(self, folder_path):
        self.folder_path = folder_path
        self.video_list = glob.glob(folder_path + '/*.avi') + glob.glob(folder_path + '/*.mp4')
        print(self.video_list)
        return self.video_list


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
        crop_size = (60, 60 * 4 / 3)
        trasform = transforms.Compose([transforms.CenterCrop(crop_size), transforms.ToTensor()])
        return trasform(Image.fromarray(resize_img))
    def stop(self):
        self.Threadactivate = False
        print('stop')
    def run_prediction(self):
        self.Threadactivate = True
        def CFAS(video_path, GT, ID_dict):
            predicted_class = np.zeros(3)
            ID = video_path.split('\\')[-1].split('_')[0]
            print(ID)
            mp_face_mesh = mp.solutions.face_mesh
            video_path = video_path.replace('\\', '/')
            cap = cv2.VideoCapture(video_path)
            count = 0
            failure = 0
            correct = 0
            with mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as face_mesh:
                while self.Threadactivate:
                    t0 = time.time()
                    ret, frame = cap.read()
                    cv2.waitKey(1)
                    if not ret:
                        print("can't read video")
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
                        show_ = cv2.resize(image_, (1920, 1080))
                        cv2.putText(show_, fps_message, (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2,
                                    cv2.LINE_AA)
                        cv2.putText(show_, "No face", (400, 350), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 3,
                                    cv2.LINE_AA)
                        backup = self.no_face_backup
                        ConvertQtformat = QImage(show_.data, show_.shape[1], show_.shape[0],show_.shape[1]*3, QImage.Format_RGB888)
                        Pic = ConvertQtformat.scaled(640, 480, Qt.KeepAspectRatio)
                        self.Imageupdate.emit(Pic)
                        ConvertQtformat_cropped = QImage(backup.data, backup.shape[1], backup.shape[0], backup.shape[1]*3,
                                                         QImage.Format_RGB888)
                        Pic_crop = ConvertQtformat_cropped.scaled(320, 240, Qt.KeepAspectRatio)
                        self.wave_plot_list.emit([[0, 0, 0]])
                        self.Cropped_Imageupdate.emit(Pic_crop)
                        face_capture_rate = ((count - failure) / count)
                        print(f'face_capture_rate = {face_capture_rate}')
                        self.face_capture_rate.emit(int(face_capture_rate))
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
                        predicted_class[result.item()] += 1
                        self.wave_plot_list.emit(prob.tolist())
                        pain_prob = (1 - prob[0][0])
                        easy_prob = prob[0][0]
                        image = cv2.resize(image, (1920,1080))
                        if result.item() == GT:
                            correct += 1
                        if int(result) != 0:
                            pred_message = f'pain! {round(float(pain_prob * 100), 1)}%'
                            cv2.putText(image, pred_message, (new_corrner[0][0] -100, new_corrner[0][1]),
                                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2, cv2.LINE_AA)
                        else:
                            pred_message = f'easy {round(float(easy_prob * 100), 1)}%'
                            cv2.putText(image, pred_message, (new_corrner[0][0] -100, new_corrner[0][1]),
                                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2, cv2.LINE_AA)
                        fps_message = f'FPS = {round((1 / ((time.time()+ 0.0001) - t0)), 2)}'
                        cv2.putText(image, fps_message, (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2,
                                    cv2.LINE_AA)
                        ConvertQtformat = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
                        Pic = ConvertQtformat.scaled(640, 480, Qt.KeepAspectRatio)
                        self.Imageupdate.emit(Pic)
                        ConvertQtformat_cropped = QImage(cropped.data, cropped.shape[1], cropped.shape[0],
                                                         QImage.Format_RGB888)
                        Pic_crop = ConvertQtformat_cropped.scaled(320, 240, Qt.KeepAspectRatio)
                        self.Cropped_Imageupdate.emit(Pic_crop)

                        face_capture_rate = ((count - failure) / count)
                        self.face_capture_rate.emit(int(face_capture_rate*100))
                        print(f'count, failure = {count, failure}')
                        print(f'face_capture_rate = {face_capture_rate}')
            print(f'GT = {GT}')
            ID_dict[ID][GT][1] += correct
            ID_dict[ID][GT][0] += count
            accuracy = (correct / (count - failure)) if (count - failure) != 0 else None
            single_video_information = f'{os.path.basename(video_path)} , ground truth = {GT} \nfacial capture rate = {face_capture_rate*100}, \naccuracy = {accuracy*100 if accuracy != None else "N/A"},\npredict class = {(predicted_class/ (count - failure))}\n---------------------------------------------------'
            self.single_video_information.emit(single_video_information)
            print(f'count failure = {count, failure}')
            return accuracy, face_capture_rate, single_video_information, ID_dict

        log = []
        ID_dict = {}
        failure_video_list =[]
        for video_path in self.video_list:
            if video_path.split('\\')[-1].split('_')[0] not in ID_dict.keys():
                ID_dict[video_path.split('\\')[-1].split('_')[0]] = [[0,0],[0,0],[0,0]]
        evaulated_video_num = 0
        for video_path in self.video_list:
            print(video_path)
            converted_label = self.label_converter(video_path, modeltype=self.modeltype)
            if converted_label != -1:
                print(f'converted_label = {converted_label}')
                evaulated_video_num +=1
                accuracy, face_capture_rate, single_video_information, ID_dict = CFAS(video_path, converted_label, ID_dict)
                if face_capture_rate == 0:
                    failure_video_list.append(video_path)
                log.append(single_video_information)
            else:
                print('skipped')
        accuracy_dict = {}
        failure_case_list = []
        for ID in ID_dict.keys():
            if ID not in accuracy_dict:
                accuracy_dict[ID] = np.zeros(4)
            for i in range(3):
                if ID_dict[ID][i][0] != 0:
                    accuracy_dict[ID][i] = (ID_dict[ID][i][1] /ID_dict[ID][i][0])  # count = 0 時不予計算。
                    accuracy_dict[ID][3] += 1
                else:
                    accuracy_dict[ID][i] = 0
        accuracy_dict_copy = copy.deepcopy(accuracy_dict)
        for ID in accuracy_dict_copy.keys():
            if accuracy_dict[ID][3] == 0:
                accuracy_dict.pop(ID)
                failure_case_list.append(ID)
        average_acc = sum([(sum(accuracy_dict[ID][:-1]) / accuracy_dict[ID][-1]) for ID in accuracy_dict.keys()]) / len(accuracy_dict)

        self.single_video_information.emit('DONE\n---------------------------------------------------')
        self.Box = QMessageBox(QMessageBox.Question, 'Finish', "Do you want to save log?")
        self.Box.setIcon(QMessageBox.Information)
        self.Box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        result = self.Box.exec_()
        if result == QMessageBox.Ok:
            log_file_path = QFileDialog.getSaveFileName(filter='*.txt')
            print(log_file_path)
            with open(log_file_path[0], 'w') as log_file:
                folder_message = f'Total video amount = {len(self.video_list)} \nEvaulated videos ={evaulated_video_num}\nTotal ID amount = {len(accuracy_dict)}\nAverage accuracy = {round(average_acc*100,3)}% \n'
                failure_IDs = f'Failure IDs = {failure_case_list}'
                failure_videos = f'Failure videos ={failure_video_list}'
                log_file.write(folder_message)
                log_file.write('\n')
                log_file.write(failure_videos)
                log_file.write('\n')
                log_file.write(failure_IDs)
                log_file.write('\n')
                log_file.write('----------------------------------------------------------')
                log_file.write('\n')
                log_file.write('----------------------------------------------------------')
                log_file.write('\n')
                for ID in accuracy_dict.keys():
                    ID_message = f'{ID} : average accuracy = {(sum(accuracy_dict[ID][:-1]) / accuracy_dict[ID][-1]) }, accuracy for each class = {accuracy_dict[ID][:-1]}, class number = {accuracy_dict[ID][-1]}'
                    log_file.write(ID_message)
                    log_file.write('\n')
                log_file.write('**********************************************************')
                log_file.write('\n')
                for info in log:
                    log_file.write(info)
                    log_file.write('\n')


    def stopThread(self):
        self.quit()