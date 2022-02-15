from PyQt5 import QtWidgets, QtTest
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from Camera import Camera
import CPOT_UI_2 as ui
import time
import numpy as np
import torch
import argparse
from cls_model12 import CLS
import pyqtgraph as pg
from video_processor import Video_processor

class Main(QMainWindow, ui.Ui_MainWindow):
    def __init__(self, device, camera_number):
        super().__init__()
        self.setupUi(self)
        self.device = device
        self.model = None
        self.stop_BTN.clicked.connect(self.control_stop)
        self.start_BTN.clicked.connect(self.connect_to_input)
        self.pen1 = pg.mkPen(color=(255, 0, 0),width=2, style=Qt.SolidLine)
        self.pen2 = pg.mkPen(color=(0, 255, 0),width=2, style=Qt.SolidLine)
        self.pen3 = pg.mkPen(color=(0, 0, 255),width=2, style=Qt.SolidLine)
        self.wave_plot.addLegend(offset=(10, 10))
        # self.wave_plot.setXRange(0, 150)
        self.wave_plot.setYRange(0, 100)
        self.x = np.linspace(-5, 0, 150)
        self.wave_plot.setXRange(min(self.x), max(self.x))
        self.pain_list = np.array([0])
        self.easy_list = np.array([0])
        self.score_one_list = np.array([0])
        self.wave_plot.setLabel(axis='left', text='confidence')
        self.wave_plot.setLabel(axis='bottom', text='seconds')
        self.data_line1 = self.wave_plot.plot(self.x[-len(self.pain_list):], self.pain_list, name="2 or non zero", pen=self.pen1,)
        self.data_line3 = self.wave_plot.plot(self.x[-len(self.score_one_list):], self.score_one_list, name="1",
                                              pen=self.pen3)
        self.data_line2 = self.wave_plot.plot(self.x[-len(self.easy_list):], self.easy_list, name="0", pen=self.pen2,)
        self.init_time = time.time()
        self.message = 'Please select model and input method'
        self.messagebrwer.setText(self.message)
        self.video_stream.toggled.connect(self.selectinputtype)
        self.video_folder.toggled.connect(self.selectinputtype)
        self.three_class_btn.toggled.connect(self.selectmodeltype)
        self.zero_and_two_btn.toggled.connect(self.selectmodeltype)
        self.zero_and_non_zero.toggled.connect(self.selectmodeltype)
        self.modelLoadingBtn.clicked.connect(self.modelLoading)
        self.Loading_message.setText('please select a model')
        self.inputMethod = ''
        self.modeltype = ''
        self.classamount = 2
        self.Face_capture_rate_show.setDigitCount(3)
        self.temp = 5
        self.path_check_button.clicked.connect(self.selectfolder)
        self.folder_path = None
        self.Camera = Camera(self.model, self.device, self.temp, camera_number)
        self.video_processor = Video_processor(self.model, self.device)
    def selectfolder(self):
        if self.inputMethod != 'video folder':
            self.show_path.setText(f'please click "video folder" as your input type')
        else:
            folder_path = QFileDialog.getExistingDirectory(self, "Open folder", "./")
            print(folder_path)
            self.folder_path = folder_path
            self.show_path.setText(f'path = {self.folder_path}')
            video_list = self.video_processor.set_folder_path(self.folder_path)
            self.message = ''
            for video in video_list:
                self.message += str(video) +'\n---------------------------------------------------\n'
            self.messagebrwer.setText(self.message)

    def clear_wave_plot(self):
        self.pain_list = np.array([0])
        self.easy_list = np.array([0])
        self.score_one_list = np.array([0])
        self.x = np.linspace(-10, 0, 150)
        self.data_line1.setData([0], self.pain_list)
        self.data_line2.setData([0], self.easy_list)
        self.data_line3.setData([0], self.score_one_list)

    def connect_to_input(self):
        self.clear_wave_plot()
        self.init_time = time.time()
        if self.model:
            if self.inputMethod == 'webcam':
                self.Camera.activate()
                self.message += '\n' + ' webcam starting'
                self.messagebrwer.setText(self.message)
                # self.Camera.start()
                self.Camera.run_prediction()
            elif self.inputMethod == 'video folder':
                print('running video folder prediction')
                self.message = 'starting\n'
                self.messagebrwer.setText(self.message)
                QtTest.QTest.qWait(1000)
                self.video_processor.run_prediction()
        else:
            self.messagebrwer.setText('no model yet')

    def control_stop(self):
        if self.inputMethod == 'webcam':
            self.Camera.stop()
        else:
            self.video_processor.stop()
            self.message += 'stop running'
            self.messagebrwer.setText(self.message)
    def selectinputtype(self):
        radioBtn = self.sender()
        if radioBtn.isChecked():
            print(radioBtn.text(), 'clicked')
            self.message += '\n' + radioBtn.text() + ' is your input method'
            self.messagebrwer.setText(self.message)
            self.inputMethod = radioBtn.text()
            if self.inputMethod == 'webcam':
                self.message += '\n' + ' webcam initialize'
                self.messagebrwer.setText(self.message)
                self.Camera.Imageupdate.connect(self.ImageUpdateSlot)
                self.Camera.wave_plot_list.connect(self.wave_plot_update)
                self.Camera.face_capture_rate.connect(self.show_capture_rate)
                self.Camera.Cropped_Imageupdate.connect(self.Crop_image_update_slot)
                self.message += '\n' + ' webcam initialize success'
                self.messagebrwer.setText(self.message)
                if self.model:
                    self.Camera.reload_model(self.model, self.temp)
            elif radioBtn.text() == 'video folder':
                # self.video_processor = Video_processor(self.model, self.device)
                self.video_processor.single_video_information.connect(self.show_videos_result)
                self.video_processor.Imageupdate.connect(self.ImageUpdateSlot)
                self.video_processor.Cropped_Imageupdate.connect(self.Crop_image_update_slot)
                self.video_processor.wave_plot_list.connect(self.wave_plot_update)
                self.video_processor.face_capture_rate.connect(self.show_capture_rate)
                if self.model:
                    self.video_processor.reload_model(self.model, self.temp, self.modeltype)

    def selectmodeltype(self):
        radioBtn = self.sender()
        if radioBtn.isChecked():
            print(radioBtn.text(), 'clicked')
            self.message += '\n' + radioBtn.text() + ' is your model'
            self.messagebrwer.setText(self.message)
            self.modeltype = radioBtn.text()
    def modelLoading(self):
        self.Loading_message.setText('model loading, please wait...')
        time.sleep(3)
        if self.modeltype =='':
            self.message += '\n' + ' please select a model first.'
            self.messagebrwer.setText(self.message)
            self.Loading_message.setText('no model selected')
            self.Loading_message.repaint()
        elif self.modeltype == '{0},{1},{2}':
            self.pain_list = np.array([0])
            self.easy_list = np.array([0])
            self.score_one_list = np.array([0])
            self.classamount = 3
            self.temp = 0.7
            self.Loading_message.setText('loading {0},{1},{2} model, please wait')
            self.Loading_message.repaint()
            QtTest.QTest.qWait(1000)
            self.model = self.load_model(3).to(device).eval()
            print('loading {0},{1},{2} model')
            if self.inputMethod == 'webcam':
                print('camera model loading')
                self.Camera.reload_model(self.model, self.temp)
            elif self.inputMethod == 'video folder':
                print('video model loading')
                self.video_processor.reload_model(self.model, self.temp, self.modeltype)
            self.message += '\n' + '{0},{1},{2} model loading success'
            self.messagebrwer.setText(self.message)
            self.Loading_message.setText('loading finish')

        elif self.modeltype == '{0},{1,2}':
            self.pain_list = np.array([0])
            self.easy_list = np.array([0])
            self.score_one_list = np.array([0])
            self.classamount = 2
            self.temp = 5
            if self.data_line3:
                self.data_line3.setData([0],[0]) # clear plot
            self.Loading_message.setText('loading {0},{1,2} model, please wait')
            self.Loading_message.repaint()
            QtTest.QTest.qWait(1000)
            self.model = self.load_model(2).to(device).eval()
            print('loading {0},{1,2} model')
            if self.inputMethod == 'webcam':
                print('camera model loading')
                self.Camera.reload_model(self.model, self.temp)
            elif self.inputMethod == 'video folder':
                print('video model loading')
                self.video_processor.reload_model(self.model, self.temp, self.modeltype)
            self.message += '\n' + '{0},{1,2} model loading success'
            self.messagebrwer.setText(self.message)
            self.Loading_message.setText('loading finish')

        elif self.modeltype == '{0},{2}':
            self.classamount = 2
            self.pain_list = np.array([0])
            self.easy_list = np.array([0])
            self.score_one_list = np.array([0])
            self.temp = 2.2
            if self.data_line3:
                self.data_line3.setData([0],[0]) # clear plot
            print('loading {0},{2} model')
            self.Loading_message.setText('loading {0},{2} model, please wait')
            self.Loading_message.repaint()
            QtTest.QTest.qWait(1000)
            self.model = self.load_model(2, no_one=True).to(device).eval()
            if self.inputMethod == 'webcam':
                print('camera model loading')
                self.Camera.reload_model(self.model, self.temp)
            elif self.inputMethod == 'video folder':
                print('video model loading')
                self.video_processor.reload_model(self.model, self.temp, self.modeltype)
            self.message += '\n' + '{0},{2} model loading success'
            self.messagebrwer.setText(self.message)
            self.Loading_message.setText('loading finish')
        else:
            print(self.modeltype)



    def load_model(self, c, no_one=False):
        print(f'loading ./cls12-none-60_c{c}{"_no1" if no_one else ""}_1l_13+18_resnet34_128_25_25_1_0.0001_512emb_pretrained_0.pt')
        model = CLS('resnet34', pretrained=True, c=c,combine='cat', two_layer=False)
        model_path = f'./cls12-none-60_c{c}{"_no1" if no_one else ""}_1l_13+18_resnet34_128_25_25_1_0.0001_512emb_pretrained_0.pt'
        model.load_state_dict(torch.load(model_path))
        print('success')
        return model

    def wave_plot_update(self, prob):
        if self.classamount !=3:
            pain_prob, easy_prob = prob[0][1], prob[0][0]
            self.x = np.append(self.x[1:], np.array(time.time() - self.init_time))
            self.wave_plot.setXRange(min(self.x), max(self.x))
            if len(self.pain_list) < 150 and len(self.easy_list) < 150:
                self.pain_list = np.append(self.pain_list, pain_prob * 100)
                self.easy_list = np.append(self.easy_list, easy_prob * 100)
                self.data_line1.setData(self.x[-len(self.pain_list):], self.pain_list)
                self.data_line2.setData(self.x[-len(self.easy_list):], self.easy_list)
            else:
                self.x = np.append(self.x[1:], np.array(time.time() - self.init_time))
                self.pain_list = np.append(self.pain_list[1:], pain_prob*100)
                self.easy_list = np.append(self.easy_list[1:], easy_prob*100)
                self.data_line1.setData(self.x, self.pain_list)
                self.data_line2.setData(self.x, self.easy_list)
        else:
            pain_prob,score_one, easy_prob = prob[0][2],prob[0][1], prob[0][0]
            self.x = np.append(self.x[1:], np.array(time.time() - self.init_time))
            self.wave_plot.setXRange(min(self.x), max(self.x))
            if len(self.pain_list) < 150 and len(self.easy_list) < 150:
                self.pain_list = np.append(self.pain_list, pain_prob * 100)
                self.score_one_list = np.append(self.score_one_list, score_one * 100)
                self.easy_list = np.append(self.easy_list, easy_prob * 100)
                self.data_line1.setData(self.x[-len(self.pain_list):], self.pain_list)
                self.data_line2.setData(self.x[-len(self.easy_list):], self.easy_list)
                self.data_line3.setData(self.x[-len(self.score_one_list):], self.score_one_list)
            else:
                self.pain_list = np.append(self.pain_list[1:], pain_prob * 100)
                self.score_one_list = np.append(self.score_one_list[1:], score_one * 100)
                self.easy_list = np.append(self.easy_list[1:], easy_prob * 100)
                self.data_line1.setData(self.x, self.pain_list)
                self.data_line2.setData(self.x, self.easy_list)
                self.data_line3.setData(self.x, self.score_one_list)

    def ImageUpdateSlot(self, Image):
        self.origional_image.setPixmap(QPixmap.fromImage(Image))
    def Crop_image_update_slot(self, croped_image):
        self.cropped_face.setPixmap(QPixmap.fromImage(croped_image))
    def show_capture_rate(self, rate):
        self.Face_capture_rate_show.display(rate)
    def show_videos_result(self, message):
        self.message += message + "\n"
        self.messagebrwer.setText(self.message)



if __name__ == '__main__':
    import sys
    parser = argparse.ArgumentParser(description='Pain training')
    parser.add_argument('--gpu', type=int, default=0, help='which gpu are you using')
    parser.add_argument('--cam', type=int, default=0, help='camera number')
    args = parser.parse_args()
    camera_number = args.cam
    print(f'camera number = {camera_number}')
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    app = QtWidgets.QApplication(sys.argv)
    window = Main(device, camera_number)
    window.show()
    sys.exit(app.exec_())