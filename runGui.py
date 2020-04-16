from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QMessageBox, QListWidgetItem, QListWidget
from PyQt5.QtCore import pyqtSignal, QThread, Qt, QUrl, QPoint, QSize
from PyQt5 import QtGui
from PyQt5.QtGui import QIcon, QDesktopServices, QPixmap
from ui import maingui
import sys
import os
from skimage import io
import numpy as np
from model.Generator import Generator
from utils.save_img import save_sample
import qdarkstyle
import tensorflow as tf

class ApplicationWindow(QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()

        self.cwd = os.getcwd()  # 获取当前程序文件位置
        self.ui = maingui.Ui_MainWindow()
        self.ui.setupUi(self)
        self.init_style()
        self.load_style_list()

    def init_style(self):
        self.style_img_path = []
        self.style_model_path = []
        self.src_img_path = None

        # match style 1
        self.style_img_path.append("icons/1.jpg")
        self.style_model_path.append("tf_model_p/as_model.npy")

        self.style_img_path.append("icons/2.jpg")
        self.style_model_path.append("tf_model_p/dm_model.npy")

        self.style_img_path.append("icons/3.jpg")
        self.style_model_path.append("tf_model_p/kh_model.npy")

        self.style_img_path.append("icons/4.jpg")
        self.style_model_path.append("tf_model_p/kp_model.npy")

        self.style_img_path.append("icons/6.jpg")
        self.style_model_path.append("tf_model_p/miyaziki_model.npy")

        self.style_img_path.append("icons/7.jpg")
        self.style_model_path.append("tf_model_p/pp_model.npy")

        self.style_img_path.append("icons/8.jpg")
        self.style_model_path.append("tf_model_p/rc_model.npy")

        self.style_img_path.append("icons/9.jpg")
        self.style_model_path.append("tf_model_p/sc_model.npy")

        self.style_img_path.append("icons/11.jpg")
        self.style_model_path.append("tf_model_p/tr_model.npy")


    def load_style_list(self):

        # load style icon
        self.style_img_icon = []
        for img_path in self.style_img_path:
            pix = QPixmap(img_path)
            pix.scaled(QSize(64,64))
            icon = QIcon()
            icon.addPixmap(pix)
            # img = QPixmap(img_path)
            # img = img.scaledToHeight(64, mode=Qt.SmoothTransformation)
            # img = img.scaledToWidth(64, mode=Qt.SmoothTransformation)
            # icon.addPixmap(img)

            self.style_img_icon.append(icon)

        index = 0
        while index < len(self.style_img_path):
            item = QListWidgetItem()
            item.setIcon(self.style_img_icon[index])
            self.ui.list_style.addItem(item)
            index += 1

        iconsize = QSize(64,64)
        self.ui.list_style.setIconSize(iconsize)

        QApplication.processEvents()

        # load style model default as none
        self.style_mode = []
        for style_model in self.style_model_path:
            # parameter_dict = np.load(style_model).item()
            # model = Generator(in_chanel=3, out_chanel=3, parameter_dict=parameter_dict)
            self.style_mode.append(None)

    def stylelistClick(self):
        # self.ui.label_style.setText(self.ui.list_style.currentItem().text())
        if self.src_img_path is None:
            return

        style_index = self.ui.list_style.currentIndex().row()

        model = self.style_mode[style_index]
        if model == None:
            parameter_dict = np.load(self.style_model_path[style_index]).item()
            model = Generator(in_chanel=3, out_chanel=3, parameter_dict=parameter_dict)

        self.process_src_img(model,self.src_img_path)

    def process_src_img(self, model, src_img_path):
        min_factor = 16
        test_path =src_img_path
        test_img = np.array(io.imread(test_path), dtype=np.float32) / 255.0 * 2.0 - 1
        h, w, c = np.shape(test_img)
        print(h,w)
        if c==1:
            test_img=np.concatenate([test_img,test_img,test_img],axis=2)
        elif c>3:
             test_img = test_img[:, :, :3]
             c=3
             print('chanel num larger 3,not rgb')
        ###########填补操作####################
        h_pad = np.ceil(h / min_factor) * min_factor - h
        if h_pad < 6 and h_pad != 0:
            h_pad += min_factor
        w_pad = np.ceil(w / min_factor) * min_factor - w
        if w_pad < 6 and w_pad != 0:
            w_pad += min_factor
        h_pad_up = int(h_pad // 2)
        h_pad_down = int(h_pad - h_pad_up)
        w_pad_left = int(w_pad // 2)
        w_pad_right = int(w_pad - w_pad_left)
        test_img = np.reshape(test_img, [1, h, w, 3])
        if h_pad == 0 and w_pad >= 6:
            test_img = tf.pad(test_img, [[0, 0], [0, 0], [3, 3], [0, 0]], mode="REFLECT")
            test_img = tf.pad(test_img, [[0, 0], [0, 0], [w_pad_left - 3, w_pad_right - 3], [0, 0]])
        elif w_pad == 0 and h_pad >= 6:
            test_img = tf.pad(test_img, [[0, 0], [3, 3], [0, 0], [0, 0]], mode="REFLECT")
            test_img = tf.pad(test_img, [[0, 0], [h_pad_up - 3, h_pad_down - 3], [0, 0], [0, 0]])
        elif h_pad >= 6 and w_pad >= 6:
            test_img = tf.pad(test_img, [[0, 0], [3, 3], [3, 3], [0, 0]], mode="REFLECT")
            test_img = tf.pad(test_img,
                              [[0, 0], [h_pad_up - 3, h_pad_down - 3], [w_pad_left - 3, w_pad_right - 3], [0, 0]])
        pre_img = model(test_img)
        pre_img = pre_img[0]
        pre_img = pre_img[h_pad_up:h_pad_up + h, w_pad_left:w_pad_left + w,:c]
        self.dst_img = pre_img

        save_sample(self.dst_img, self.src_img_path+'_temp.jpg')
        temp_dst = QPixmap(self.src_img_path+'_temp.jpg')
        temp_dst = temp_dst.scaledToHeight(256, mode=Qt.SmoothTransformation)
        temp_dst = temp_dst.scaledToWidth(256, mode=Qt.SmoothTransformation)
        self.ui.label_dst.setPixmap(temp_dst)
        os.remove(self.src_img_path+'_temp.jpg')

    def open_src_img(self):
        fileName_choose, filetype = QFileDialog.getOpenFileName(self, "选取图片文件",
                                                                self.cwd,  # 起始路径
                                                                "(*.jpg);;(*.png);;")  # 设置文件扩展名过滤,用双分号间隔
        if fileName_choose == "":
            print("\n取消选择")
            return

        self.src_img_path = fileName_choose
        self.src_img = QPixmap(fileName_choose)
        self.src_img = self.src_img.scaledToHeight(256, mode=Qt.SmoothTransformation)
        self.src_img = self.src_img.scaledToWidth(256, mode=Qt.SmoothTransformation)
        self.ui.label_src.setPixmap(self.src_img)

    def save_dst_img(self):
        fileName_choose, filetype = QFileDialog.getSaveFileName(self,
                                                                "文件保存",
                                                                self.cwd,  # 起始路径
                                                                "(*.jpg);;(*.png);;")
        if fileName_choose == "":
            print("\n取消选择")
            return

        save_sample(self.dst_img, fileName_choose)


if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps) # high resolution pix use this
    app = QApplication(sys.argv)

    icon = QIcon()
    icon.addPixmap(QtGui.QPixmap("ui/icons.tga"),QtGui.QIcon.Normal, QtGui.QIcon.Off)

    window = ApplicationWindow()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window.setWindowIcon(icon)
    window.setFixedSize(641,482)
    window.show()

    sys.exit(app.exec_())