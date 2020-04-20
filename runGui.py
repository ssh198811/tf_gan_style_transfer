from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QMessageBox, QListWidgetItem, QListWidget
from PyQt5.QtCore import pyqtSignal, QThread, Qt, QUrl, QPoint, QSize
from PyQt5 import QtGui
from PyQt5.QtGui import QIcon, QDesktopServices, QPixmap, QPainter
from ui import maingui
import sys
import os
from skimage import io
import numpy as np
from model.Generator import Generator
from utils.save_img import save_sample
import qdarkstyle
import tensorflow as tf
import cv2
import loadDict

from utils import img_process

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
        self.pairs = loadDict.load_dict()

        for pair in self.pairs:
            self.style_img_path.append(pair['icon'])
            self.style_model_path.append(pair['model'])

        # # match style 1
        # self.style_img_path.append("icons/1.jpg")
        # self.style_model_path.append("tf_model_p/as_model.npy")
        #
        # self.style_img_path.append("icons/2.jpg")
        # self.style_model_path.append("tf_model_p/dm_model.npy")
        #
        # self.style_img_path.append("icons/3.jpg")
        # self.style_model_path.append("tf_model_p/kh_model.npy")
        #
        # self.style_img_path.append("icons/4.jpg")
        # self.style_model_path.append("tf_model_p/kp_model.npy")
        #
        # self.style_img_path.append("icons/6.jpg")
        # self.style_model_path.append("tf_model_p/miyaziki_model.npy")
        #
        # self.style_img_path.append("icons/7.jpg")
        # self.style_model_path.append("tf_model_p/pp_model.npy")
        #
        # self.style_img_path.append("icons/8.jpg")
        # self.style_model_path.append("tf_model_p/rc_model.npy")
        #
        # self.style_img_path.append("icons/9.jpg")
        # self.style_model_path.append("tf_model_p/sc_model.npy")
        #
        # self.style_img_path.append("icons/11.jpg")
        # self.style_model_path.append("tf_model_p/tr_model.npy")


    def load_style_list(self):

        # load style icon
        self.style_img_icon = []
        for img_path in self.style_img_path:
            pix = QPixmap(img_path)
            # pix.scaled(QSize(64,64))

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
            item.setSizeHint(QSize(64,64))
            item.setTextAlignment(Qt.AlignHCenter|Qt.AlignVCenter)
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

        self.process_src_img(model, self.src_img_path)

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


        #先临时存下完全风格后的图片
        self.temp_dst_img_path = './temp/dst.jpg'
        save_sample(self.dst_img, self.temp_dst_img_path)
        # os.remove(self.src_img_path+'_temp.jpg')

        # 显示完全风格后的图片
        self.ui.list_dst_show.clear()
        item = QListWidgetItem()
        item.setIcon(QIcon(self.temp_dst_img_path))
        item.setSizeHint(QSize(256,256))
        self.ui.list_dst_show.setIconSize(QSize(256,256))
        self.ui.list_dst_show.addItem(item)

        #显示lerp后的图片
        self.load_scaled_dst_img()
        self.update_lerp_view(self.lerp())

    def open_src_img(self):
        fileNames_choose, filetype = QFileDialog.getOpenFileNames(self, "选取图片文件",
                                                                self.cwd,  # 起始路径
                                                                "(*.jpg);;(*.png);;")  # 设置文件扩展名过滤,用双分号间隔
        if len(fileNames_choose) == 0:
            print("\n取消选择")
            return
        #
        # for i, fileName in enumerate(fileNames_choose):
        #     if self.is_contain_chinese(fileName) is True:
        #         fileNames_choose[i] = os.path.



        self.src_img_path = fileNames_choose[0]
        # self.src_img = QPixmap(fileNames_choose[0])
        # self.src_img = self.src_img.scaledToHeight(256, mode=Qt.SmoothTransformation)
        # self.src_img = self.src_img.scaledToWidth(256, mode=Qt.SmoothTransformation)
        # self.ui.label_src.setPixmap(self.src_img)

        self.ui.list_src_show.clear()
        item = QListWidgetItem()
        item.setIcon(QIcon(self.src_img_path))
        item.setSizeHint(QSize(256,256))
        self.ui.list_src_show.setIconSize(QSize(256,256))
        self.ui.list_src_show.addItem(item)


        self.load_scaled_src_img()
        #加载其他选中的文件
        self.src_img_paths = fileNames_choose
        self.load_all_src_img()

    def load_all_src_img(self):
        # load src icon
        index = 0
        while index < len(self.src_img_paths):
            item = QListWidgetItem()
            item.setSizeHint(QSize(64, 64))
            item.setIcon(QIcon(self.src_img_paths[index]))
            self.ui.list_src.addItem(item)
            index += 1
        self.ui.list_src.setIconSize(QSize(64,64))
        QApplication.processEvents()

    def load_scaled_src_img(self):
        self.scaled_src_image = img_process.read_img(self.src_img_path)
        self.scaled_src_image = cv2.resize(self.scaled_src_image, (256, 256), interpolation=cv2.INTER_CUBIC)

    def load_scaled_dst_img(self):
        self.scaled_dst_image = img_process.read_img(self.temp_dst_img_path)
        self.scaled_dst_image = cv2.resize(self.scaled_dst_image, (256, 256), interpolation=cv2.INTER_CUBIC)

    def save_dst_img(self):
        fileName_choose, filetype = QFileDialog.getSaveFileName(self,
                                                                "文件保存",
                                                                self.cwd,  # 起始路径
                                                                "(*.jpg);;(*.png);;")
        if fileName_choose == "":
            print("\n取消选择")
            return

        save_sample(self.dst_img, fileName_choose)

        self.resave_dst_img(self.src_img_path, self.temp_dst_img_path, fileName_choose)

    def resave_dst_img(self, src_img_path, dst_img_path, save_img_path):
        # 重新保存当前
        self.src_img_path = src_img_path
        self.temp_dst_img_path = dst_img_path
        img_lerp = self.lerp()
        cv2.imwrite(save_img_path, img_lerp)

    def lerp(self):
        if self.src_img_path is None:
            return
        if self.temp_dst_img_path is None:
            return


        lerp_value = self.ui.lerp_slider.value()
        img_add = img_process.lerp_img(self.src_img_path, self.temp_dst_img_path, lerp_value)
        self.update_lerp_view(img_add)
        return img_add


    def update_lerp_view(self, img_add):
        self.temp_lerp_img_path ='./temp/lerp.jpg'
        cv2.imwrite(self.temp_lerp_img_path, img_add)

        # self.lerp_img = QPixmap(self.temp_lerp_img_path)
        # self.lerp_img = self.lerp_img.scaledToHeight(256, mode=Qt.SmoothTransformation)
        # self.lerp_img = self.lerp_img.scaledToWidth(256, mode=Qt.SmoothTransformation)
        # self.ui.label_lerp.setPixmap(self.lerp_img)
        self.ui.list_lerp_show.clear()
        item = QListWidgetItem()
        item.setIcon(QIcon(self.temp_lerp_img_path))
        item.setSizeHint(QSize(256,256))
        self.ui.list_lerp_show.setIconSize(QSize(256,256))
        self.ui.list_lerp_show.addItem(item)

    def src_list_click(self):
        if self.src_img_paths is None:
            return

        src_index = self.ui.list_src.currentIndex().row()

        self.src_img_path = self.src_img_paths[src_index]
        self.ui.list_src_show.clear()
        self.ui.list_dst_show.clear()
        self.ui.list_lerp_show.clear()
        self.ui.lerp_slider.setValue(100)

        item = QListWidgetItem()
        item.setIcon(QIcon(self.src_img_path))
        item.setSizeHint(QSize(256,256))
        self.ui.list_src_show.setIconSize(QSize(256,256))
        self.ui.list_src_show.addItem(item)

        QApplication.processEvents()

if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps) # high resolution pix use this
    app = QApplication(sys.argv)

    icon = QIcon()
    icon.addPixmap(QtGui.QPixmap("ui/icons.tga"),QtGui.QIcon.Normal, QtGui.QIcon.Off)

    window = ApplicationWindow()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window.setWindowIcon(icon)
    window.setFixedSize(933,548)
    window.show()

    sys.exit(app.exec_())