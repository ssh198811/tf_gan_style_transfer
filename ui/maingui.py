# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'maingui.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(641, 482)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.list_style = QtWidgets.QListWidget(self.centralwidget)
        self.list_style.setGeometry(QtCore.QRect(20, 340, 601, 91))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.list_style.sizePolicy().hasHeightForWidth())
        self.list_style.setSizePolicy(sizePolicy)
        self.list_style.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.list_style.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.list_style.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.list_style.setAutoScroll(False)
        self.list_style.setFlow(QtWidgets.QListView.LeftToRight)
        self.list_style.setResizeMode(QtWidgets.QListView.Adjust)
        self.list_style.setViewMode(QtWidgets.QListView.ListMode)
        self.list_style.setObjectName("list_style")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(20, 10, 91, 21))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(320, 10, 81, 21))
        self.pushButton_2.setObjectName("pushButton_2")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(20, 40, 291, 291))
        self.groupBox.setObjectName("groupBox")
        self.label_src = QtWidgets.QLabel(self.groupBox)
        self.label_src.setGeometry(QtCore.QRect(10, 20, 256, 256))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_src.sizePolicy().hasHeightForWidth())
        self.label_src.setSizePolicy(sizePolicy)
        self.label_src.setText("")
        self.label_src.setObjectName("label_src")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(320, 40, 301, 291))
        self.groupBox_2.setObjectName("groupBox_2")
        self.label_dst = QtWidgets.QLabel(self.groupBox_2)
        self.label_dst.setGeometry(QtCore.QRect(20, 20, 256, 256))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_dst.sizePolicy().hasHeightForWidth())
        self.label_dst.setSizePolicy(sizePolicy)
        self.label_dst.setText("")
        self.label_dst.setObjectName("label_dst")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 641, 18))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.list_style.setCurrentRow(-1)
        self.list_style.clicked['QModelIndex'].connect(MainWindow.stylelistClick)
        self.pushButton.clicked.connect(MainWindow.open_src_img)
        self.pushButton_2.clicked.connect(MainWindow.save_dst_img)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "剑网3 Ai滤镜 beta v1.0.0"))
        self.pushButton.setText(_translate("MainWindow", "打开原始图片"))
        self.pushButton_2.setText(_translate("MainWindow", "保存生成图片"))
        self.groupBox.setTitle(_translate("MainWindow", "原始图片"))
        self.groupBox_2.setTitle(_translate("MainWindow", "风格图片"))
