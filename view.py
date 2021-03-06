# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'qt_ilk.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!
from PyQt5.uic import loadUi
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMainWindow,QWidget
from PyQt5.QtGui import QPixmap, QImage


class loadUi_example(QWidget):

    def __init__(self, cam_num):
        super().__init__()
        loadUi("view.ui", self)
        self.cam_num = cam_num
        self.cam_num.open()
        self.scrollArea.setWidget(self.textEdit) #size of defects will be shown in scrollArea
        self.scrollArea_2.setWidget(self.textEdit_2)
        #self.label_2.setGeometry(75,10,70,170)
        self.timer = QTimer()
        self.timer.timeout.connect(self.readNextFrame)
        self.timer.start(1000 / 24) #24 fps

    def readNextFrame(self):
        self.frame = self.cam_num.read()

    @pyqtSlot()
    def on_pushButton_clicked(self):
        if self.frame is not None:
            image = QImage(self.frame, self.frame.shape[1], self.frame.shape[0],
                           QImage.Format_RGB888)  # The image is stored using a 24-bit RGB format (8-8-8).
            self.pixmap = QPixmap.fromImage(image)
        self.label.setPixmap(self.pixmap)

    @pyqtSlot()
    def on_pushButton_2_clicked(self):

        
        lower_sample,upper_sample = self.cam_num.find_biggest_contour()

        lowerPartDefects,the_list = self.cam_num.find_defects(lower_sample)

        qImg = self.fit_image(lowerPartDefects)
        self.label_2.setPixmap(QtGui.QPixmap.fromImage(qImg))

        self.list_defects(the_list)
        self.textEdit.setText(self.line)

        upperPartDefects, the_list = self.cam_num.find_defects(upper_sample)

        qImg = self.fit_image(upperPartDefects)
        self.label_4.setPixmap(QtGui.QPixmap.fromImage(qImg))

        self.list_defects(the_list)
        self.textEdit_2.setText(self.line)

    def list_defects(self,the_list):
        self.line = ""
        i = 1
        for x in the_list:
            self.line += "Size of defect %d = %d\n" % (i, x)
            i = i + 1
        return self.line

    def fit_image(self,image):
        h, w, ch = image.shape
        bytesPerLine = ch * w
        qImg = QtGui.QImage(image.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
        return qImg







