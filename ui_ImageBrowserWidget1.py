# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets

"被jiemian调用 显示图片的缩略图 并且用滚动条"

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(400, 574)
        self.sliderScale = QtWidgets.QSlider(Form)
        self.sliderScale.setGeometry(QtCore.QRect(20, 0, 361, 22))
        self.sliderScale.setOrientation(QtCore.Qt.Horizontal)
        self.sliderScale.setObjectName("sliderScale")
        self.listWidgetImages = QtWidgets.QListWidget(Form)
        self.listWidgetImages.setGeometry(QtCore.QRect(10, 30, 371, 501))
        self.listWidgetImages.setDragEnabled(True)
        self.listWidgetImages.setMovement(QtWidgets.QListView.Static)
        self.listWidgetImages.setFlow(QtWidgets.QListView.LeftToRight)
        self.listWidgetImages.setResizeMode(QtWidgets.QListView.Adjust)
        self.listWidgetImages.setViewMode(QtWidgets.QListView.IconMode)
        self.listWidgetImages.setModelColumn(0)
        self.listWidgetImages.setObjectName("listWidgetImages")
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(10, 540, 113, 32))
        self.pushButton.setObjectName("pushButton")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.pushButton.setText(_translate("Form", "show result"))
