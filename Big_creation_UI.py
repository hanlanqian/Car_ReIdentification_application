# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Big_creation_UI.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1420, 860)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout_4.addWidget(self.pushButton_2)
        self.pushButton_9 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_9.setObjectName("pushButton_9")
        self.verticalLayout_4.addWidget(self.pushButton_9)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout_4.addWidget(self.pushButton)
        self.horizontalLayout_5.addLayout(self.verticalLayout_4)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_5.addWidget(self.label_2)
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout_5.addWidget(self.lineEdit)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.pushButton_11 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_11.setObjectName("pushButton_11")
        self.verticalLayout.addWidget(self.pushButton_11)
        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setObjectName("pushButton_6")
        self.verticalLayout.addWidget(self.pushButton_6)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.comboBox_2 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_2.setObjectName("comboBox_2")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.comboBox_2)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setObjectName("label_5")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.comboBox_3 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_3.setObjectName("comboBox_3")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.comboBox_3)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setObjectName("label_6")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.comboBox_4 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_4.setObjectName("comboBox_4")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.comboBox_4)
        self.verticalLayout.addLayout(self.formLayout)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_7.addWidget(self.label_3)
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.horizontalLayout_7.addWidget(self.comboBox)
        self.verticalLayout.addLayout(self.horizontalLayout_7)
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setObjectName("pushButton_4")
        self.verticalLayout.addWidget(self.pushButton_4)
        self.horizontalLayout_3.addLayout(self.verticalLayout)
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.pushButton_10 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_10.setObjectName("pushButton_10")
        self.verticalLayout_8.addWidget(self.pushButton_10)
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setObjectName("graphicsView")
        self.verticalLayout_8.addWidget(self.graphicsView)
        self.horizontalLayout_3.addLayout(self.verticalLayout_8)
        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(1, 9)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout_2.addWidget(self.progressBar)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton_8 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_8.setObjectName("pushButton_8")
        self.horizontalLayout_2.addWidget(self.pushButton_8)
        self.pushButton_7 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_7.setObjectName("pushButton_7")
        self.horizontalLayout_2.addWidget(self.pushButton_7)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setObjectName("textBrowser")
        self.horizontalLayout.addWidget(self.textBrowser)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.verticalLayout_3.addLayout(self.verticalLayout_2)
        self.verticalLayout_3.setStretch(0, 2)
        self.verticalLayout_3.setStretch(1, 1)
        self.horizontalLayout_4.addLayout(self.verticalLayout_3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1420, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_2.setText(_translate("MainWindow", "载入视频"))
        self.pushButton_9.setText(_translate("MainWindow", "载入图片集"))
        self.pushButton.setText(_translate("MainWindow", "添加查询车辆"))
        self.label_2.setText(_translate("MainWindow", "摄像头id:"))
        self.pushButton_11.setText(_translate("MainWindow", "查看数据集"))
        self.pushButton_6.setText(_translate("MainWindow", "车辆重识别"))
        self.label_4.setText(_translate("MainWindow", "颜色"))
        self.label_5.setText(_translate("MainWindow", "车型"))
        self.label_6.setText(_translate("MainWindow", "TextLabel"))
        self.label_3.setText(_translate("MainWindow", "Rank"))
        self.comboBox.setItemText(0, _translate("MainWindow", "1"))
        self.comboBox.setItemText(1, _translate("MainWindow", "5"))
        self.comboBox.setItemText(2, _translate("MainWindow", "10"))
        self.pushButton_4.setText(_translate("MainWindow", "可视化结果"))
        self.pushButton_10.setText(_translate("MainWindow", "打开原始图片"))
        self.pushButton_8.setText(_translate("MainWindow", "暂停进程"))
        self.pushButton_7.setText(_translate("MainWindow", "停止进程"))
        self.label.setText(_translate("MainWindow", "状后台态"))
