import sys
import os
import shutil
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QGraphicsScene, QMessageBox, QStyleFactory
from PyQt5.QtGui import QPixmap
from Big_creation_UI import Ui_MainWindow
from threads import process_video_thread, reid_thread, show_result_thread
import configparser

conf = configparser.ConfigParser()
conf.read('./app.conf')


class Mymain(QMainWindow):
    def __init__(self, parent=None):
        super(Mymain, self).__init__(parent)
        self.setObjectName('MainWindow')
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setStyle(QStyleFactory.create('Fusion'))
        self.init_ui()
        with open('./AppStyleSheet.css', 'r') as f:
            self.setStyleSheet(f.read())

    def init_ui(self):
        self.ui.pushButton_2.clicked.connect(self.process_video_button)
        self.ui.pushButton_4.clicked.connect(self.show_reid_result)
        self.ui.pushButton_6.clicked.connect(self.reid_process_button)
        self.ui.pushButton_8.clicked.connect(self.pause_process)
        self.ui.pushButton_9.clicked.connect(self.load_pictures)
        self.ui.pushButton_10.clicked.connect(self.open_result_pic)
        self.ui.pushButton_11.clicked.connect(self.show_datasets)
        self.ui.pushButton.clicked.connect(self.add_query_vehicle)

    def process_video_button(self):
        fname, flag = QFileDialog.getOpenFileName(self, '载入视频文件', '.')
        if flag:
            if not self.ui.lineEdit.text():
                QMessageBox.warning(self, "视频处理警告", "请输入视频摄像头id后再进行视频处理！")
            else:
                self.pv_thread = process_video_thread(fname, self.ui.lineEdit.text())
                self.pv_thread.process_video_signal.connect(self.process_bar_value)
                self.pv_thread.info_signal.connect(self.show_information)
                self.pv_thread.start()

    def show_datasets(self):
        datasets_abs_path = os.path.abspath(conf.get('video_process', 'OUTPUT'))
        os.startfile(datasets_abs_path)

    def process_bar_value(self, value):
        self.ui.progressBar.setValue(value)

    def show_information(self, info: str):
        self.ui.textBrowser.append(info)
        self.ui.textBrowser.moveCursor(self.ui.textBrowser.textCursor().End)

    def add_query_vehicle(self):

        fnames, flag = QFileDialog.getOpenFileNames(self, "添加查询车辆图像", '')
        if flag:
            if not self.ui.lineEdit.text():
                QMessageBox.warning(self, "警告", "未设定摄像头id，将使用默认设摄像头id: 1")
                camera_id = 1
            else:
                camera_id = self.ui.lineEdit.text()
            query_folder = os.path.join(conf.get('video_process', 'OUTPUT'), 'image_query')
            if not os.path.exists(query_folder):
                os.makedirs(query_folder)
            for i, fname in enumerate(fnames):
                shutil.copy(fname, os.path.join(query_folder, f"{i}_{camera_id}_c{i}_.jpg"))

    def cancel_process(self):
        if self.pv_thread:
            pass

    def pause_process(self):
        if not (getattr(self, 'pv_thread') or getattr(self, 'reid_thread')):
            QMessageBox.information(self, '信息', '没有正在运行的线程')

        if self.ui.pushButton_8.text() == '暂停进程':
            if self.pv_thread.isRunning():
                self.pv_thread.wait()
            if self.reid_thread.isRunning():
                self.reid_thread.wait()
            self.ui.pushButton_8.setText('开始进程')
        elif self.ui.pushButton_8.text() == '开始进程':
            if not self.pv_thread.isRunning():
                self.pv_thread.start()
            if not self.reid_thread.isRunning():
                self.reid_thread.start()
            self.ui.pushButton_8.setText('暂停进程')

    def reid_process_button(self):
        self.reid_thread = reid_thread()
        self.ui.progressBar.setValue(0)
        self.reid_thread.info_signal.connect(self.show_information)
        self.reid_thread.process_video_signal.connect(self.process_bar_value)
        self.reid_thread.start()

    def load_pictures(self):
        fnames, flag = QFileDialog.getOpenFileNames(self, "添加待查询数据集", '')
        if flag:
            for fname in fnames:
                shutil.copy(fname, conf.get('video_process', 'OUTPUT'))

    def show_reid_result(self):
        rank_num = int(self.ui.comboBox.currentText()) if self.ui.comboBox.currentText() else 5
        self.show_result_thread = show_result_thread(rank_num=rank_num)
        self.show_result_thread.start()
        self.show_result_thread.wait()
        sence = QGraphicsScene()
        Qimg = QPixmap(conf.get('reid', 'PICTURE_PATH'))
        sence.addPixmap(Qimg)
        self.ui.graphicsView.setScene(sence)
        self.ui.graphicsView.show()

    def open_result_pic(self):
        result_pic_path = os.path.abspath(conf.get('reid', 'PICTURE_PATH'))
        os.startfile(result_pic_path)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = Mymain()
    main_window.show()
    sys.exit(app.exec_())
