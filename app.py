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
        self.threads = []
        with open('./AppStyleSheet.css', 'r') as f:
            self.setStyleSheet(f.read())

    def init_ui(self):
        self.ui.pushButton_2.clicked.connect(self.process_video_button)
        self.ui.pushButton_4.clicked.connect(self.show_reid_result)
        self.ui.pushButton_6.clicked.connect(self.reid_process_button)
        # self.ui.pushButton_8.clicked.connect(self.pause_process)
        self.ui.pushButton_7.clicked.connect(self.cancel_process)
        self.ui.pushButton_9.clicked.connect(self.load_pictures)
        self.ui.pushButton_10.clicked.connect(self.open_result_pic)
        self.ui.pushButton_11.clicked.connect(self.show_datasets)
        self.ui.pushButton.clicked.connect(self.add_query_vehicle)

        self.ui.progressBar.valueChanged.connect(self.check_processbar)

    def process_video_button(self):
        fname, flag = QFileDialog.getOpenFileName(self, '载入视频文件', '.')
        if flag:
            if not self.ui.lineEdit.text():
                QMessageBox.warning(self, "警告", "未设定摄像头id，将使用默认设摄像头id: 1")
                camera_id = 1
            else:
                camera_id = self.ui.lineEdit.text()
            self.pv_thread = process_video_thread(fname, camera_id)
            self.pv_thread.process_video_signal.connect(self.process_bar_value)
            self.pv_thread.info_signal.connect(self.show_information)
            self.pv_thread.start()
            self.threads.append(self.pv_thread)

    def check_processbar(self, value):
        if value == 100:
            QMessageBox.information(self, '信息', '当前线程已完成')

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
            QMessageBox.information(self, '信息', '成功将图片集载入数据集！')

    def cancel_process(self):
        if not self.threads:
            QMessageBox.information(self, '提示', '目前没有正在运行的进程')
        for thread in self.threads:
            if thread.isRunning():
                thread.exit()
            QMessageBox.information(self, '信息', '当前运行进程已停止运行')

    def reid_process_button(self):
        self.reid_thread = reid_thread()
        self.ui.progressBar.setValue(0)
        self.reid_thread.info_signal.connect(self.show_information)
        self.reid_thread.process_video_signal.connect(self.process_bar_value)
        self.reid_thread.start()
        self.threads.append(self.reid_thread)

    def load_pictures(self):
        fnames, flag = QFileDialog.getOpenFileNames(self, "添加待查询数据集", '')
        if flag:
            if not self.ui.lineEdit.text():
                QMessageBox.warning(self, "警告", "未设定摄像头id，将使用默认设摄像头id: 1")
                camera_id = 1
            else:
                camera_id = self.ui.lineEdit.text()
            gallery_folder = os.path.join(conf.get('video_process', 'OUTPUT'), 'image_test')
            if not os.path.exists(gallery_folder):
                os.makedirs(gallery_folder)
            for i, fname in enumerate(fnames):
                shutil.copy(fname, os.path.join(gallery_folder, f"{i}_{camera_id}_c{i}_.jpg"))

        QMessageBox.information(self, '信息', '图片集已成功载入')

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
        if os.path.exists(result_pic_path):
            os.startfile(result_pic_path)
        else:
            QMessageBox.warning(self, '错误', '结果文件不存在')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = Mymain()
    main_window.show()
    sys.exit(app.exec_())
