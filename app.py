import sys
import os
import shutil
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QGraphicsScene, QMessageBox, QStyleFactory, QWidget, \
    QListView
from PyQt5.QtGui import QPixmap, QCloseEvent
from PyQt5.QtCore import Qt
from Big_creation_UI import Ui_MainWindow
from config import Ui_Form
from threads import process_video_thread, reid_thread, show_result_thread
import configparser

conf = configparser.ConfigParser()
conf_path = './app.conf'
conf.read(conf_path)


class Mymain(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(Mymain, self).__init__(parent)
        self.config_window = ConfigWindow()
        self.setObjectName('MainWindow')
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setStyle(QStyleFactory.create('Fusion'))
        self.init_ui()
        self.threads = []
        with open('./AppStyleSheet.css', 'r', encoding='utf-8') as f:
            self.setStyleSheet(f.read())

    def init_ui(self):
        self.ui.pushButton_2.clicked.connect(self.process_video_button)
        self.ui.pushButton_4.clicked.connect(self.show_reid_result)
        self.ui.pushButton_6.clicked.connect(self.reid_process_button)
        self.ui.pushButton_7.clicked.connect(self.cancel_process)
        self.ui.pushButton_9.clicked.connect(self.load_pictures)
        self.ui.pushButton_10.clicked.connect(self.open_result_pic)
        self.ui.pushButton_11.clicked.connect(self.show_datasets)
        self.ui.pushButton.clicked.connect(self.add_query_vehicle)
        self.ui.progressBar.valueChanged.connect(self.check_processbar)
        self.ui.actionSetting.triggered.connect(self.setting)
        list_view1 = QListView()
        list_view2 = QListView()
        list_view3 = QListView()
        self.ui.comboBox.setView(list_view1)
        self.ui.comboBox_2.setView(list_view2)
        self.ui.comboBox_3.setView(list_view3)

    def setting(self):
        self.config_window.init_setting()
        self.config_window.show()

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
            start = len(os.listdir(query_folder))
            for i, fname in enumerate(fnames):
                shutil.copy(fname, os.path.join(query_folder, f"{i + start}_{camera_id}_c{i + start}_.jpg"))
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
            start = len(os.listdir(gallery_folder))
            for i, fname in enumerate(fnames):
                shutil.copy(fname, os.path.join(gallery_folder, f"{i + start}_{camera_id}_c0_.jpg"))

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

    def closeEvent(self, a0: QCloseEvent) -> None:
        reply = QMessageBox.question(self,
                                     '本程序',
                                     "是否要退出程序？",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.config_window.close()
            a0.accept()
        else:
            a0.ignore()


class ConfigWindow(QWidget, Ui_Form):
    def __init__(self):
        super(ConfigWindow, self).__init__()
        self.setObjectName('ConfigWindow')
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setStyle(QStyleFactory.create('Fusion'))
        self.init_ui()
        with open('./config.css', 'r', encoding='utf-8') as f:
            self.setStyleSheet(f.read())

    def init_ui(self):
        self.ui.pushButton.clicked.connect(self.save_setting)
        self.ui.horizontalSlider.valueChanged.connect(lambda: self.show_value(self.ui.label_9))
        self.ui.horizontalSlider_2.valueChanged.connect(lambda: self.show_value(self.ui.label_10))
        self.ui.pushButton_2.clicked.connect(lambda: self.choose_folder(self.ui.label_12))
        self.ui.pushButton_5.clicked.connect(lambda: self.choose_folder(self.ui.label_18))
        self.ui.pushButton_3.clicked.connect(lambda: self.choose_path(self.ui.label_14))
        self.ui.pushButton_4.clicked.connect(lambda: self.choose_path(self.ui.label_16))
        list_view = QListView()
        self.ui.comboBox.setView(list_view)

    def show_value(self, label):
        label.setText(str(self.sender().value() / 100))

    def choose_path(self, label):
        path, flag = QFileDialog.getSaveFileName()
        label.setText(path)

    def choose_folder(self, label):
        dirs = QFileDialog.getExistingDirectory()
        label.setText(dirs)

    def save_setting(self):
        conf.set('video_process', 'DEVICE', self.ui.comboBox.currentText())
        conf.set('video_process', 'FRAME_INTERVAL', str(int(self.ui.lineEdit.text())))
        conf.set('video_process', 'IMAGE_SIZE', str(int(self.ui.lineEdit_2.text())))
        conf.set('video_process', 'CONF_THRES', str(self.ui.horizontalSlider_2.value() / 100))
        conf.set('video_process', 'IOU_THRES', str(self.ui.horizontalSlider.value() / 100))
        conf.set('video_process', 'OUTPUT', self.ui.label_12.text())

        conf.set('reid', 'PKL_PATH', self.ui.label_14.text())
        conf.set('reid', 'CONFIG_FILE', self.ui.label_16.text())
        conf.set('reid', 'OUTPUT', self.ui.label_18.text())
        conf.set('reid', 'ALPHA', self.ui.lineEdit_3.text())
        conf.set('reid', 'BETA', self.ui.lineEdit_6.text())
        conf.set('reid', 'LAMBDA2', self.ui.lineEdit_5.text())
        conf.set('reid', 'LAMBDA1', self.ui.lineEdit_4.text())

        conf.set('default', 'PLOT_DPI', self.ui.lineEdit_7.text())
        if self.ui.checkBox.checkState() == Qt.Checked:
            conf.set('default', 'INFER_FLAG', 'yes')
        else:
            conf.set('default', 'INFER_FLAG', 'no')
        with open('./app.conf', 'w') as f:
            conf.write(f)
        QMessageBox.information(self, '信息', '配置保存成功！')

    def init_setting(self):
        self.ui.comboBox.setCurrentText(conf.get('video_process', 'DEVICE'))
        self.ui.lineEdit.setText(conf.get('video_process', 'FRAME_INTERVAL'))
        self.ui.lineEdit_2.setText(conf.get('video_process', 'IMAGE_SIZE'))
        self.ui.horizontalSlider_2.setValue(conf.getfloat('video_process', 'CONF_THRES') * 100)
        self.ui.horizontalSlider.setValue(conf.getfloat('video_process', 'IOU_THRES') * 100)
        self.ui.label_12.setText(conf.get('video_process', 'OUTPUT'))
        self.ui.label_14.setText(conf.get('reid', 'PKL_PATH'))
        self.ui.label_16.setText(conf.get('reid', 'CONFIG_FILE'))
        self.ui.label_18.setText(conf.get('reid', 'OUTPUT'))
        self.ui.lineEdit_3.setText(conf.get('reid', 'ALPHA'))
        self.ui.lineEdit_6.setText(conf.get('reid', 'BETA'))
        self.ui.lineEdit_5.setText(conf.get('reid', 'LAMBDA2'))
        self.ui.lineEdit_4.setText(conf.get('reid', 'LAMBDA1'))
        self.ui.lineEdit_7.setText(conf.get('default', 'PLOT_DPI'))
        if conf.getboolean('default', 'INFER_FLAG'):
            self.ui.checkBox.setCheckState(Qt.Checked)
        else:
            self.ui.checkBox.setCheckState(Qt.Unchecked)
        if conf.getboolean('reid', 'remove_junk'):
            self.ui.checkBox_2.setCheckState(Qt.Checked)
        else:
            self.ui.checkBox_2.setCheckState(Qt.Unchecked)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = Mymain()
    main_window.show()
    sys.exit(app.exec_())
