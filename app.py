import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QWidget
from PyQt5.QtCore import QThread, pyqtSignal
from Big_creation_UI import Ui_MainWindow
from threads import process_video_thread
import configparser

conf = configparser.ConfigParser()
conf.read('./app.conf')


class Mymain(QMainWindow):
    def __init__(self, parent=None):
        super(Mymain, self).__init__(parent)
        self.setObjectName('MainWindow')
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.init_ui()

    def init_ui(self):
        # self.process_video = process_video_thread()
        self.ui.pushButton_2.clicked.connect(self.process_video_button)
        self.ui.pushButton_11.clicked.connect(self.show_datasets)
        # self.process_video.start()

    def process_video_button(self):
        fname, flag = QFileDialog.getOpenFileName(self, '载入视频文件', '.')
        if flag:
            self.pv_thread = process_video_thread(fname)
            self.pv_thread.process_video_signal.connect(self.process_video_schedule)
            self.pv_thread.info_signal.connect(self.show_information)
            self.pv_thread.start()

    def show_datasets(self):
        datasets_abs_path = os.path.abspath(conf.get('video_process', 'OUTPUT'))
        os.startfile(datasets_abs_path)



    def process_video_schedule(self, value):
        self.ui.progressBar.setValue(value)

    def show_information(self, info: str):
        self.ui.textBrowser.append(info)
        self.ui.textBrowser.moveCursor(self.ui.textBrowser.textCursor().End)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = Mymain()
    main_window.show()
    sys.exit(app.exec_())
