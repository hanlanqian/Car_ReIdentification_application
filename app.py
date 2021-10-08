import sys
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
        self.ui.pushButton_2.clicked.connect(self.pushButton_2_clicked)
        # self.process_video.start()

    def pushButton_2_clicked(self):
        fname, _ = QFileDialog.getOpenFileName(self, '载入视频文件', '.')
        print(fname)
        self.pv_thread = process_video_thread(fname)
        self.pv_thread.start()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = Mymain()
    main_window.show()
    sys.exit(app.exec_())
