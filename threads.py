from PyQt5.QtCore import QThread, pyqtSignal
from video_process.detect_frame import process_video
import configparser

conf = configparser.ConfigParser()
conf.read('./app.conf')


class process_video_thread(QThread):
    process_video_signal = pyqtSignal(int)
    info_signal = pyqtSignal(str)

    def __init__(self, fname):
        super(process_video_thread, self).__init__()
        self.fname = fname

    def run(self) -> None:
        process_video(self.fname, conf.get('video_process', 'OUTPUT'), conf,
                      [self.process_video_signal, self.info_signal])


class reid_thread(QThread):
    info_signal = pyqtSignal(str)

    def __init__(self):
        super(reid_thread, self).__init__()

    def run(self) -> None:
        pass
