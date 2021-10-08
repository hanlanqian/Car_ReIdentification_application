from PyQt5.QtCore import QThread, pyqtSignal
from video_process.detect_2 import process_video
import configparser

conf = configparser.ConfigParser()
conf.read('./app.conf')


class process_video_thread(QThread):
    process_video_signal = pyqtSignal(str, str)

    def __init__(self, fname):
        super(process_video_thread, self).__init__()
        self.fname = fname

    def run(self) -> None:
        process_video(self.fname, conf.get('video_process', 'OUTPUT'), conf)
        # print(f'thread {self.fname}')