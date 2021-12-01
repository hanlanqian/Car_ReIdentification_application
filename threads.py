from PyQt5.QtCore import QThread, pyqtSignal
from video_process.detect_frame import process_video
from car_reid.inference import inference
from car_reid.plot_rank import N_rank, img_save
from car_reid.configs.default_config import cfg

import configparser
import os

conf = configparser.ConfigParser()
conf_path = './app.conf'


class process_video_thread(QThread):
    process_video_signal = pyqtSignal(int)
    info_signal = pyqtSignal(str)

    def __init__(self, fname, camera_id):
        super(process_video_thread, self).__init__()
        self.fname = fname
        self.camera_id = camera_id

    def run(self) -> None:
        conf.read(conf_path)
        self.info_signal.emit("处理视频线程已启动")
        process_video(self.fname, conf, [self.process_video_signal, self.info_signal], self.camera_id,
                      num=conf.getint('video_process', 'FRAME_INTERVAL'))


class reid_thread(QThread):
    process_video_signal = pyqtSignal(int)
    info_signal = pyqtSignal(str)

    def __init__(self):
        super(reid_thread, self).__init__()

    def run(self) -> None:
        conf.read(conf_path)
        self.info_signal.emit("重识别线程已启动")
        inference(conf, cfg, [self.process_video_signal, self.info_signal])


class show_result_thread(QThread):
    info_signal = pyqtSignal(str)

    def __init__(self, rank_num):
        super(show_result_thread, self).__init__()
        self.rank_num = rank_num

    def run(self) -> None:
        self.info_signal.emit("可视化结果线程已启动")
        image_path, image_query_path = N_rank(self.rank_num,
                                              os.path.join(conf.get('reid', 'OUTPUT'), 'test_output.pkl'))
        img_save(image_query_path, image_path, self.rank_num, conf.get('reid', 'PICTURE_PATH'))
