1. 运行程序前首先要装好库，可以执行这条语句完成：pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

2. 要将yolo模型的权重文件yolo4_weights.pth下载好放入model_data文件夹内，下载地址为链接：https://pan.baidu.com/s/1sG9VdVt_HR-mi53UcII3ZA 提取码：n7uc

3. 准备好运行环境后，将要检测的图片或视频放入inference文件夹内，运行detect_1.py或者detect_2.py，两者的区别为：前者先进行车辆检测，将检测结果保存后再将读取进行车牌识别；
后者是在进行车辆检测的同时将图片直接用于车牌识别

4. 运行结果为车辆检测的结果图片保存到inference/output文件夹内，同时生成一个包括车辆检测和车牌识别结果的csv文件