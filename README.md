# Car_ReIdentification_application

## Technology Stack

- pytorch(视频检测和reid部分)
- pyqt(软件基本界面)

## UI design

- to be decided

## Workflow

1. 用户导入监控视频
    - 创建临时的数据集文件夹
    - 软件通过视频检测模块检测视频中的车辆，保存至临时文件夹中形成gallery数据集
2. 用户导入目标车辆图像
3. 重识别模块根据目标车辆图像在gallery数据集中找到正确的车辆，生成结果文件
4. 根据结果文件和用户进一步的筛选结果(color, Vtype)可视化最终结果