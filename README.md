# 简介
这是一个基于https://github.com/ljl86092297/Dinomaly  复现 并在MVTEC-AD数据集上推理并可视化结果， 本人电脑RTX3050 平均每张图在15ms以下。  
工业缺陷检测
原论文地址：
https://arxiv.org/pdf/2405.14325

# 环境
python==3.8

pip install -r requirements.txt

# 数据下载：MVTEC-AD
因为我只基于了这个数据进行训练 如果你想尝试其他数据集的操作可以跳转原作者查看其他数据集下载操作 https://github.com/ljl86092297/Dinomaly
Download the MVTec-AD dataset from [URL](https://www.mvtec.com/company/research/datasets/mvtec-ad).
Unzip the file to `../mvtec_anomaly_detection`.
```
|-- mvtec_anomaly_detection
    |-- bottle
    |-- cable
    |-- capsule
    |-- ....
```
# 预训练模型下载：
https://github.com/ljl86092297/Anomaly_Detection/releases/tag/v1.0.0 将下好的权重文件放在weight下即可

# 推理并可视化
python inference7visualize.py --data_path 数据集目录

结果就会保存在visualize目录下

# 随机挑选结果展示

![image](https://github.com/ljl86092297/Anomaly_Detection/assets/59467994/cf094eb7-ba85-4205-a9a6-c037b1920a84)
![image](https://github.com/ljl86092297/Anomaly_Detection/assets/59467994/8b08f4d5-47e1-451b-a5a2-5df540618413)
![image](https://github.com/ljl86092297/Anomaly_Detection/assets/59467994/fc4374ae-9ce7-4497-ace7-6422729c9739)



