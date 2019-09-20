菜鸟 学习 深度学习的第一份代码，是ODIR比赛的，很遗憾最后没有来得及提交结果，伤心。留作纪念吧!

任务介绍：该比赛是一个八标签二分类问题。基于彩色眼底图片进行疾病的分类。
官方网站：https://odir2019.grand-challenge.org/dates/ 可以进行数据集的下载

实现步骤及运行说明：(注意修改路径)
1.由于官方没给验证集，划分训练集和验证集用 “测试机训练集划分.py” ：将训练集中的3500张图片随机分为2500张训练集和1000张测试集，
并在设置的目录下生成相应训练集和测试集xlsx文件及对应的训练集和测试集。
2.采用的是左右眼分开训练然后再用决策树将左右眼的结果合并，“excel_progress.py” ：将对应文件划为左眼和右眼，并分别生成相应的xlsx文件。
3.“dataset.py“：为神经网络输入数据的处理模块。训练时对数据进行随机裁剪区域和随机翻转进行数据増广.
4.”SE_resnet34.py”: 是模型代码部分。使用的网络为在SENet,在Resnet34的每个Block中加了Squeeze和Excitation操作。
5.“main.py”:是代码的训练和测试部分。
6.“”


文件说明：main.py 包含模型的训练和测试代码。
SE_resnet34.py 是模型代码部分。
dataset.py 是数据集载入部分。
测试机训练集划分.py
