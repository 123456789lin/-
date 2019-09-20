from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import openpyxl
import csv
import pandas as pd
import numpy as np
import pdb

# 选择调参模式为True, 得到测试结果选False
Debug = True

root = '/home/aaa/PycharmProjects/fgd/'

root_excel = root + 'ODIR-5K_training/ODIR-5K_training-Chinese.xlsx'
# root_excel = root + 'ODIR-5K_training/testingset.xlsx'
root_csv_left = root + 'result_left.csv'
root_csv_right = root + 'result_right.csv'
if Debug:pass
else:
    root_test_left = root + 'result_left.csv'
    root_test_right = root + 'result_right.csv'

two_list = []
one_list = []

two_list_pred_left = []
one_list_pred_left = []
two_list_pred_right = []
one_list_pred_right = []

#read excel
wb = openpyxl.load_workbook(root_excel)
sheet1 = wb.get_sheet_by_name('Sheet1')
for column in sheet1.rows:
    for i in range(3):
        one_list.append(column[i].value)
    for i in range(7,15):
        one_list.append(column[i].value)

    two_list.append(one_list.copy())
    one_list.clear()
del two_list[0] #删除二维列表中的第一行

# read csv
#左眼结果
data = pd.read_csv(root_csv_left)
for i in data.values:
# print(i)#与上边不同的是从数据区域开始读取的。
    for j in i:
        one_list_pred_left.append(j)
    two_list_pred_left.append(one_list_pred_left.copy())
    one_list_pred_left.clear()
id = np.array(two_list_pred_left)[:, 0].copy() #保留第一列信息
two_list_pred_left = np.delete(two_list_pred_left,0,axis=1).tolist() #删除二维列表中的第一列
#右眼结果
data = pd.read_csv(root_csv_right)
for i in data.values:
    # print(i)#与上边不同的是从数据区域开始读取的。
    for j in i:
        one_list_pred_right.append(j)
    two_list_pred_right.append(one_list_pred_right.copy())
    one_list_pred_right.clear()
ids = np.array(two_list_pred_right)[:, 0].copy() #保留第一列信息
two_list_pred_right = np.delete(two_list_pred_right,0,axis=1).tolist() #删除二维列表中的第一列

for i in range(len(id)):
    if id[i] != ids[i]:
        print('Erroe:left and right eye data is different')
        pdb.set_trace()
#如果非Debug 还要读入测试集结果
if Debug:pass
else:
    one_list_test_left = []
    two_list_test_left = []
    one_list_test_right = []
    two_list_test_right = []
    # 左眼结果
    data = pd.read_csv(root_test_left)
    for i in data.values:
        # print(i)#与上边不同的是从数据区域开始读取的。
        for j in i:
            one_list_test_left.append(j)
        two_list_test_left.append(one_list_test_left.copy())
        one_list_test_left.clear()
    id = np.array(two_list_test_left)[:, 0].copy()  # 保留第一列信息
    two_list_test_left = np.delete(two_list_test_left, 0, axis=1).tolist()  # 删除二维列表中的第一列
    # 右眼结果
    data = pd.read_csv(root_test_right)
    for i in data.values:
        # print(i)#与上边不同的是从数据区域开始读取的。
        for j in i:
            one_list_test_right.append(j)
        two_list_test_right.append(one_list_test_right.copy())
        one_list_test_right.clear()
    ids = np.array(two_list_test_right)[:, 0].copy()  #保留第一列信息
    two_list_test_right = np.delete(two_list_test_right, 0, axis=1).tolist()  #删除二维列表中的第一列


# print(two_list)#编号，年龄，性别，标签*8
# print(two_list_pred)#标签*8

#数据处理
two_array_excel = np.array(two_list)
two_array_pred_left = np.array(two_list_pred_left)
two_array_pred_right = np.array(two_list_pred_right)
if Debug:pass
else:
    two_array_test_left = np.array(two_list_test_left)
    two_array_test_right = np.array(two_list_test_right)
a, _ = two_array_excel .shape
c, _ = two_array_pred_left.shape
if a == c:
    #数据
    # input_age_gender = two_array[:,1:3].copy()
    #把 年龄 性别 标签信息拼接在一起
    # input_data = np.hstack((input_age_gender, two_array_pred.copy()))
    # print(input_data.shape)
    pred_rusult = []
    score_list = []
    for j in range(3,11):
        input_label = two_array_excel[:, j].copy()
        input_data = np.hstack((two_array_pred_left.copy(),two_array_pred_right.copy()))
        if Debug:
            Xtrain, Xtest, Ytrain, Ytest = train_test_split(input_data, input_label, test_size=0.3)
        else:
            Xtrain = input_data
            Ytrain = input_label
            Xtest = np.hstack((two_array_test_left.copy(),two_array_test_right.copy()))
        #建立树
        clf = tree.DecisionTreeClassifier(criterion='entropy'
                                          ,random_state=30
                                          ,splitter='random'
                                          ,max_depth=10 #树的最大深度为3
                                          ,min_samples_leaf=10#一个节点在分之后的每个子节点包含的样本量小于10怎分支不会发生
                                          ,min_samples_split=10)#一个节点的样本数小于10则不会产生新的分支
        #训练
        clf = clf.fit(Xtrain, Ytrain)
        #测试
        if Debug:
            score = clf.score(Xtest,Ytest)
            print(score)
            score_list.append(score)
            feature_name = ['NL','DL','GL','CL','AL','HL','ML','OL','NR','DR','GR','CR','AR','HR','MR','OR']
            print([*zip(feature_name, clf.feature_importances_)])
        else:
            pred_rusult.append(clf.predict(Xtest))
    #结果处理
    if Debug:
        s = 0
        for i  in range(len(score_list)):
            s += score_list[i]
        print('Final score: %f'%(s/len(score_list)))
    else:
        pred_result = np.array(pred_rusult).T.tolist()
        #在矩阵的第一列插入id
        csv_content = []
        csv_content_auxiliary = []
        for m in range(len(id)):
            csv_content_auxiliary.append(int(id[m]))
            for n in pred_result[m]:
                csv_content_auxiliary.append(n)
            csv_content.append(csv_content_auxiliary.copy())
            csv_content_auxiliary.clear()
        #排序
        csv_content = sorted(csv_content, key=lambda x: int(x[0]))
        file_name = root +'Final_result.csv'
        with open(file_name, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'N', 'D', 'G', 'C', 'A', 'H', 'M', 'O'])
            writer.writerows(csv_content)
else:
    print('Error:length does not match')

