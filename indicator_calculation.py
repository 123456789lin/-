import openpyxl
import csv
import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment
from sklearn import metrics

left_eye = False
root = '/home/aaa/PycharmProjects/fgd/'
# 输入文件路径,以/结尾
if left_eye:
    root_excel = root+'ODIR-5K_training/testingset_left_eye.xlsx'
    # root_excel = root + 'ODIR-5K_training/left_eye.xlsx'
    root_csv = root + 'result_left.csv'
else:
    root_excel = root+'ODIR-5K_training/testingset_right_eye.xlsx'
    # root_excel = root + 'ODIR-5K_training/right_eye.xlsx'
    root_csv = root + 'result_right.csv'

# root_excel = root + 'ODIR-5K_training/testingset.xlsx'
# root_csv = root + 'Final_result.csv'

two_list = []
one_list = []

two_list_pred = []
one_list_pred = []
#read excel
wb = openpyxl.load_workbook(root_excel)
sheet1 = wb.get_sheet_by_name('Sheet1')
for column in sheet1.rows:
    for i in range(7,15):
        one_list.append(column[i].value)
    two_list.append(one_list.copy())
    one_list.clear()
del two_list[0] #删除二维列表中的第一行
#read csv
data = pd.read_csv(root_csv)
for i in data.values:
    # print(i)#与上边不同的是从数据区域开始读取的。
    for j in i:
        one_list_pred.append(j)
    two_list_pred.append(one_list_pred.copy())
    one_list_pred.clear()
two_list_pred = np.delete(two_list_pred,0,axis=1).tolist() #删除二维列表中的第一列


temp1 = np.array(two_list_pred)
temp2 = np.array(two_list)
a,b = temp1.shape
c,d = temp2.shape
if a == c and b==d:
    #N,D,G,C,A,H,M,O
    true_label = [0,0,0,0,0,0,0,0]
    kappa_label = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    auc_ture = []
    auc_ture_auxiliary = []
    auc_pred = []
    auc_pred_auxiliary = []
    for i in range(len(two_list_pred)):
        for j in range(8):#N,D,G,C,A,H,M,O
            #统计每类分类正确的样本数
            if round(two_list_pred[i][j]) == round(two_list[i][j]):
                true_label[j] += 1
            #统计混淆矩阵
            #正负(P/N)样本,正确错误(T/F)
            # TP FP
            # FN TN
            if round(two_list_pred[i][j]) == 1 and round(two_list[i][j]) ==1:kappa_label[j][0] += 1
            if round(two_list_pred[i][j]) == 1 and round(two_list[i][j]) == 0: kappa_label[j][1] += 1
            if round(two_list_pred[i][j]) == 0 and round(two_list[i][j]) == 1: kappa_label[j][2] += 1
            if round(two_list_pred[i][j]) == 0 and round(two_list[i][j]) == 0: kappa_label[j][3] += 1
    #为计算auc而统计的标签
    for i in range(8):
        for p in two_list_pred:
            auc_pred_auxiliary.append(p[i])
        auc_pred.append(auc_pred_auxiliary.copy())
        auc_pred_auxiliary.clear()
    for i in range(8):
        for p in two_list:
            auc_ture_auxiliary.append(p[i])
        auc_ture.append(auc_ture_auxiliary.copy())
        auc_ture_auxiliary.clear()

    # 计算各个指标
    total_num = len(two_list_pred)
    label_str = ['N','D','G','C','A','H','M','O']
    if left_eye:print('left_eye:')
    else: print('right_eye:')
    kappa_sum = 0
    f1_sum = 0
    auc_sum = 0
    for i in range(8):
        #计算kappa所需要的元素
        diagonal = kappa_label[i][0]+kappa_label[i][3]
        gt_T = kappa_label[i][0]+kappa_label[i][2]
        gt_F = kappa_label[i][1]+kappa_label[i][3]
        pred_T = kappa_label[i][0]+kappa_label[i][1]
        pred_F = kappa_label[i][2]+kappa_label[i][3]
        P_0 = diagonal/total_num
        P_e = (pred_T*gt_T +pred_F*gt_F)/(total_num*total_num)
        #计算准确率与召回率
        correct_rate = (kappa_label[i][0]+kappa_label[i][3]) / (kappa_label[i][0]+kappa_label[i][1]+kappa_label[i][2]+kappa_label[i][3])
        recall_rate = kappa_label[i][0] / (kappa_label[i][0]+kappa_label[i][2])
        #计算kappa系数
        kappa = (P_0-P_e)/(1-P_e)
        #计算F_1分数
        f1_score = 2*((correct_rate*recall_rate)/(correct_rate + recall_rate))
        #计算AUC面积，调用sklearn库的函数来实现
        y = np.array(auc_ture[i])
        scores = np.array(auc_pred[i])
        auc = metrics.roc_auc_score(y,scores)
        #输出指标的结果
        print('label:%s, kappa:%0.3f, F1_score:%0.3f, AUC:%0.3f, Confusion matrix:'
              %(label_str[i],kappa,f1_score,auc))
        print(kappa_label[i][0],kappa_label[i][1])
        print(kappa_label[i][2],kappa_label[i][3])
        kappa_sum += kappa
        f1_sum += f1_score
        auc_sum += auc
    ka_avg = kappa_sum/8
    f1_avg = f1_sum/8
    auc_avg = auc_sum/8
    final = (ka_avg+f1_avg+auc_avg)/3
    print('-----------------------------统计得分-------------------------------')
    print('kappa:%0.3f, F1_score:%0.3f, AUC:%0.3f'%(ka_avg,f1_avg,auc_avg))
    print('Final_score:{}'.format(final))
else:
    print('Error:Matrix size does not match')
