#coding:utf8
import os
import torch
import torch as t
# import models
from dataset import ODIR
from torch.utils.data import DataLoader
from tqdm import tqdm
from SE_resnet34 import SE_ResNet34

import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
@t.no_grad()
def test(kwargs):
    # 模型
    model = SE_ResNet34().eval()
    model = torch.nn.DataParallel(model)
    if kwargs["load_model_path"]:
        model.load_state_dict(t.load(kwargs["load_model_path"],map_location = 'cpu'))
        # model.load_state_dict(t.load(kwargs["load_model_path"]))
    model.to(device)
    # 数据
    train_data = ODIR(kwargs["test_data_root"],test=True,left=kwargs["left"])
    test_dataloader = DataLoader(train_data,batch_size=kwargs["batch_size"],shuffle=False,num_workers=kwargs["num_workers"])
    results = []
    print('start test')
    for ii,(data,path) in tqdm(enumerate(test_dataloader)):
        input = data.to(device)
        score = model(input)
        path = list(path)
        probability = torch.sigmoid(score).detach().tolist()
        #四舍五入
        # for i in range(len(probability)):
        #     for j in range(len(probability[i])):
        #         probability[i][j] = round(probability[i][j])

        batch_results = [(path_,format(probability_[0],"0.1f"),format(probability_[1],"0.1f"),format(probability_[2],"0.1f"),format(probability_[3],"0.1f"),format(probability_[4],"0.1f"),format(probability_[5],"0.1f"),format(probability_[6],"0.1f"),format(probability_[7],"0.1f")) for path_, probability_ in zip(path, probability)]
        results += batch_results

    write_csv(results,kwargs["result_file"])
    print('test finish')
    return results

def write_csv(results,file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','N','D','G','C','A','H','M','O'])
        writer.writerows(results)
    
def train(kwargs):
    # step1: configure model
    model = SE_ResNet34()
    model = torch.nn.DataParallel(model)
    if kwargs["load_model_path"]:
        model.load_state_dict(t.load(kwargs["load_model_path"]))
        print('model loaded')
    model.to(device)
    
    # step2: data
    train_data = ODIR(kwargs["train_data_root"],test=False,left=kwargs["left"])
    val_data = ODIR(kwargs["train_data_root"],test=False,left=kwargs["left"])
    train_dataloader = DataLoader(train_data,kwargs["batch_size"],shuffle=True,num_workers=kwargs["num_workers"])
    val_dataloader = DataLoader(val_data,kwargs["batch_size"],shuffle=False,num_workers=kwargs["num_workers"])
    print('Data processing completed')
    
    # step3: criterion and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    lr = kwargs["lr"]
    weight_decay = kwargs["weight_decay"]
    optimizer = t.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # train
    print('start train')
    for epoch in range(kwargs["max_epoch"]):
        # loss_meter.reset()
        # confusion_matrix.reset()
        for ii,(data,label) in tqdm(enumerate(train_dataloader)):
            # train model 
            input = data.to(device)
            target = label.to(device)
            target = target.float()
            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score,target)
            loss.backward()
            optimizer.step()
            print("Epoch: %d, Step: %d, Loss: %f"%(epoch,ii,loss))
            
        #存储模型
        if epoch / 5 == 0:
            prefix = kwargs["train_data_root"] + 'checkpoints/ResNet34' + '_'
            # name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
            name = prefix + 'epoch' + epoch
            t.save(model.state_dict(), name)
            print('model saved')

        # 如果损失不再下降，则降低学习率
        # if loss_meter.value()[0] > previous_loss:
        #     lr = lr * kwargs["lr_decay"]
        #     # 第二种降低学习率的方法:不会有moment等信息的丢失
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        # previous_loss = loss_meter.value()[0]


##本机端
if __name__=='__main__':

   # kwargs = {
   #     "train_data_root":"/home/aaa/PycharmProjects/fgd/",
   #     "load_model_path":False,#'/home/aaa/PycharmProjects/fgd/checkpoints/ResNet34_0911_15:49:49.pth',
   #     "lr":0.005,
   #     "lr_decay":0.5,
   #     "weight_decay":0e-5,
   #     "batch_size":32,
   #     "num_workers":4,
   #     "max_epoch":50,
   #     "print_freq":20,
   #     "left":True
   #     }
   # train(kwargs)
    
    # kwargs = {
    #     "test_data_root":"/home/aaa/PycharmProjects/fgd/",
    #     "load_model_path":'/home/aaa/PycharmProjects/fgd/checkpoints/left/ResNet34_epoch250.pth',
    #     "batch_size":128,
    #     "result_file":'result_left.csv',
    #     "num_workers":12,
    #     "left": True
    #  }
    # test(kwargs)

    kwargs = {
        "test_data_root": "/home/aaa/PycharmProjects/fgd/",
        "load_model_path": '/home/aaa/PycharmProjects/fgd/checkpoints/right/ResNet34_epoch300.pth',
        "batch_size": 128,
        "result_file": 'result_right.csv',
        "num_workers": 12,
        "left": False
    }
    test(kwargs)