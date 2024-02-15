import datetime
import os
import copy
import sys
import csv
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from tqdm import tqdm

import read_dataset
import log

def main():
    num_epochs = 200


    parser = argparse.ArgumentParser(description='bento train')

    parser.add_argument('--batch_size', help='batch size',default=32)
    parser.add_argument('--in_w', help='in_w',default=128)
    parser.add_argument('--lr', help='lr',default=0.001) 
    parser.add_argument('--weight_decay', help='weight decay',default=0.001)  
    parser.add_argument('--optim', help='optim',default="SGD", type=str)
    parser.add_argument('--seed', help='seed',default= 1)

    args = parser.parse_args()

    print('~~~~~~~~~~ training start ~~~~~~~~~~~~~')
    # ~~~~~~~~~~~~~~~~ param ~~~~~~~~~~~~~~~~~~~
    batch_size = int(args.batch_size)#16
    in_w = int(args.in_w)#128
    in_h = in_w
    lr = float(args.lr)#0.001
    weight_decay = float(args.weight_decay)#0.001
    optim_flg = str(args.optim)
    seed = int(args.seed)

    print('batch_size:',batch_size,
          ', in_w:', in_w, ', lr:', lr, 
          ', weight_decay:', weight_decay, 
          ' ,epoch:', num_epochs)

    # ~~~~~~~~~~~~~~~~ log folder ~~~~~~~~~~~~~~~~~~~~
    log_path = '/home/taki/B4_report/log/bento_train_log/closs_valid/bento-saigen'
    # フォルダ作成
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    param_folder = f'0911_updatedata_optim{optim_flg}_batch{batch_size}_w,h{in_w}_lr{lr}_wDecay{weight_decay}'
    path = os.path.join(log_path, param_folder)
    # フォルダ作成
    if not os.path.exists(path):
        os.mkdir(path)

    path = os.path.join(path, f"seed{seed}")
    if not os.path.exists(path):
        os.mkdir(path)



    # ~~~~~~~~~~~~ set data transforms ~~~~~~~~~~~~~~~
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.Resize((in_w, in_h)),
        transforms.ToTensor()#,
        #transforms.Normalize((0.5,), (0.5,)) #グレースケールのとき
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #これやるとrgb値変になる
    ])

    test_transform = transforms.Compose([
        transforms.Resize((in_w, in_h)),
        transforms.ToTensor()#,
        #transforms.Normalize((0.5,), (0.5,))
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    col = ["彩り", "健康的", "満足感", "ユニークさ", "食べやすさ", "適量", "くずれない"]

    ##################  mydata #####################
    df = pd.read_csv("./score_ml_promax_7_update.csv")
    # 交差検証法(5分割)
    kf = KFold(n_splits=5, shuffle=True, random_state=2020)
    train_list = []
    test_list = []

    # dfをtrainとtestに分ける
    # train_listは800*7のdfが分割数（5個）だけ入っている
    for train_idx, test_idx in kf.split(df):
        train_data = df.iloc[train_idx].reset_index(drop=True)
        test_data = df.iloc[test_idx].reset_index(drop=True)
        train_list.append(train_data)
        test_list.append(test_data)

    #############################  交差検証法 #######################################{{{
    for idx in range(len(train_list)):

        out_path = os.path.join(path, f"closs_valid{idx}")
        if not os.path.exists(out_path):
            os.mkdir(out_path)     

        train_dataset = read_dataset.read_dataset(
            train_list[idx], transform=train_transform)
        test_dataset = read_dataset.read_dataset(
            test_list[idx], transform=test_transform)

        validation_size = len(test_dataset) / 2
        #test_size       = validation_size

        # valをtestと分ける
        test_dataset, val_dataset = torch.utils.data.random_split(
            test_dataset, [int(validation_size), int(validation_size)])

        # # drop_lastで余りを入れるか入れないか
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)  
        val_loader = torch.utils.data.DataLoader(
            val_dataset,   batch_size=25, shuffle=True, num_workers=4, drop_last=False)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,  batch_size=25, shuffle=True, num_workers=4, drop_last=False)


    #~~~~~~~~~~~~~~~~~~~  gpu setup~~~~~~~~~~~~~~~~~~~~~~~~
        random_seed = seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.backends.cudnn.deterministic = True

        # gpuが使えるならgpuを使用、無理ならcpu
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  


        #~~~~~~~~~~~~~~~~~~~  net setup~~~~~~~~~~~~~~~~~~~~~~~~
        net = models.resnet18(pretrained=True)
        #net = models.vgg16(pretrained=True)

        num_ftrs = net.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        net.fc = nn.Linear(num_ftrs, 7)
        net = net.to(device)

        # criterion
        criterion = nn.MSELoss()
        l1_norm   = nn.L1Loss()


        # Observe that all parameters are being optimized
        if optim_flg == "SGD":
            optimizer = optim.SGD(  net.parameters(), 
                                    lr=lr, momentum=0.9, 
                                    weight_decay=weight_decay)
        elif optim_flg == "Adam" :
            optimizer = optim.AdamW(net.parameters(), 
                                    lr=lr,
                                    weight_decay=weight_decay)
        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)


        ############################  training ###################################
        # Training. {{{
        # =====
        history = log.History(keys=('train_loss',
                                    'val_loss',
                                    'epoch_cnt',
                                    'train_L1_loss',
                                    'val_L1_loss',
                                    'test_L1_loss',),
                                    output_dir=out_path)

        #best_model_wts = copy.deepcopy(net.state_dict())
        #best_acc = 0.0

        output_list = []
        input_list  = []
        score_list  = []
        index_list  = []

        min_loss    = 10
        min_L1_loss = 10
        best_model_wts = 0

        for epoch in range(num_epochs):# {{{epoch
            loop = tqdm(train_loader, unit='batch',desc='Epoch {:>3}'.format(epoch+1))

            # Train Step. {{{
            # =====

            # test,trainで使用されたりされないモードがあるので気を付ける
            net.train()
            for _, batch in enumerate(loop):
                # print(batch.shape)
                G_meter = log.AverageMeter()
                inputs, score, _ = batch

                # gpuに送る
                inputs = inputs.to(device)
                score = score.to(device)

                # Update network. {{{
                # =====
                # forward network
                outputs = net(inputs)

                # backward network{{{
                optimizer.zero_grad()  # 勾配を０
                loss = criterion(outputs, score)
                l1_loss = l1_norm(outputs, score)
                loss.backward()
                optimizer.step()
                # }}}backward network

                # Get losses. {{{}}
                G_meter.update(loss.item(), inputs[0].size()[0])
                history({'train_loss': loss.item()})
                history({'train_L1_loss': l1_loss.item()})
                    # }}} get loss
                
                # }}}Update network

            # Print training log. {
            # =====
            msg = "[Train {}] Epoch {}/{}".format(
                'ResNet18', epoch + 1, num_epochs)
            msg += " - {}: {:.4f}".format('train_loss', G_meter.avg)
            msg += " - {}: {:.4f}".format('learning rate',
                                        scheduler.get_last_lr()[0])
            history({'epoch_cnt': epoch})

            print(msg)

            # }}}Train Step.

            # Validation Step. {
            # =====

            with torch.no_grad():  # 勾配
                net.eval()
                loop_val = tqdm(val_loader, unit='batch',desc='Epoch {:>3}'.format(epoch + 1))
                epoch_loss    = 0
                epoch_L1_loss = 0
                iter_cnt      = 0

                for _, batch in enumerate(loop_val):
                    iter_cnt = iter_cnt + 1
                    G_meter = log.AverageMeter()
                    inputs, score, index = batch

                    inputs = inputs.to(device)
                    score = score.to(device)
                    index  = index.numpy().copy()

                    outputs = net(inputs)

                    loss = criterion(outputs, score)
                    l1_loss = l1_norm(outputs, score)

                    epoch_loss    = epoch_loss + loss
                    epoch_L1_loss = epoch_L1_loss + l1_loss

                    G_meter.update(loss.item(), inputs[0].size()[0])
                    history({'val_loss': loss.item()})
                    history({'val_L1_loss': l1_loss.item()})

                    inputs_img  = inputs.to('cpu').detach().numpy().copy()
                    output = outputs.to('cpu').detach().numpy().copy()
                    score  = score.to('cpu').detach().numpy().copy()

                # 50epoch毎にレーダーチャート保存{{
                if epoch % 50 == 0:
                        # フォルダ作成
                    val_path = out_path + f"/{epoch}_epoch_result"
                    if not os.path.exists(val_path):
                        os.mkdir(val_path)
                    output_list.extend(output)
                    input_list.extend(inputs_img)
                    score_list.extend(score)
                    index_list.extend(index)
                    history.radar_chart(input_img=input_list[:10],
                                        output=output_list[:10], 
                                        score=score_list[:10], 
                                        dataframe=test_list[idx], 
                                        index_list=index_list,
                                        save_path=val_path,
                                        filename='epoch_radar_chart.png')
                #}}
            # deep copy the model{{{
            epoch_loss_mean    = epoch_loss / iter_cnt
            epoch_L1_loss_mean = epoch_L1_loss / iter_cnt

            if epoch_loss_mean < min_loss:
                min_loss = epoch_loss_mean
                best_model_wts = copy.deepcopy(net.state_dict())

            if epoch_L1_loss_mean < min_L1_loss:
                min_L1_loss = epoch_L1_loss_mean
            #}}}

            # } val step

            # Print validation log. {
            # =====
            msg = "[Validation {}] Epoch {}/{}".format(
                'CNN', epoch + 1, num_epochs)
            msg += " - {}: {:.4f}".format('val_loss', G_meter.avg)

            print(msg)
            # } val log

            # sheduler step
            scheduler.step()
        # }}}} epoch
        
        # 重み保存
        torch.save(best_model_wts, out_path+"/model_dict.pth")

        output_list = []
        score_list  = []
        index_list  = []
        # ~~~~~~~~~~~~~~ testdataに対する推論 ~~~~~~~~~~~~~~~~~~~~~
        print("~~~~~~~~~~~~~~ eval test data ~~~~~~~~~~~~~~~~~~")
        with torch.no_grad():  # 勾配の消失
            for data in test_loader:
                inputs, score, index = data

                inputs = inputs.to(device)
                score = score.to(device)

                index  = index.numpy().copy()

                outputs = net(inputs)

                l1_loss = l1_norm(outputs, score)
                history({'test_L1_loss': l1_loss.item()})
                
                inputs_img  = inputs.to('cpu').detach().numpy().copy()
                output = outputs.to('cpu').detach().numpy().copy()
                score  = score.to('cpu').detach().numpy().copy()

                input_list.extend(inputs_img)
                output_list.extend(output)
                score_list.extend(score)
                index_list.extend(index)

        test_path = out_path + "/test_result"
        if not os.path.exists(test_path):
            os.mkdir(test_path)

        history.radar_chart(input_img=input_list,
                            output=output_list[:30], 
                            score=score_list[:30], 
                            dataframe=test_list[idx], 
                            index_list=index_list,
                            save_path=test_path,
                            filename='test_radar_chart.png')

        output_list = np.array(output_list)
        score_list  = np.array(score_list)
        index_list  = np.array(index_list)

        o_df = pd.DataFrame(output_list)
        s_df = pd.DataFrame(score_list)
        i_df = pd.DataFrame(test_list[idx]["path"][index_list])

        #out_df   = pd.concat([i_df, o_df],axis=1)
        #out_df = out_df.dropna(how='all')
        #out_df = out_df.reset_index(drop=True)


        #score_df = pd.concat([i_df, s_df],axis=1)
        #score_df = score_df.dropna(how='all')
        #score_df = score_df.reset_index(drop=True)

        o_df.to_csv(out_path + "/test_out_df.csv")
        s_df.to_csv(out_path + "/test_score_df.csv")
        i_df.to_csv(out_path+ "testdata_index.csv")

        corr_list = {}
        for i in range(7):
            corr_list[col[i]] = np.corrcoef(score_list[:,i],output_list[:,i])[0,1]

        # 相関係数
        

        # ~~~~~~ plot graph ~~~~~~~~
        #print("# ~~~~~~ plotting graph ~~~~~~~~")
        plt.rcParams["figure.figsize"] = (6.4, 4.8)
        history.plot_loss("MSE") # 引数：MSE or MAE
        history.plot_loss("MAE")
        history.score_pred_plot(output=output_list, score=score_list)
        history.save()

        # ~~~~~~ save log ~~~~~~~~~ 
        #dt_now = datetime.datetime.now()
        savepath = os.path.join(log_path, 'log.csv')

        if not os.path.exists(savepath):
            with open(savepath, 'w', encoding='utf_8_sig') as f: # 'w' 上書き
                writer = csv.writer(f)
                writer.writerow(["seed",
                            "optim",
                            "batch_size",
                            "in_w",
                            "lr",
                            "weight_decay",
                            "min_L1_loss",
                            "corr_list",
                            "min_MSE_loss",
                            "closs_valid",
                            "corr mean"])
            

        with open(savepath, 'a', encoding='utf_8_sig') as f: # 'a' 追記
            writer = csv.writer(f)
            writer.writerow([seed,
                            optim_flg,
                            batch_size,
                            in_w,
                            lr,
                            weight_decay,
                            min_L1_loss.to('cpu').detach().numpy().copy(),
                            corr_list,
                            min_loss.to('cpu').detach().numpy().copy(),
                            idx,
                            sum(corr_list.values()) / len(corr_list)])
        print(f"completed{idx}_phrase")

    print("~~~~~~~~~~ completed ~~~~~~~~~~~~")
if __name__ == '__main__':
    main()
