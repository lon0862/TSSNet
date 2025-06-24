import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir_file', type=str, default='/save_ckpt/ckpt_2/info_v29')
    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path  = os.path.join(dir_path, args.dir_file)
    all_file_name = os.listdir(dir_path)
    num = 0
    epochs = []
    train_loss = []
    val_loss = []
    train_cls_loss = []
    val_cls_loss = []
    train_min_fde = []
    val_min_fde = []
    train_min_ade = []
    val_min_ade = []

    if(len(all_file_name)>0):
        for i in range(len(all_file_name)):
            file_path =  dir_path + "/info_"+str(i)+".txt"
            epochs.append(i)
            with open(file_path, 'r') as f:
                for line in f:
                    key, value = line.strip().split(':')
                    if key == 'train_loss':
                        train_loss.append(float(value))
                    elif key == 'val_loss':
                        val_loss.append(float(value))
                    elif key == 'train_cls_loss':
                        train_cls_loss.append(float(value))
                    elif key == 'val_cls_loss':
                        val_cls_loss.append(float(value))
                    elif key == 'train_minFDE':
                        train_min_fde.append(float(value))
                    elif key == 'val_minFDE':
                        val_min_fde.append(float(value))
                    elif key == 'train_minADE':
                        train_min_ade.append(float(value))
                    elif key == 'val_minADE':
                        val_min_ade.append(float(value))

        train_reg_loss = np.array(train_loss) - np.array(train_cls_loss)
        val_reg_loss = np.array(val_loss) - np.array(val_cls_loss)

        for i in range(len(train_loss)):
            print("[epoch {}]: train_loss: {}, val_loss: {}".format(i+1, train_loss[i], val_loss[i]))

        for i in range(len(train_min_fde)):
            print("[epoch {}]: train_min_fde: {}, val_min_fde: {}".format(i+1, train_min_fde[i], val_min_fde[i]))
        
        for i in range(len(train_min_ade)):
            print("[epoch {}]: train_min_ade: {}, val_min_ade: {}".format(i+1, train_min_ade[i], val_min_ade[i]))

        size = int(len(train_loss)*0.1)
        plt.plot(epochs[size:], train_loss[size:], color='red', label='train_loss')
        plt.plot(epochs[size:], val_loss[size:], color='blue', label='val_loss')
        plt.legend()
        plt.xlabel('epoch') # 設定 x 軸標題
        plt.ylabel('loss') # 設定 y 軸標題
        plt.show()

        plt.plot(epochs[size:], train_min_fde[size:], color='red', label='train_min_fde')
        plt.plot(epochs[size:], val_min_fde[size:], color='blue', label='val_min_fde')
        plt.legend()
        plt.xlabel('epoch') # 設定 x 軸標題
        plt.ylabel('fde') # 設定 y 軸標題
        plt.show()

        plt.plot(epochs[size:], train_min_ade[size:], color='red', label='train_min_ade')
        plt.plot(epochs[size:], val_min_ade[size:], color='blue', label='val_min_ade')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('ade')
        plt.show()

