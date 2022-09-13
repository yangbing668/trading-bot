import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from shutil import rmtree
from trading_bot.utils import get_stock
import os
import datetime

def mkdir(directory):
    if os.path.exists(directory):
        rmtree(directory)
    os.makedirs(directory)

def walkFile(root_path):
    fileList = []
    for root, dirs, files in os.walk(root_path):
        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list
        # 遍历文件
        for f in files:
            fileList.append(f)
    return fileList

def plot(path, stocks):
    for stock in stocks:
        data = get_stock(path + stock)

        data_train = data[data.index < '2019-12-11']
        data_test = data[data.index >= '2019-12-11']

        dataName = os.path.splitext(stock)[0]

        sns.set(rc={'figure.figsize': (9, 5)})
        #plt.figure(dpi=300, figsize=(9, 5))
        df1 = pd.Series(data_train.Close, index=data.index)
        df2 = pd.Series(data_test.Close, index=data.index)

        ax = data.Close.plot(label='')
        df1.plot(ax=ax, color='b', label='train')
        df2.plot(ax=ax, color='r', label='test')
        ax.set(xlabel='Date', ylabel='Close Price')
        ax.set_title(f'Train and Test sections of dataset {dataName}')

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.legend()
        plt.savefig(f'../downloadData/Data/img/{dataName}.jpg', dpi=1200)
        # plt.show()
        plt.close()

mkdir('../downloadData/Data/img')
path = '../downloadData/Data/all/'
stocks = walkFile(path)
plot(path, stocks)

