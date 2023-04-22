import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文
# # plt.plot([1,2,3], label = 'Whatever')
# plt.gcf().subplots_adjust(left=0.25,top=0.90,bottom=0.25)
# plt.xlabel('xaxis', fontsize = 16)
# plt.ylabel('yaxis', fontsize = 20)
# # plt.legend(fontsize = 18)
# # plt.subplots_adjust(left=0.5, bottom=0.8, right=0.8, top=0.9)
# plt.xticks(fontsize = 25)
# plt.yticks(fontsize = 25)
# # plt.title('PLOT', fontsize = 20)
# x_axis_data = [10, 20, 30, 40, 50]
# # y_axis_data = [92, 95,   122 , 93 , 87]
#
# # #top-1
# # y_axis_data = [92, 95,   122 , 93 , 87]
# # #top-3
# # y_axis_data = [175, 176,  192 , 172 , 176]
# # #top-5
# # y_axis_data = [220, 223,   228 , 218 , 224]
# # #MAR
# # y_axis_data = [11.50, 11.26 ,   11.17  , 11.23 , 11.25]
# # #MFR
# # y_axis_data = [8.73, 9.11,   7.99 , 9.02 , 8.97]
# # #P
# y_axis_data = [23.54, 23.54,   30.89 , 23.54 , 23.54]
# # plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
# plt.plot(x_axis_data, y_axis_data, 'ro-', color='#4169E1', alpha=0.7, linewidth=1, label = "SupConFL")
# # 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
# plt.legend(loc="upper right")
# plt.xlabel('epochs',fontsize = 30)
# plt.ylabel('p%',fontsize = 30)
# plt.legend(fontsize=16)
# plt.show()
# [5.9929e-05],
#          [1.4850e-03],
#          [1.8522e-02],
#          [6.3850e-02],
#          [1.2426e-01],
#          [1.3408e-01],
#          [1.1314e-01],
#          [8.2888e-02],
#          [6.1227e-02],
#          [4.9413e-02],
#          [4.3125e-02],
#          [3.9301e-02],
#          [3.6742e-02],
#          [3.5049e-02],
#          [3.4006e-02],
#          [3.3357e-02],
#          [3.2904e-02],
#          [3.2534e-02],
#          [3.2200e-02],
#          [3.1852e-02]]], grad_fn=<SoftmaxBackward>)

# ([[[0.0508],
#          [0.0466],
#          [0.0471],
#          [0.0474],
#          [0.0475],
#          [0.0475],
#          [0.0475],
#          [0.0475],
#          [0.0475],
#          [0.0475],
#          [0.0475],
#          [0.0475],
#          [0.0475],
#          [0.0475],
#          [0.0475],
#          [0.0475],
# fruit_dict = {
#     1.5968e-06: 0.0508,
#     1.4850e-03: 0.0466,
#     1.8522e-02:0.0471,
#     6.3850e-02:0.0474,
#     1.2426e-01:0.0475,
#     1.3408e-01: 0.0475,
#     1.1314e-01:0.0475,
#     8.2888e-02:0.0475,
#     6.1227e-02:0.0475,
#     4.9413e-02: 0.0475
# }
# fruit_dict = {
#     "controlled Attention":[ 1.5968e-06,
#     1.4850e-03,
#     1.8522e-02,
#     6.3850e-02,
#     1.2426e-01,
#     1.3408e-01,
#     1.1314e-01,
#     8.2888e-02,
#     6.1227e-02,
#     4.9413e-02],
# 'uncontrolled Attention':[
#  0.0508,
#  0.0466,
# 0.0471,
# 0.0474,
# 0.0475,
#  0.0475,
# 0.0475,
# 0.0475,
# 0.0475,
#  0.0475
#
# ],
# "row1":[1,2,3,4,5,6,7,8,9,10],
# # "cul":['controlled Attention', 'uncontrolled Attention']
# }

# import csv
#
# person = [
# (6.3850e-02,0.0474),
# (1.2426e-01,0.0478),
# (1.3408e-01, 0.0508),
# (1.1314e-01,0.0475),
# (8.2888e-02,0.0475),
# (6.1227e-02,0.0466),
# ]
# # 表头
# header = ['CA', 'UCA']
#
# with open('person1.csv', 'w', encoding='utf-8') as file_obj:
#     # 1:创建writer对象
#     writer = csv.writer(file_obj)
#     # 2:写表头
#     writer.writerow(header)
#     # 3:遍历列表，将每一行的数据写入csv
#     for p in person:
#         writer.writerow(p)
#
# file_obj.close()
#
#
#
#
#
#
# sns.set()
# plt.rcParams['font.sans-serif']='SimHei'#设置中文显示，必须放在sns.set之后
# np.random.seed(0)
# # uniform_data = pd.DataFrame(list(fruit_dict.items()),
# #                    columns=['controlled Attention', 'uncontrolled Attention'])
#
# uniform_data = pd.read_csv("person1.csv")
# uniform_data.index = [1,2,3,4,5,6]
# uniform_data.columns = ['CA', 'UCA']
# # uniform_data = pd.DataFrame(fruit_dict)
# # uniform_data = uniform_data
# # result = uniform_data.pivot(index="row1", columns=1, values=['controlled Attention', 'uncontrolled Attention'])
# # uniform_data = np.random.rand(10, 2) #设置二维矩阵
# f, ax = plt.subplots(figsize=(5, 6))
#
# #heatmap后第一个参数是显示值,vmin和vmax可设置右侧刻度条的范围,
# #参数annot=True表示在对应模块中注释值
# # 参数linewidths是控制网格间间隔
# #参数cbar是否显示右侧颜色条，默认显示，设置为None时不显示
# #参数cmap可调控热图颜色，具体颜色种类参考：https://blog.csdn.net/ztf312/article/details/102474190
# h = sns.heatmap(uniform_data , ax=ax,vmin=0,vmax=0.2,cmap='YlOrRd',annot=True,annot_kws={'size':25},linewidths=2,cbar=False)
#
# # ax.set_title('hello') #plt.title('热图'),均可设置图片标题
# # ax.set_ylabel('y_label')  #设置纵轴标签
# # ax.set_xlabel('x_label')  #设置横轴标签
# cb=h.figure.colorbar(h.collections[0]) #显示colorbar
#
# cb.ax.tick_params(labelsize=25) #设置colorbar刻度字体大小。
# #设置坐标字体方向，通过rotation参数可以调节旋转角度
# label_y = ax.get_yticklabels()
# plt.setp(label_y, rotation=360, horizontalalignment='center',fontsize = 25)
# label_x = ax.get_xticklabels()
# plt.setp(label_x, rotation=360, horizontalalignment='center',fontsize = 25)
#
# plt.show()
#导入库
# encoding=utf-8
import matplotlib.pyplot as plt
# from pylab import *         #支持中文
# mpl.rcParams['font.sans-serif'] = ['SimHei']
# def to_percent(temp, position):
#   return '%1.0f'%(1*temp) + '%'
#
# names = ['0%','10%', '20%', '30%', '40%', '50%', '60%','70%','80%','90%','100%']
# x = range(len(names))
#
# y = [0,15.75,31.50, 45.94, 57.74, 74.28, 82.41,87.66,93.43,98.16,100]
# # y1=[0,17, 63, 71, 105, 159,187,201,361,369,395]
# # y2=[0,33,79,133,197,291,323,363,387,390,395]
#
# y33=[0,32, 77, 121, 183, 256,310,345,382,389,395]
# y44=[0,27, 70, 117, 149, 207,282,309,366,373,395]
# y55=[0,25, 67, 98, 135, 197,249,280,360,372,395]
# y66=[0,19, 66, 104, 118, 164,194,232,365,370,395]
# y3 = list()
# y4 = list()
# y5 = list()
# y6 = list()
#
#
#
# for i in range(len(y33)):
#     y3.append(round(float(y33[i]/3.95), 2) )
#     y4.append(round(float(y44[i] / 3.95), 2))
#     y5.append(round(float(y55[i]/3.95), 2) )
#     y6.append(round(float(y66[i] / 3.95), 2))
#
# # print(y3)
# # print(y4)
# # print(y5)
# # print(y6)
#
#
#
# y1=[0,4.30, 15.94, 17.97, 26.58, 40.25,47.34,50.88,91.39,93.41,100]
# y2=[0,8.35,20,33.67,49.87,73.67,83.92,91.90,97.97,98.73,100]
# # y3=[0,, 15.94, 17.97, 26.58, 40.25,47.34,50.88,91.39,93.41,100]
# # y4=[0,4.30, 15.94, 17.97, 26.58, 40.25,47.34,50.88,91.39,93.41,100]
# # y5=[0,4.30, 15.94, 17.97, 26.58, 40.25,47.34,50.88,91.39,93.41,100]
# # y6=[0,4.30, 15.94, 17.97, 26.58, 40.25,47.34,50.88,91.39,93.41,100]
#
#
#
# #plt.plot(x, y, 'ro-')
# #plt.plot(x, y1, 'bo-')
# #pl.xlim(-1, 11) # 限定横轴的范围
# #pl.ylim(-1, 110) # 限定纵轴的范围
# plt.plot(x, y, marker='o', mec='r', mfc='w',label=u'y=SupConFL')
# plt.plot(x, y1, marker='o', ms=7,label=u'y=Ochiai')
# plt.plot(x, y2, marker='o', ms=7,label=u'y=DEEPRL4FL')
# plt.plot(x, y3, marker='o', ms=7,label=u'y=DeepFL')
# plt.plot(x, y4, marker='o', ms=7,label=u'y=MUSE')
# plt.plot(x, y5, marker='o', ms=7,label=u'y=Metallaxis')
# plt.plot(x, y6, marker='o', ms=7,label=u'y=Dstar')
#
# plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
#
# plt.legend() # 让图例生效
# plt.xticks(x, names, rotation=0)
# plt.margins(0)
# plt.subplots_adjust(bottom=0.15)
# plt.xlabel(u"% of executable statements that need to be examined (i.e., EXAM score)",fontsize = 12) #X轴标签
# plt.ylabel("%of faulty version",fontsize = 12) #Y轴标签
#
#
# plt.show()

#导入需要的包
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os  # 导入os库
def to_percent(temp, position):
  return '%1.0f'%(1*temp) + '%'
x = ['Ochiai','Dstar','MUSE','Metallaxis','DeepFL','DEEPRL4FL']  # 产生0-10之间30个元素的等差数列

y1 = [14.71,16.41,21.99,23.10,33.17,39.32]
plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置字体为SimHei显示中文\n",
plt.rc('font', size=14)  # 设置图中字号大小\n",
plt.figure(figsize=(6, 4))  # 设置画布\n",
plt.bar(x, y1, width=0.4)  # 绘制柱状图\n",
# for a,b in zip(x,y1):
#
#     plt.text(a,b,b)
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))

plt.show()