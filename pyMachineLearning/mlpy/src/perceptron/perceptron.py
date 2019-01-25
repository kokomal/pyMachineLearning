# coding:utf-8
# Filename: train2.2.py
# Author锛歨ankcs
# Date: 2015/1/31 15:15
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
 
training_set = np.array([[[3, 3], 1], [[4, 3], 1], [[1, 1], -1], [[5, 2], -1]])  # 训练样本
 
a = np.zeros(len(training_set), np.float)  # alpha为训练集的数据采样数N*1的矩阵，为float
print(a)
b = 0.0  # b的初始化为0.0
Gram = None  # Gram矩阵
y = np.array(training_set[:, 1])  # y=[1 1 -1 -1]
x = np.empty((len(training_set), 2), np.float)  # x为N*2的矩阵
for i in range(len(training_set)):  # x=[[3., 3.], [4., 3.], [1., 1.], [5., 2.]]
    x[i] = training_set[i][0]
print('*' * 80)
history = []  # history记录每次迭代结果


def cal_gram():
    N = len(training_set)
    g = np.empty((N, N), np.int)  # Gram矩阵
    for i in range(N):
        for j in range(N):
            g[i][j] = np.dot(training_set[i][0], training_set[j][0])  # G=[xi*xj]
    return g

 
# 随机梯度下降更新
def update(i):
    global a, b
    eta = 0.25
    a[i] += eta  # 根据误分点i更新alpha
    b = b + eta * y[i]  # 更新截距b
    history.append([np.dot(a * y, x), b])  # history记录每次迭代
    print(a, b)

 
# 计算yi(Gram*xi+b),用来判断是否是误分类点
def cal(i):
    global a, b, x, y
    res = np.dot(a * y, Gram[i])
    print('a*y', a*y)
    print('GRAM', Gram[i])
    print('res', res)
    res = (res + b) * y[i]  # 返回是否误分
    return res
 
 
# 检查是否已经正确分类
def check():
    global a, b, x, y
    flag = False
    for i in range(len(training_set)):  # 遍历每个点
        if cal(i) <= 0:  # 如果yi(Gram*xi+b)<=0.则是误分类点
            flag = True
            update(i)  # 用误分类点更新参数
    if not flag:  # 如果已正确分类
        w = np.dot(a * y, x)  # 计算w
        print ("RESULT: w: " , str(w) , " b:", str(b))  # 输出最后结果
        return False
    return True
 
 
if __name__ == "__main__":
    Gram = cal_gram()  # 初始化 Gram矩阵
    for i in range(1000):  # 迭代1000次
        if not check(): break  # 如果已经正确分类则退出
 
    # 以下代码是将迭代过程可视化,数据来源于history
    # first set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
    line, = ax.plot([], [], 'g', lw=2)
    label = ax.text([], [], '')
 
    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        x, y, x_, y_ = [], [], [], []
        for p in training_set:
            if p[1] > 0:
                x.append(p[0][0])
                y.append(p[0][1])
            else:
                x_.append(p[0][0])
                y_.append(p[0][1])
 
        plt.plot(x, y, 'bo', x_, y_, 'rx')
        plt.axis([-6, 6, -6, 6])
        plt.grid(True)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('PerceptronAlgorithm 2 (www.hankcs.com)')
        return line, label
 
    # animation function. this is called sequentially
    def animate(i):
        global history, ax, line, label
 
        w = history[i][0]
        b = history[i][1]
        if w[1] == 0: return line, label
        x1 = -7.0
        y1 = -(b + w[0] * x1) / w[1]
        x2 = 7.0
        y2 = -(b + w[0] * x2) / w[1]
        line.set_data([x1, x2], [y1, y2])
        x1 = 0.0
        y1 = -(b + w[0] * x1) / w[1]
        label.set_text(str(history[i][0]) + ' ' + str(b))
        label.set_position([x1, y1])
        return line, label
 
    # call the animator. blit=true means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(history), interval=1000, repeat=False,
                                   blit=True)
    plt.show()
    anim.save('D:/perceptron2.gif',fps=2, writer='imagemagick') # 似乎需要安装ImageMagick
