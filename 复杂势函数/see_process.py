import numpy as np
import torch
import matplotlib.pyplot as plt

La=-10
Lb=10
N=3000
en_num=50


file_name=f'./fun_images/V_{La}_{Lb}_{N}_{en_num}/V_diag_{i}.pth'

from IPython import display
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline

x = list(np.linspace(0,6,50))
y = np.sin(x)
for i in range(len(x)):
    plt.cla()  # 清除图片 
    plt.xlim((0, 6))
    plt.ylim((-1.1, 1.1))
    xx = x[:i]
    yy = y[:i]
    plt.plot(xx, yy, color='black')  # 绘制曲线
    plt.scatter(x[i-1], y[i-1], color='red', s=20)  # 绘制最新点
    plt.title('sin(x)')
    display.clear_output(wait=True)
    plt.pause(0.00000001) 
plt.show()
