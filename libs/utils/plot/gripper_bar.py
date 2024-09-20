#!/usr/bin/env python3
# coding=utf-8
'''
Author       : Jay jay.zhangjunjie@outlook.com
Date         : 2024-08-20 11:28:41
LastEditTime : 2024-08-22 10:00:57
LastEditors  : Jay jay.zhangjunjie@outlook.com
Description  : 进度条样式
'''
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

# 禁用工具栏
plt.rcParams['toolbar'] = 'none'

# 创建图和轴
fig, ax = plt.subplots(figsize=(8, 2))  # 调整窗口大小
plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.3)  # 调整布局

# 设置初始的最小值、最大值和颜色映射
vmin, vmax = 0, 50
cmap = plt.cm.viridis

# 创建 ScalarMappable
norm = plt.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# 添加水平 colorbar
cbar = fig.colorbar(sm, cax=ax, orientation='horizontal')

# 在 colorbar 上方添加标题
ax.set_title('Gripper Position', fontsize=14, pad=10)


# 更新函数
def update(value):
    # 设置 colorbar 的新 clim 范围
    sm.set_clim(vmin=vmin, vmax=vmax)
    
    # 动态调整 cmap 的范围，使 value 到 vmax 映射为黑色
    new_cmap = cmap(np.linspace(0, 1, 256))
    idx = int((value - vmin) / (vmax - vmin) * 255)
    new_cmap[idx:] = [0, 0, 0, 1]  # 将 value 到 vmax 的部分设置为黑色
    sm.set_cmap(plt.cm.colors.ListedColormap(new_cmap))

    cbar.update_normal(sm)
    ax.set_title(f'Gripper Position:{int(value)}', fontsize=14, pad=10)

# 创建动画
ani = animation.FuncAnimation(fig, update, frames=np.linspace(20, 50, 600), interval=10, blit=False)

plt.show()
