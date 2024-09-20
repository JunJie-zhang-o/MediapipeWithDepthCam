#!/usr/bin/env python3
# coding=utf-8
'''
Author       : Jay jay.zhangjunjie@outlook.com
Date         : 2024-08-20 11:28:41
LastEditTime : 2024-08-20 17:58:10
LastEditors  : Jay jay.zhangjunjie@outlook.com
Description  : 中心对称的夹爪colorbar
'''
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from xmlrpc.client import ServerProxy




# 禁用工具栏
plt.rcParams['toolbar'] = 'none'

# 创建图和轴
fig, ax = plt.subplots(figsize=(8, 2))  # 调整窗口大小
plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.2)  # 调整布局

# 设置中心对称的最小值、最大值和颜色映射
vmin, vmax = -25, 25
cmap = plt.cm.gist_heat

colors = [
    (0.0, 'yellow'),  # 起始颜色
    (0.5, 'red'), # 中心颜色
    (1.0, 'yellow')    # 结束颜色
]
# 创建对称的色带
cmap = LinearSegmentedColormap.from_list('symmetric_cmap', colors)
# 创建色带
# cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=symmetric_cmap), cax=ax, orientation='horizontal')



# 创建 ScalarMappable
norm = plt.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# 添加水平 colorbar
cbar = fig.colorbar(sm, cax=ax, orientation='horizontal')

# 在 colorbar 上方添加标题
ax.set_title('Dynamic Colorbar', fontsize=14, pad=10)

# 更新函数
def update(value):
    
    v = ServerProxy("http://192.168.1.7:9120/", allow_none=True).position()
    print(v)
    value = (50 - int(v)) / 2
    # 设置 colorbar 的新 clim 范围
    sm.set_clim(vmin=vmin, vmax=vmax)
    
    # 动态调整 cmap 的范围，使 -value 到 value 映射为有颜色，其余为黑色
    new_cmap = cmap(np.linspace(0, 1, 256))
    
    # 计算中心点和索引范围
    center_idx = 128
    value_idx = int((value / vmax) * 127)  # value 相对于 25 的比例索引
    
    # 将颜色映射范围外的部分设置为黑色
    new_cmap[:center_idx-value_idx] = [0, 0, 0, 1]  # -value 以外的部分为黑色
    new_cmap[center_idx+value_idx:] = [0, 0, 0, 1]  # value 以外的部分为黑色

    sm.set_cmap(plt.cm.colors.ListedColormap(new_cmap))

    cbar.update_normal(sm)

    # 更新标题，显示当前 value
    ax.set_title(f'Gripper Position: {int(value)}', fontsize=14, pad=10)

# 创建动画
ani = animation.FuncAnimation(fig, update, frames=np.linspace(25, 0, 100), interval=200, blit=False)

plt.show()
