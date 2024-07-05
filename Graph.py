import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import numpy as np
import numpy as np
import pandas as pd

linewidth=0.6
plt.rcParams['axes.linewidth'] = linewidth
# # plt.rcParams["font.family"] = fm.arial
# # # temporal locality

cache_size = ['1MB', '2MB', '4MB', '8MB', '1MB', '2MB', '4MB', '8MB']
totals_random = [2, 4, 8, 16, 0, 0, 0, 0]
Q_col8 = [0, 0, 0, 0, 33.2, 36.6, 40.1, 41.7]
R_col8 = [0, 0, 0, 0, 46.1, 47, 49.7, 49.7]
totals_col8 = [0, 0, 0, 0, 79.3, 83.7, 89.8, 91.5]

n_groups = len(totals_random)
fig, ax = plt.subplots()

index = np.array([0, 1.2, 2.4, 3.6, 4.75+0.6+0.2, 6.45+0.6+0.1+0.2, 8.15+0.6+0.1+0.4, 9.85+0.6+0.1+0.6])
bar_width = 0.45

ax.bar(index, totals_random, bar_width, label='Total hit', edgecolor='black', color='#707070', linewidth=linewidth)
ax.bar(index - bar_width, Q_col8, bar_width, label='Q table hit', edgecolor='black', color='#1a80bb', linewidth=linewidth)
ax.bar(index, R_col8, bar_width, label='R table hit', edgecolor='black', color='#f2c45f', linewidth=linewidth)
ax.bar(index + bar_width, totals_col8, bar_width, edgecolor='black', color='#707070', linewidth=linewidth)

ax.set_ylabel('Cache Hit Rate (%)', fontsize=14)
ax.xaxis.set_tick_params(width=0.55)
ax.set_xticks(index)
ax.set_xticklabels(cache_size)
# ax.xaxis.labelpad = 20
ax.xaxis.set_label_coords(0, -2)

trans = fig.transFigure
ax.plot([0.14,0.14],[0.206,0.03], color="k", transform=trans, clip_on=False, linewidth=linewidth)
ax.plot([0.465,0.465],[0.206,0.03], color="k", transform=trans, clip_on=False, linewidth=linewidth)
ax.plot([0.968, 0.968],[0.206,0.03], color="k", transform=trans, clip_on=False, linewidth=linewidth)

plt.text(1.8, -14, 'Random', ha='center', va='top', fontsize=12)
plt.text(8.1, -14, 'Criteo Dataset (col 8)', ha='center', va='top', fontsize=12)

handles, labels = plt.gca().get_legend_handles_labels()
order = [1,2,0]
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], bbox_to_anchor=(0.02, 1.12), frameon=False, loc='upper left', ncol=3, handlelength=0.7, columnspacing=0.8)

plt.xlim(-0.7, 12.35)
plt.show()
plt.gcf().set_size_inches(4.75,3.45)
plt.tight_layout()


plt.savefig('./temporal_locality.jpg', dpi=1000)



# # spatial locality

cache_size = ['64B', '128B', '256B', '512B']
spatial_Q_hits = [34.8, 32.2, 29.4, 27.6]
spatial_R_hits = [47.7, 48.8, 45.6, 47.3]
spatial_total_hits = [82.5, 81, 75, 74.9]

n_groups = len(spatial_Q_hits)
fig, ax = plt.subplots()
index = np.arange(n_groups)
index = np.array([0, 0.45, 0.9, 1.35])
bar_width = 0.09

ax.bar(index - bar_width, spatial_Q_hits, bar_width, label='Q table hit', edgecolor='black', color='#1a80bb', linewidth=0.55)
ax.bar(index, spatial_R_hits, bar_width, label='R table hit', edgecolor='black', color='#f2c45f', linewidth=0.55)
ax.bar(index + bar_width, spatial_total_hits, bar_width, label='Total hit', edgecolor='black', color='#707070', linewidth=0.55)

ax.set_ylabel('Cache Hit Rate (%)', fontsize=14)
ax.set_xlabel('Cache Block Size', fontsize=12)
ax.xaxis.set_label_coords(0.5, -0.14)
ax.set_xticks(index)
ax.set_xticklabels(cache_size)

plt.legend(bbox_to_anchor=(0.01, 1.12), frameon=False, loc='upper left', ncol=3, handlelength=0.7, columnspacing=0.8)
plt.xlim(-0.25, 1.6)
plt.show()
plt.gcf().set_size_inches(3.75,3.5)
plt.tight_layout()
 
plt.savefig('./spatial_locality_graph.jpg', dpi=1000)