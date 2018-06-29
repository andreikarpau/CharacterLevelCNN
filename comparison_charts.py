import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


#-----------RMSE----------
fig, ax = plt.subplots()

ind = np.arange(4)
ax.bar(ind, [0.865, 0.859, 0.945, 0.916])
ax.set_title("RMSE")
ax.set_xticks(ind)
ax.set_xticklabels(['Standard', 'Standard Group', 'ASCII', 'ASCII Group'])
ax.set_ylim([0, 1.5])

fig.savefig("./output/plots/compare/rmse.png")
plt.show()

#-----------Time----------
fig, ax = plt.subplots()

ind = np.arange(4)
bar_width = 0.34
plt.bar(ind, [21.22, 22.15, 4.88, 5.37], bar_width,
                 color='b',
                 label='Training')

plt.bar(ind + bar_width, [4, 4.45, 0.83, 0.92], bar_width,
                 color='r',
                 label='Testing')

blue_patch = mpatches.Patch(color='b', label='Training time per epoch')
red_patch = mpatches.Patch(color='r', label='Testing time')
plt.legend(handles=[blue_patch, red_patch])

ax.set_title("Testing and training time per epoch")
ax.set_ylabel("Minutes")
ax.set_xticks([0.17, 1.17, 2.17, 3.17])
ax.set_xticklabels(['Standard', 'Standard Group', 'ASCII', 'ASCII Group'])
ax.set_ylim([0, 25])

fig.savefig("./output/plots/compare/time.png")
plt.show()


#-----------AUC Accuracy----------
fig, ax = plt.subplots()

ind = np.arange(4)
bar_width = 0.16
plt.bar(ind, [0.909, 0.910, 0.880, 0.893], bar_width,
                 color='b',
                 label='Accuracy polarity')

plt.bar(ind + bar_width, [0.969, 0.970, 0.952, 0.959], bar_width,
                 color='r',
                 label='AUC polarity')

rects3 = plt.bar(ind + 2 * bar_width, [0.482, 0.488, 0.446, 0.460], bar_width,
                 color='g',
                 label='Accuracy muti-class')

blue_patch = mpatches.Patch(color='b', label='Accuracy polarity')
red_patch = mpatches.Patch(color='r', label='AUC polarity')
green_patch = mpatches.Patch(color='g', label='Accuracy multiclass')
plt.legend(handles=[blue_patch, red_patch, green_patch])

ax.set_title("Accuracy and AUC")
ax.set_xticks([0.24, 1.24, 2.24, 3.24])
ax.set_xticklabels(['Standard', 'Standard Group', 'ASCII', 'ASCII Group'])
ax.set_ylim([0, 1.3])

fig.savefig("./output/plots/compare/AccuracyAUC.png")
plt.show()