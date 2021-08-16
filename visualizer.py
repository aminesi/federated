import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

root = 'results/cifar/'
for folder in tqdm(os.listdir(root)):
    for file in os.listdir(root + folder):
        if file.endswith('npy'):
            x = np.arange(1, 1001)
            y = np.load(root + folder + '/' + file)
            mean = y.cumsum() / x
            metric = file.replace('.npy', '')
            plt.figure()
            plt.plot(x, y, label=metric.replace('_', ' '))
            plt.plot(x, mean, label=metric.replace('_', ' ') + ' mean')
            plt.xlabel('num of round')
            plt.grid(True)
            plt.yticks(np.arange(0, 1, 0.1))
            plt.legend()
            plt.savefig(root + folder + '/' + metric + '.png')
