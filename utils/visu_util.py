# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018
# revised by Hyeontae Son
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_pcd_three_views(filename, pcds, titles, is_from_decoder=None, suptitle='', sizes=None, cmap='Reds', zdir='y',
                         xlim=(-0.3, 0.3), ylim=(-0.3, 0.3), zlim=(-0.3, 0.3)):
    if sizes is None:
        sizes = [0.5 for i in range(len(pcds))]

    if is_from_decoder is None:
        fig = plt.figure(figsize=(len(pcds) * 3, 9))

        for i in range(3):
            elev = 30
            azim = -45 + 90 * i
            for j, (pcd, size) in enumerate(zip(pcds, sizes)):
                color = pcd[:, 0]
                ax = fig.add_subplot(3, len(pcds), i * len(pcds) + j + 1, projection='3d')
                ax.view_init(elev, azim)
                ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=size, cmap=cmap, vmin=-1, vmax=0.5)
                ax.set_title(titles[j])
                ax.set_axis_off()
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_zlim(zlim)
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
        plt.suptitle(suptitle)
        fig.savefig(filename)
        plt.close(fig)

    else:
        fig = plt.figure(figsize=((len(pcds) + 2) * 3, 9)) # add two columns(from upsampling module, from decoder)
        is_from_decoder = np.array(is_from_decoder)
        for i in range(3):
            elev = 30
            azim = -45 + 90 * i
            for j, (pcd, size) in enumerate(zip(pcds, sizes)):
                color = pcd[:, 0]
                if j == 1: # output
                    # output as it is
                    ax = fig.add_subplot(3, len(pcds) + 2, i * (len(pcds)+2) + j + 1, projection='3d')
                    ax.view_init(elev, azim)
                    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=size, cmap=None, vmin=-1, vmax=0.5)
                    ax.set_title(titles[j])
                    ax.set_axis_off()
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    ax.set_zlim(zlim)

                    # from upsampling module. colored blue
                    pcd_up = pcd[~is_from_decoder]
                    pcd_up_color = pcd_up[:, 0]
                    ax = fig.add_subplot(3, len(pcds) + 2, i * (len(pcds)+2) + j + 2, projection='3d')
                    ax.view_init(elev, azim)
                    ax.scatter(pcd_up[:, 0], pcd_up[:, 1], pcd_up[:, 2], zdir=zdir, c=pcd_up_color, s=size, cmap='Blues', vmin=-1, vmax=0.5)
                    ax.set_title(titles[j + 1])
                    ax.set_axis_off()
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    ax.set_zlim(zlim)

                    # from decoder. colored red
                    pcd_dec = pcd[is_from_decoder]
                    pcd_dec_color = pcd_dec[:, 0]
                    ax = fig.add_subplot(3, len(pcds) + 2, i * (len(pcds)+2) + j + 3, projection='3d')
                    ax.view_init(elev, azim)
                    ax.scatter(pcd_dec[:, 0], pcd_dec[:, 1], pcd_dec[:, 2], zdir=zdir, c=pcd_dec_color, s=size, cmap='Reds', vmin=-1, vmax=0.5)
                    ax.set_title(titles[j + 2])

                elif j == 2:
                    ax = fig.add_subplot(3, len(pcds) + 2, i * (len(pcds)+2) + j + 3, projection='3d')
                    ax.view_init(elev, azim)
                    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=size, cmap=None, vmin=-1, vmax=0.5)
                    ax.set_title(titles[j+2])

                else: # j==0
                    ax = fig.add_subplot(3, len(pcds)+2, i * (len(pcds)+2) + j + 1, projection='3d')
                    ax.view_init(elev, azim)
                    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=size, cmap=None, vmin=-1, vmax=0.5)
                    ax.set_title(titles[j])
                ax.set_axis_off()
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_zlim(zlim)

        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
        plt.suptitle(suptitle)
        fig.savefig(filename)
        plt.close(fig)

def plot_pcd_nn_dist(filename, pcds, nn_dists, titles, suptitle='', sizes=None, cmap='rainbow', zdir='y',
                         xlim=(-0.3, 0.3), ylim=(-0.3, 0.3), zlim=(-0.3, 0.3)):
    if sizes is None:
        sizes = [0.5 for i in range(len(pcds))]

    fig = plt.figure(figsize=(len(pcds) * 3, 9))

    for i in range(3):
        elev = 30
        azim = -45 + 90 * i
        for j, (pcd, size) in enumerate(zip(pcds, sizes)):
            color = nn_dists[j]
            ax = fig.add_subplot(3, len(pcds), i * len(pcds) + j + 1, projection='3d')
            ax.view_init(elev, azim)
            nn_plt = ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=size, cmap=cmap, vmin=0.0, vmax=0.04)

            ax.set_axis_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
            ax.set_title(titles[j])
            plt.colorbar(nn_plt, ax=ax)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)