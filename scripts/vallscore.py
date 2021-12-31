from matplotlib import pyplot as plt
import numpy as np
import os
from scipy.integrate import simps


if __name__=="__main__":
    allscores = np.load("data/results/kshap/pixel_flip_kshap_2000s_100i_colab/allscore.npy")

    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ratios = np.linspace(0,1, allscores.shape[1])
    y = allscores.sum(axis=0)
    ax.plot(ratios, y)
    ax.set_xlabel("dropped ratio")
    ax.set_ylabel(f"sum of scores")
    ax.grid()

    dx = 1.0/(allscores.shape[1]-1)
    auc = np.trapz(y, dx=dx)
    print(f"auc(trapz) = {auc}")

    auc = simps(y, dx=dx)
    print(f"auc(simps) = {auc}")


    plt.show()
    # pixel_flip_dir = f"data/results/kshap/pixel_flip_{filename}"
    # os.makedirs(pixel_flip_dir, exist_ok=True)

    # scorefile = os.path.join(pixel_flip_dir, f"allscore.npy")
    # np.save(scorefile, allscores)

    # figfile = os.path.join(pixel_flip_dir, f"score_plot.png")
    # fig.savefig(figfile, dpi=300)