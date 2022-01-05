from matplotlib import pyplot as plt
import numpy as np
import os
import argparse
from scipy.integrate import simps


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Validate Score")
    parser.add_argument(
        "score_file",
        type=str,
        default=None,
        help="Path to npy score file",
    )
    args = parser.parse_args()
    assert os.path.isfile(args.score_file), f"Cannot find file: {args.score_file}"
    allscores = np.load(args.score_file)

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