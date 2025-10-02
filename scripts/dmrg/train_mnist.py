# -*- coding: utf-8 -*-
import argparse
from sys import path, argv
import os

# Add the parent directory to Python path
path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mps.mps_cumulant import MPS_c
import os
import re
import numpy as np

np.set_printoptions(5, linewidth=4 * 28)
from time import strftime
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import wandb


def sample_image(mps, typ):
    dat = mps.generate_sample()
    a = int(np.sqrt(dat.size))
    img = dat.reshape((a, a))
    if typ == "s":
        for n in range(1, a, 2):
            img[n, :] = img[n, ::-1]
    return img


def sample_plot(mps, typ, nn):
    ncol = int(np.sqrt(nn))
    while nn % ncol != 0:
        ncol -= 1
    fig, axs = plt.subplots(nn // ncol, ncol)
    for ax in axs.flatten():
        ax.matshow(sample_image(mps, typ), cmap=mpl.cm.gray_r)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig("samples.pdf")


def loss_plot(mps, spars):
    fig, ax = plt.subplots()
    nsteps = 2 * mps.space_size - 4
    if spars:
        ax.plot(np.arange(len(mps.Loss)) * (nsteps // 2), mps.Loss, ".")
    else:
        ax.plot(mps.Loss)
    ax.xaxis.set_major_locator(MultipleLocator(nsteps))
    ax.xaxis.set_minor_locator(MultipleLocator(nsteps // 2))
    ax.xaxis.grid(which="both")
    ax.set_xticks([])
    plt.savefig("Loss.pdf")


def find_latest_MPS(ckpt_dir="checkpoints/dmrg"):
    """Searching for the last MPS in the most recent timestamped directory"""
    # Find the most recent MNIST-contin directory
    name_list = os.listdir(ckpt_dir)
    timestamp_dirs = [d for d in name_list if d.startswith("MNIST-contin_on_")]

    if not timestamp_dirs:
        print("No MPS Found")
        return None

    # Get the most recent directory (they're sorted by timestamp)
    latest_dir = sorted(timestamp_dirs)[-1]
    print(f"Looking in directory: {latest_dir}")

    # Look for MPS files in that directory
    # mps_files = os.listdir(f"./{latest_dir}")
    mps_files = os.listdir(os.path.join(ckpt_dir, latest_dir))
    pmx = -1
    mx = 0
    pattrn = r"Loop(\d+)MPS"
    for prun in range(len(mps_files)):
        mach = re.match(pattrn, mps_files[prun])
        if mach is not None:
            nl = int(mach.group(1))
            if nl >= mx:
                pmx = prun
                mx = nl

    if pmx == -1:
        print("No MPS Found")
        return None
    else:
        return mx, f"{latest_dir}/{mps_files[pmx]}", latest_dir


def start(ckpt_dir="checkpoints/dmrg"):
    """Start the training, in a relatively high cutoff, over usually just 1 epoch"""
    dtset = np.load(train_dataset_name)

    filepath = os.path.join(ckpt_dir, strftime("MNIST-contin_on_%B_%d_%H%M"))
    os.makedirs(filepath, exist_ok=True)
    os.chdir(filepath)
    f = open("DATA_" + train_dataset_name.split("/")[-1] + ".txt", "w")
    f.write("../" + train_dataset_name)
    f.close()

    m.verbose = 0
    m.left_cano()
    m.designate_data(dtset)
    m.init_cumulants()
    # m.verbose = 1
    m.nbatch = 10
    m.descenting_step_length = 0.05
    m.descent_steps = 10
    m.cutoff = 0.3

    print(m.Show_Loss())
    nlp = 1
    cut_rec = m.train(nlp, True)
    m.cutoff = cut_rec
    m.saveMPS("Loop%d" % (nlp - 1), True)


def onecutrain(
    lr_shrink, loopmax, safe_thres=0.5, lr_inf=1e-10, ckpt_dir="checkpoints/dmrg"
):
    """Continue the training, in a fixed cutoff, train until loopmax is finished"""
    dtset = np.load(train_dataset_name)
    m.designate_data(dtset)

    # wandb init
    wandb.init(
        project="ptn-dmrg",
        name=f"mnist-lr_shrink_{lr_shrink}-loopmax_{loopmax}-safe_thres_{safe_thres}-lr_inf_{lr_inf}",
    )

    result = find_latest_MPS()
    if result is None:
        print("No MPS found to resume from. Please run 'start' first.")
        return
    mx, folder, latest_dir = result
    print("Resuming: ", folder)

    loop_last = mx
    nlp = 5
    m.verbose = 0
    m.loadMPS(os.path.join(ckpt_dir, folder))
    # m.descent_steps = 10
    m.init_cumulants()
    # m.verbose = 1
    # m.cutoff = 1e-7

    """Set the hyperparameters here"""
    m.maxibond = 800
    m.nbatch = 20
    m.descent_steps = 10
    m.descenting_step_length = 0.001

    lr = m.descenting_step_length
    while loop_last < loopmax:
        if m.minibond > 1 and m.bond_dimension.mean() > 10:
            m.minibond = 1
            print("From now bondDmin=1")

        # train tentatively
        loss_last = m.Loss[-1]
        while True:
            try:
                print(f"Training loop {loop_last} with nlp={nlp}")
                m.train(nlp, False)
                if m.Loss[-1] - loss_last > safe_thres:
                    print("lr=%1.3e is too large to continue safely" % lr)
                    raise Exception("lr=%1.3e is too large to continue safely" % lr)

                # Compute test loss
                print("Computing test loss...")
                test_loss = m.Calc_Loss(np.load(test_dataset_name))
                print(f"Test loss: {test_loss}")
                wandb.log(
                    {
                        "train/loop": loop_last,
                        "train/loss_last": float(m.Loss[-1]),
                        "train/lr": float(lr),
                        "eval/test_loss": float(test_loss),
                        "mps/bond_mean": (
                            float(m.bond_dimension.mean())
                            if hasattr(m, "bond_dimension")
                            else None
                        ),
                        "mps/cutoff": float(m.cutoff) if hasattr(m, "cutoff") else None,
                    }
                )
            except:
                lr *= lr_shrink
                if lr < lr_inf:
                    print("lr becomes negligible.")
                    wandb.finish()
                    return
                m.loadMPS("Loop%dMPS" % loop_last)
                m.designate_data(dtset)
                m.init_cumulants()
                m.descenting_step_length = lr
            else:
                break

        loop_last += nlp
        m.saveMPS(os.path.join(ckpt_dir, latest_dir, "Loop%d" % loop_last), True)
        print("Loop%d Saved" % loop_last)

    wandb.finish()


if __name__ == "__main__":
    # train_dataset_name = "data/mnist-rand1k_28_thr50_z/_data.npy"
    train_dataset_name = "data/mnist/test.npy"
    test_dataset_name = "data/mnist/test.npy"

    m = MPS_c(28 * 28)

    if argv[1] == "start":
        start()
    elif argv[1] == "one":
        onecutrain(0.9, 250, 0.05)
    elif argv[1] == "plot":
        m.loadMPS("./Loop%dMPS" % int(argv[2]))

    # loss_plot(m, True)
    np.random.seed(1996)
    sample_plot(m, "z", 20)
