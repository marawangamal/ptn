# -*- coding: utf-8 -*-
import argparse
from sys import path, argv
import os
import time

import torch

# Add the parent directory to Python path
path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mps.mps_cumulant_pt import MPS_c
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
    m.cutoff = 1e-7

    """Set the hyperparameters here"""
    m.maxibond = 64
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
                        "train/loss": float(m.Loss[-1]),
                        "train/lr": float(lr),
                        "train/bond_mean": (
                            float(m.bond_dimension.mean())
                            if hasattr(m, "bond_dimension")
                            else None
                        ),
                        "train/cutoff": (
                            float(m.cutoff) if hasattr(m, "cutoff") else None
                        ),
                        "eval/loss": float(test_loss),
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


def train(
    m,
    loopmax,
    lr=0.001,
    lr_shrink=0.9,
    safe_thres=0.5,
    lr_inf=1e-10,
    maxibond=64,
    num_batch=20,
    train_dataset_name="data/mnist/train.npy",
    test_dataset_name="data/mnist/test.npy",
    ckpt_dir="checkpoints/dmrg",
):
    # 0. Setup
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1. Initialize MPS in a relatively high cutoff, over usually just 1 epoch
    dtset = np.load(train_dataset_name)
    m.verbose = 0
    m.left_cano()
    m.designate_data(dtset)
    m.init_cumulants()
    m.nbatch = 10
    m.descenting_step_length = 0.05
    m.descent_steps = 10
    m.cutoff = 0.3

    nlp = 1
    cut_rec = m.train(nlp, True)
    m.cutoff = cut_rec

    # 2. Continue the training, in a fixed cutoff, train until loopmax is finished
    dtset = np.load(train_dataset_name)
    m.designate_data(dtset)

    loop_last = 0
    nlp = 5
    m.verbose = 0
    m.init_cumulants()
    m.cutoff = 1e-7

    m.maxibond = maxibond
    m.nbatch = num_batch
    m.descent_steps = 10
    m.descenting_step_length = lr

    while loop_last < loopmax:
        m.saveMPS(ckpt_dir)
        if (
            m.minibond > 1
            and torch.tensor(m.bond_dimension).to(torch.float32).mean() > 10
        ):
            m.minibond = 1
            print("From now bondDmin=1")

        # train tentatively
        loss_last = m.Loss[-1]
        while True:
            try:
                print(f"Training loop {loop_last} with nlp={nlp}")
                start_time = time.time()
                m.train(nlp, False)
                if m.Loss[-1] - loss_last > safe_thres:
                    print("lr=%1.3e is too large to continue safely" % lr)
                    raise Exception("lr=%1.3e is too large to continue safely" % lr)

                # Compute test loss
                test_loss = m.Calc_Loss(np.load(test_dataset_name))
                print(
                    f"[Loop {loop_last}/{loopmax}] Test loss: {test_loss} | Time: {time.time() - start_time:.2f}s"
                )
                print(f"Bond dimension: {m.bond_dimension}")
                wandb.log(
                    {
                        "train/loss": torch.tensor(m.Loss[-1]).item(),
                        "train/lr": torch.tensor(lr).item(),
                        "train/bond_mean": (
                            torch.tensor(m.bond_dimension)
                            .to(torch.float32)
                            .mean()
                            .item()
                            if hasattr(m, "bond_dimension")
                            else None
                        ),
                        "train/cutoff": (
                            torch.tensor(m.cutoff).item()
                            if hasattr(m, "cutoff")
                            else None
                        ),
                        "eval/loss": torch.tensor(test_loss).item(),
                        "horizon": len(m.matrices),
                    }
                )
            except:
                lr *= lr_shrink
                print(
                    f"Shrinking lr to {lr} (loss: {m.Loss[-1]}, loss_last: {loss_last})"
                )
                if lr < lr_inf:
                    print("lr becomes negligible.")
                    wandb.finish()
                    return

                # Load checkpoint
                m.loadMPS(ckpt_dir)
                m.designate_data(dtset)
                m.init_cumulants()
                m.descenting_step_length = lr
            else:
                break

        loop_last += nlp

    wandb.finish()


def trainv2(
    m,
    max_loops=50,
    num_loops=5,
    lr=0.001,
    lr_shrink=0.9,
    safe_thresh=0.5,
    lr_inf=1e-10,
    batch_size=32,
    num_grad_steps=10,
    eps_trunc=1e-7,
    max_bond_dim=64,
    train_dataset_name="data/mnist/train.npy",
    test_dataset_name="data/mnist/test.npy",
    ckpt_dir="checkpoints/dmrg",
):

    # Setup
    ckpt_dir = os.path.join(ckpt_dir, strftime("MNIST-contin_on_%B_%d_%H%M"))
    os.makedirs(ckpt_dir, exist_ok=True)

    # Init HPs for first epoch
    print("Initializing MPS for first epoch...")
    m.verbose = 0
    m.left_cano()
    m.designate_data(np.load(train_dataset_name))
    m.init_cumulants()
    m.nbatch = 10
    m.descenting_step_length = 0.05
    m.descent_steps = 10
    m.cutoff = 0.3
    cut_rec = m.train(1, True)
    m.cutoff = cut_rec

    # 2. Continue the training, in a fixed cutoff, train until loopmax is finished
    m.verbose = 0
    m.cutoff = eps_trunc
    m.maxibond = max_bond_dim
    m.nbatch = batch_size
    m.descent_steps = num_grad_steps
    m.descenting_step_length = lr

    loss_prev = float("inf")
    print("Starting training...")
    for loop_idx in range(0, max_loops, num_loops):
        # Save checkpoint
        m.saveMPS(ckpt_dir)

        # Reset minibond after mean bond dimension > 10
        if (
            m.minibond > 1
            and torch.tensor(m.bond_dimension).to(torch.float32).mean() > 10
        ):
            m.minibond = 1
            print("Resetting minibond to 1")

        # Train
        m.train(num_loops, False)
        test_loss = m.Calc_Loss(np.load(test_dataset_name))
        wandb.log(
            {
                "train/loss": torch.tensor(m.Loss[-1]).item(),
                "train/lr": torch.tensor(lr).item(),
                "train/bond_mean": (
                    torch.tensor(m.bond_dimension).to(torch.float32).mean().item()
                    if hasattr(m, "bond_dimension")
                    else None
                ),
                "train/cutoff": (
                    torch.tensor(m.cutoff).item() if hasattr(m, "cutoff") else None
                ),
                "eval/loss": torch.tensor(test_loss).item(),
                "horizon": len(m.matrices),
            }
        )
        print(
            f"Loop {loop_idx+1}/{max_loops} | Train Loss: {m.Loss[-1]:.4f} | Test Loss: {test_loss:.4f}"
        )
        print(f"Bond dimension: {m.bond_dimension}")

        # Reset and decay LR
        if m.Loss[-1] - loss_prev > safe_thresh:
            lr *= lr_shrink
            if lr < lr_inf:
                print("Ending training due to negligible LR.")
                wandb.finish()
                return
            m.loadMPS(ckpt_dir)
            m.designate_data(np.load(train_dataset_name))
            m.init_cumulants()
            m.descenting_step_length = lr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_grad_steps", type=int, default=10)
    parser.add_argument("--num_loops", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--tags", type=str, nargs="*", default=[])
    args = parser.parse_args()

    # Data
    train_dataset_name = f"data/{args.dataset}/train.npy"
    test_dataset_name = f"data/{args.dataset}/test.npy"

    # Initialize wandb
    exp_name = f"{args.dataset}-epochs{args.epochs}-l{args.lr}-b{args.batch_size}-ngs{args.num_grad_steps}-nl{args.num_loops}-r{args.rank}"
    wandb.init(
        project="ptn-dmrg",
        name=exp_name,
        config=vars(args),
        tags=args.tags,
    )

    # Get num features
    num_samples, num_features = np.load(train_dataset_name).shape
    dv = args.device
    if dv is None:
        dv = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {dv}")
    m = MPS_c(num_features, device=dv)
    # trainv2(
    #     m,
    #     lr=args.lr,
    #     batch_size=args.batch_size,
    #     max_loops=args.epochs,
    #     num_loops=args.num_loops,
    #     max_bond_dim=args.rank,
    #     num_grad_steps=args.num_grad_steps,
    #     train_dataset_name=train_dataset_name,
    #     test_dataset_name=test_dataset_name,
    # )
    num_batch = num_samples // args.batch_size
    train(
        m,
        ckpt_dir=f"checkpoints/dmrg/{exp_name}",
        loopmax=args.epochs,
        lr=args.lr,
        train_dataset_name=train_dataset_name,
        test_dataset_name=test_dataset_name,
    )


# Looping:  20%|█████████████████▍                                                                     | 1/5 [02:45<11:01, 165.36s/it, loss=94.6]
