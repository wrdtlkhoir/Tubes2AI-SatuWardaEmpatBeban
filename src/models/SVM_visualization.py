import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import imageio.v2 as imageio
import os


def training_visualization_callback(model, epoch, X, y):
    if epoch % 5 == 0:
        obj = model._objective(X, y)
        y_pred = model.predict(X)
        acc = np.mean(y_pred == y)

        f1 = f1_score(y, y_pred, average="macro")

        if not hasattr(model, "history_loss"):
            model.history_loss = []
            model.history_acc = []
            model.history_f1 = []
            model.history_epochs = []

        model.history_loss.append(obj)
        model.history_acc.append(acc)
        model.history_f1.append(f1)
        model.history_epochs.append(epoch)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

        ax1.plot(model.history_epochs, model.history_loss, "b-o", linewidth=2)
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Loss (Objective)", fontsize=12)
        ax1.set_title("Val Loss", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        ax2.plot(model.history_epochs, model.history_acc, "g-o", linewidth=2)
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Accuracy", fontsize=12)
        ax2.set_title("Val Accuracy", fontsize=14, fontweight="bold")
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3)

        ax3.plot(model.history_epochs, model.history_f1, "r-o", linewidth=2)
        ax3.set_xlabel("Epoch", fontsize=12)
        ax3.set_ylabel("Macro F1 Score", fontsize=12)
        ax3.set_title("Val Macro F1 Score", fontsize=14, fontweight="bold")
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3)

        ax1.text(
            0.02,
            0.98,
            f"Loss: {obj:.4f}",
            transform=ax1.transAxes,
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        ax2.text(
            0.02,
            0.98,
            f"Acc: {acc:.4f}",
            transform=ax2.transAxes,
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        ax3.text(
            0.02,
            0.98,
            f"F1: {f1:.4f}",
            transform=ax3.transAxes,
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()
        plt.savefig(f"../test/frames/training_{epoch:04d}.png", dpi=80)
        plt.close()


def create_training_animation(
    output_path="../test/output/svm_training_progress.gif",
    fps=2,
    frames_dir="../test/frames",
):
    if not os.path.exists(frames_dir):
        return 0

    frames_list = sorted(
        [f for f in os.listdir(frames_dir) if f.startswith("training_")]
    )

    if len(frames_list) == 0:
        return 0

    frames = []
    for fname in frames_list:
        frames.append(imageio.imread(os.path.join(frames_dir, fname)))

    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"GIF saved : {output_path}")

    return len(frames_list)
