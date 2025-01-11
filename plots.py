import json
import matplotlib.pyplot as plt
import sys
import os


def create_plots(json_name):
    json_path = f"checkpoints/{json_name}.json"

    # Check if file exists
    if not os.path.exists(json_path):
        print(f"Error: File {json_path} not found")
        sys.exit(1)

    with open(json_path, "r") as f:
        metrics = json.load(f)

    # Extract values from the metrics list
    epochs = [epoch["epoch"] for epoch in metrics]
    train_losses = [epoch["train_loss"] for epoch in metrics]
    val_losses = [epoch["val_loss"] for epoch in metrics]
    train_accuracies = [epoch["train_accuracy"] for epoch in metrics]
    val_accuracies = [epoch["val_accuracy"] for epoch in metrics]

    has_lr = "lr" in metrics[0]
    if has_lr:
        learning_rates = [epoch["lr"] for epoch in metrics]
        # Create separate figure for learning rate
        plt.figure(figsize=(7, 6))
        plt.plot(epochs, learning_rates)
        plt.xlabel("Epochs")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.xticks([1] + epochs[9::10] + [epochs[-1]])
        plt.tight_layout()
        plt.savefig("lr_plot.png", dpi=300)
        plt.close()

    # Plot Training and Validation Loss
    plt.figure(figsize=(14, 6))

    # Subplot for Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.xticks([1] + epochs[9::10] + [epochs[-1]])

    # Subplot for Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Training Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.xticks([1] + epochs[9::10] + [epochs[-1]])

    plt.tight_layout()
    plt.savefig("metrics_plot.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plots.py <json_file_name>")
        print("Example: python plots.py metrics")
        sys.exit(1)

    create_plots(sys.argv[1])
