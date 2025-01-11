import os
import argparse
import torch
import torch.nn as nn
from torch import optim
from dist_train import TrainingConfig, DistributedTrainer

from dataset import JesterDataset
from model import TimeSformer


def create_model():
    return TimeSformer(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_frames=16,
        num_classes=27,
        embed_dim=384,
        depth=8,
        num_heads=6,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.1
    )


def create_optimizer(ddp_model):
    return optim.AdamW(
        ddp_model.parameters(),
        lr=1e-4,
        weight_decay=0.05,
        eps=1e-8,
        foreach=True
    )


def create_scheduler(optimizer):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=5,
        threshold=1e-4,
        min_lr=1e-7
    )


def main():
    parser = argparse.ArgumentParser(description="TimeSFormer training configuration")
    parser.add_argument("--test-only", action="store_true",
                        help="Run only testing (no training)")
    parser.add_argument("--gpu-ids", type=str, default=None,
                        help="Comma-separated list of GPU IDs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory for checkpoints")
    parser.add_argument("--init-checkpoint", type=str, default=None,
                        help="Initial checkpoint to load")
    parser.add_argument("--save-checkpoint", type=str, default="best",
                        help="Name for saving checkpoint")
    parser.add_argument("--port", type=int, default=29427,
                        help="Port for distributed training")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # Training config
    config = TrainingConfig(
        epochs=50,
        batch_size=16,
        train_workers=24,
        val_workers=12,
        prefetch_factor=8,
        use_amp=True,
        max_grad_norm=1.0,
        early_stopping_patience=10,
        checkpoint_dir=args.checkpoint_dir,
        save_checkpoint=args.save_checkpoint,
        wandb_project="jester-timesformer",
        init_checkpoint=args.init_checkpoint,
        gpu_ids=args.gpu_ids,
        if_train=not args.test_only,
        if_test=True,
        port=args.port
    )

    # Dataset config
    train_dataset = JesterDataset(
        root_dir="/data/s3647951/jester_dataset/",
        split="train",
        num_frames=16,
        frame_size=(224, 224),
        frame_stride=2
    )

    val_dataset = JesterDataset(
        root_dir="/data/s3647951/jester_dataset/",
        split="val",
        num_frames=16,
        frame_size=(224, 224),
        frame_stride=2,
        validation_subset=0.1
    )

    val_dataset_full = JesterDataset(
        root_dir="/data/s3647951/jester_dataset/",
        split="val",
        num_frames=16,
        frame_size=(224, 224),
        frame_stride=2
    ) if config.if_test else None

    # Criterion config
    criterion = nn.CrossEntropyLoss()

    # Initialize the distributed trainer
    trainer = DistributedTrainer(
        config=config,
        create_model=create_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=val_dataset_full,
        criterion=criterion,
        create_optimizer=create_optimizer,
        create_scheduler=create_scheduler,
    )

    # Start distributed training
    trainer.train()


if __name__ == "__main__":
    main()
