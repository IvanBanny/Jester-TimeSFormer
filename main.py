import os
import torch
import torch.nn as nn
from torch import optim
from dist_train import TrainingConfig, DistributedTrainer

from dataset import JesterDataset
from model import TimeSformer


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    # Training configuration
    config = TrainingConfig(
        epochs=50,
        batch_size=64,
        use_amp=True,
        max_grad_norm=1.0,
        early_stopping_patience=10,
        checkpoint_dir="checkpoints",
        save_checkpoint="best",
        wandb_project="jester-timesformer",
        init_checkpoint=None,
        gpu_ids=None,
        if_test=False,
        port=29427
    )

    # Dataset
    train_dataset = JesterDataset(
        root_dir="/data/s3647951/jester_dataset/",
        split='train',
        num_frames=16,
        frame_size=(224, 224),
        frame_stride=2
    )

    val_dataset = JesterDataset(
        root_dir="/data/s3647951/jester_dataset/",
        split='val',
        num_frames=16,
        frame_size=(224, 224),
        frame_stride=2,
        validation_subset=0.1
    )

    # val_dataset_full = JesterDataset(
    #     root_dir="/data/s3647951/jester_dataset/",
    #     split='val',
    #     num_frames=16,
    #     frame_size=(224, 224),
    #     frame_stride=2
    # )

    # Create model
    model = TimeSformer(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_frames=16,
        num_classes=27,
        embed_dim=512,
        depth=8,
        num_heads=8,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        time_first=True
    )
    model.set_gradient_checkpointing(True)

    # Setup loss, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.05,
        eps=1e-8,
        foreach=True
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        threshold=1e-4,
        min_lr=1e-7
    )

    # Initialize and start distributed training
    trainer = DistributedTrainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=None,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    trainer.train()


if __name__ == "__main__":
    main()
