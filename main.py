import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from dataset import JesterDataset
from model import create_model
from pathlib import Path
import wandb
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, epoch, device):
    """Train for one epoch."""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()

    progress_bar = tqdm(train_loader)
    for batch_idx, (videos, targets) in enumerate(progress_bar):
        videos, targets = videos.to(device), targets.to(device)

        # Zero gradients
        optimizer.zero_grad(set_to_none=True)  # More memory efficient

        # Mixed precision forward pass
        with autocast(device_type='cuda'):  # Updated autocast
            outputs = model(videos)
            loss = criterion(outputs, targets)

        # Mixed precision backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Compute accuracy
        with torch.no_grad():
            acc1 = (outputs.argmax(dim=1) == targets).float().mean() * 100

        # Update metrics
        losses.update(loss.item(), videos.size(0))
        top1.update(acc1.item(), videos.size(0))

        # Clear cache periodically
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

        progress_bar.set_description(
            f'Epoch: {epoch} | Loss: {losses.avg:.4f} | Acc: {top1.avg:.2f}%'
        )

    scheduler.step()
    return losses.avg, top1.avg


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for videos, targets in tqdm(val_loader):
            videos, targets = videos.to(device), targets.to(device)

            with autocast(device_type='cuda'):  # Updated autocast
                outputs = model(videos)
                loss = criterion(outputs, targets)

            # Compute accuracy
            acc1 = (outputs.argmax(dim=1) == targets).float().mean() * 100

            # Update metrics
            losses.update(loss.item(), videos.size(0))
            top1.update(acc1.item(), videos.size(0))

            # Clear cache periodically
            torch.cuda.empty_cache()

    return losses.avg, top1.avg


def train(args):
    """Main training function."""
    logging.info("Starting training setup...")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()  # Clear cache before starting

    # Initialize wandb
    wandb.init(project="jester-timesformer", config=args)

    # Create model
    model = create_model(
        distributed=False,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_frames=16,
        num_classes=27,
        embed_dim=512,  # 768
        depth=8,  # 12
        num_heads=8,  # 12
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        time_first=True
    ).to(device)

    # Enable gradient checkpointing
    model.set_gradient_checkpointing(True)

    logging.info("Model created successfully")

    # Create datasets and dataloaders
    train_dataset = JesterDataset(
        root_dir=args['data_path'],
        split='train',
        num_frames=16,
        frame_size=(224, 224)
    )

    val_dataset = JesterDataset(
        root_dir=args['data_path'],
        split='val',
        num_frames=16,
        frame_size=(224, 224)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args['batch_size'],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args['lr'],
        weight_decay=args['weight_decay']
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args['epochs'],
        eta_min=1e-6
    )

    # Updated GradScaler initialization
    scaler = GradScaler(device_type='cuda')

    logging.info("Starting training loop")
    best_acc = 0

    try:
        for epoch in range(args['epochs']):
            # Train and validate
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer,
                scheduler, scaler, epoch, device
            )

            val_loss, val_acc = validate(
                model, val_loader, criterion, device
            )

            # Log metrics
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': optimizer.param_groups[0]['lr']
            })

            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                }, f'checkpoints/best_model.pth')

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise
    finally:
        wandb.finish()
        torch.cuda.empty_cache()


def main():
    """Setup and start training."""
    args = {
        'data_path': '/path/to/jester/dataset',
        'batch_size': 16,  # 32
        'lr': 1e-4,
        'weight_decay': 0.05,
        'epochs': 50,
    }

    Path('checkpoints').mkdir(exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
