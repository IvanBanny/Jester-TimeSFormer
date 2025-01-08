import os
import json
import warnings
from typing import Tuple, List, Dict, Callable, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm
import wandb
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LRScheduler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


@dataclass
class TrainingConfig:
    """Configuration Class for distributed training parameters"""
    epochs: int
    batch_size: int
    use_amp: bool = False
    max_grad_norm: Optional[float] = None
    early_stopping_patience: int = 10
    checkpoint_dir: str = "checkpoints"
    save_checkpoint: str = "best"
    wandb_project: Optional[str] = None
    init_checkpoint: Optional[str] = None
    gpu_ids: Optional[List[int]] = None
    if_test: bool = False
    port: int = 29427


class DistributedTrainer:
    def __init__(
            self,
            config: TrainingConfig,
            model: torch.nn.Module,
            train_dataset: Dataset,
            val_dataset: Optional[Dataset] = None,
            test_dataset: Optional[Dataset] = None,
            criterion: Optional[torch.nn.Module] = None,
            optimizer: Optional[torch.optim.Optimizer] = None,
            scheduler: Optional[LRScheduler] = None
    ):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        if self.config.gpu_ids is None:
            self.config.gpu_ids = list(range(torch.cuda.device_count()))
        self.world_size = len(self.config.gpu_ids)

    @staticmethod
    def setup(rank_id: int, gpu_id: int, world_size: int, port: int):
        """Initialize the process group"""
        torch.cuda.empty_cache()
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)
        torch.cuda.set_device(gpu_id)
        dist.init_process_group("nccl", rank=rank_id, world_size=world_size)
        dist.barrier(device_ids=[gpu_id])

    @staticmethod
    def cleanup(gpu_id: int):
        """Cleanup the distributed environment"""
        if dist.is_initialized():
            # Ensure all processes are finished
            torch.cuda.synchronize()
            dist.barrier(device_ids=[gpu_id])
            dist.destroy_process_group()
            torch.cuda.empty_cache()

    def _train_worker(self, rank_id: int):
        """Main loop for each worker"""
        # warnings.filterwarnings("ignore")

        wandb_run = None
        if self.config.wandb_project is not None and rank_id == 0:
            wandb_run = wandb.init(
                project=self.config.wandb_project,
                config={
                    "epochs": self.config.epochs,
                    "batch_size": self.config.batch_size,
                    "use_amp": self.config.use_amp,
                    "max_grad_norm": self.config.max_grad_norm,
                    "early_stopping_patience": self.config.early_stopping_patience,
                    "gpu_ids": self.config.gpu_ids,
                    "init_checkpoint": self.config.init_checkpoint,
                    "save_checkpoint": self.config.save_checkpoint
                }
            )

        try:
            gpu_id = self.config.gpu_ids[rank_id]

            # Initialize the process group
            self.setup(rank_id, gpu_id, self.world_size, self.config.port)

            # Create samplers and dataloaders
            train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=rank_id,
                shuffle=True,
                seed=29
            )

            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                sampler=train_sampler,
                num_workers=8,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=3
            )

            if self.val_dataset:
                val_sampler = DistributedSampler(
                    self.val_dataset,
                    num_replicas=self.world_size,
                    rank=rank_id,
                    shuffle=False
                )

                val_loader = DataLoader(
                    self.val_dataset,
                    batch_size=self.config.batch_size,
                    sampler=val_sampler,
                    num_workers=2,
                    pin_memory=True
                )

            model = self.model.to(gpu_id)

            # Convert BatchNorm to SyncBatchNorm for distributed training
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

            # Initialize DDP model
            ddp_model = DDP(
                model,
                device_ids=[gpu_id],
                output_device=gpu_id,
                find_unused_parameters=False,
                gradient_as_bucket_view=True
            )

            # Initialize criterion if not provided
            if self.criterion is None:
                self.criterion = torch.nn.CrossEntropyLoss(
                    reduction="mean",
                    label_smoothing=0.1
                )

            # Initialize optimizer if not provided
            if self.optimizer is None:
                self.optimizer = torch.optim.SGD(
                    ddp_model.parameters(),
                    lr=0.001,
                    momentum=0.9,
                    dampening=0,
                    weight_decay=1e-4,
                    nesterov=True
                )

            if self.config.use_amp:
                scaler = GradScaler()

            # Metrics
            epoch = 0
            metrics = []
            current_lr = None

            best_accuracy = 0.0
            epochs_without_improvement = 0
            should_stop = torch.tensor(0, device=gpu_id)  # for early stopping

            # Load checkpoint if provided
            if self.config.init_checkpoint:
                base_path = os.path.join(self.config.checkpoint_dir, self.config.init_checkpoint.replace('.pth', ''))

                ddp_model.load_state_dict(torch.load(f"{base_path}_model.pth", map_location=f"cuda:{gpu_id}"))
                self.optimizer.load_state_dict(torch.load(f"{base_path}_optimizer.pth", map_location=f"cuda:{gpu_id}"))
                if self.scheduler is not None:
                    scheduler_path = f"{base_path}_scheduler.pth"
                    if os.path.exists(scheduler_path):
                        self.scheduler.load_state_dict(torch.load(scheduler_path, map_location=f"cuda:{gpu_id}"))

                metrics_path = f"{self.config.init_checkpoint}.json"
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                        if metrics:
                            epoch = metrics[-1].get("epoch", 1) - 1
                            current_lr = metrics[-1].get("lr", None)

            # Training loop
            for _ in range(self.config.epochs):
                # Set model to training mode
                ddp_model.train()
                # For shuffling to work properly
                train_sampler.set_epoch(epoch)

                # Make a pbar on rank 0
                if rank_id == 0:
                    print()
                    pbar = tqdm(
                        total=len(train_loader),
                        desc=f"Epoch: {epoch + 1}/{self.config.epochs}",
                        position=0,
                        leave=True
                    )

                # Epoch metrics
                running_loss = 0.0
                correct_predictions = 0
                total_samples = 0

                # Actual DDP training, loop through batches
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(gpu_id), labels.to(gpu_id)

                    # Reset gradients
                    self.optimizer.zero_grad(set_to_none=True)

                    if self.config.use_amp:
                        # Use autocast for mixed precision
                        with autocast('cuda'):
                            outputs = ddp_model(inputs)
                            loss = self.criterion(outputs, labels)

                        # Scale loss and do backward pass
                        scaler.scale(loss).backward()

                        if self.config.max_grad_norm:
                            scaler.unscale_(self.optimizer)
                            clip_grad_norm_(
                                ddp_model.parameters(),
                                self.config.max_grad_norm
                            )

                        # Optimizer step with scaler
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        # Non-AMP forward/backward
                        outputs = ddp_model(inputs)
                        loss = self.criterion(outputs, labels)
                        loss.backward()

                        if self.config.max_grad_norm:
                            clip_grad_norm_(
                                ddp_model.parameters(),
                                self.config.max_grad_norm
                            )

                        self.optimizer.step()

                    running_loss += loss.item() * labels.size(0)
                    _, predicted = outputs.max(1)
                    correct_predictions += (predicted == labels).sum().item()
                    total_samples += labels.size(0)

                    if rank_id == 0:
                        pbar.update(1)
                        pbar.set_postfix(loss=loss.item())

                if rank_id == 0:
                    pbar.close()

                # Gather the training metrics from all processes
                dist.all_reduce(torch.tensor([running_loss]).to(gpu_id), op=dist.ReduceOp.SUM)
                dist.all_reduce(torch.tensor([correct_predictions]).to(gpu_id), op=dist.ReduceOp.SUM)
                dist.all_reduce(torch.tensor([total_samples]).to(gpu_id), op=dist.ReduceOp.SUM)

                avg_train_loss = running_loss / total_samples
                avg_train_accuracy = correct_predictions / total_samples
                avg_val_loss, avg_val_accuracy = None, None

                # Validation
                if self.val_dataset is not None:
                    with torch.no_grad():
                        # Metrics
                        running_loss = 0.0
                        correct_predictions = 0
                        total_samples = 0

                        # Actual validation, loop through all batches
                        for inputs, labels in val_loader:
                            inputs, labels = inputs.to(gpu_id), labels.to(gpu_id)

                            outputs = ddp_model(inputs)
                            loss = self.criterion(outputs, labels)

                            running_loss += loss.item() * labels.size(0)
                            _, predicted = outputs.max(1)
                            correct_predictions += (predicted == labels).sum().item()
                            total_samples += labels.size(0)

                        # Gather the validation metrics from all processes
                        dist.all_reduce(torch.tensor([running_loss]).to(gpu_id), op=dist.ReduceOp.SUM)
                        dist.all_reduce(torch.tensor([correct_predictions]).to(gpu_id), op=dist.ReduceOp.SUM)
                        dist.all_reduce(torch.tensor([total_samples]).to(gpu_id), op=dist.ReduceOp.SUM)

                        avg_val_loss = running_loss / total_samples
                        avg_val_accuracy = correct_predictions / total_samples

                if epoch % 5 == 0:  # Clear every 5 epochs
                    torch.cuda.empty_cache()

                # Log metrics, save model
                if rank_id == 0:
                    epoch_metrics = {
                        "epoch": epoch + 1,
                        "train_loss": avg_train_loss,
                        "train_accuracy": avg_train_accuracy,
                        "val_loss": avg_val_loss,
                        "val_accuracy": avg_val_accuracy
                    }
                    if current_lr is not None:
                        epoch_metrics["lr"] = current_lr

                    metrics.append(epoch_metrics)
                    with open(f"{self.config.save_checkpoint}.json", 'w') as f:
                        json.dump(metrics, f, indent=4)
                    if self.config.wandb_project is not None:
                        wandb.log(epoch_metrics)
                    print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}")
                    if avg_val_loss is not None and avg_val_accuracy is not None:
                        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy:.4f}")

                    # Check for early stopping
                    current_accuracy = (avg_val_accuracy if avg_val_accuracy is not None else avg_train_accuracy)
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        epochs_without_improvement = 0

                        # Save the best model
                        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
                        base_path = os.path.join(self.config.checkpoint_dir, self.config.save_checkpoint)
                        torch.save(ddp_model.module.state_dict(), f"{base_path}_model.pth")
                        torch.save(self.optimizer.state_dict(), f"{base_path}_optimizer.pth")
                        if self.scheduler is not None:
                            torch.save(self.scheduler.state_dict(), f"{base_path}_scheduler.pth")
                    else:
                        epochs_without_improvement += 1

                        # Early stopping condition
                        if epochs_without_improvement >= self.config.early_stopping_patience:
                            print(f"Early stopping at epoch {epoch + 1}")
                            should_stop = torch.tensor(1, device=gpu_id)

                # Broadcast the early stopping decision to all processes
                dist.barrier(device_ids=[gpu_id])
                dist.broadcast(should_stop, src=self.config.gpu_ids[0])
                # Early stopping
                if should_stop.item():
                    break

                # Step the scheduler
                if self.scheduler is not None:
                    self.scheduler.step(avg_train_loss if avg_val_loss is None else avg_val_loss)

                    if self.scheduler.get_last_lr()[0] != current_lr:
                        current_lr = self.scheduler.get_last_lr()[0]
                        if epoch != 0:
                            print(f"New learning rate: {current_lr}")

                epoch += 1
        finally:
            if wandb_run is not None:
                wandb_run.finish()
            self.cleanup(gpu_id)

    def train(self):
        """Main function to start distributed training"""
        # For logging
        if self.config.wandb_project is not None:
            wandb.login()

        try:
            # Clean up any existing CUDA memory
            torch.cuda.empty_cache()

            # Spawn trainer processes for each GPU
            mp.spawn(
                self._train_worker,
                nprocs=self.world_size,
                join=True
            )
        except Exception as e:
            print(f"Error during training: {e}")
        finally:
            # Ensure cleanup
            torch.cuda.empty_cache()
            if wandb.run is not None:
                wandb.finish()
