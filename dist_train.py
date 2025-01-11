import os
import json
from typing import Tuple, List, Dict, Callable, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm
import wandb
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.amp import autocast, GradScaler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


@dataclass
class TrainingConfig:
    """Configuration Class for distributed training parameters"""
    epochs: int
    batch_size: int
    train_workers: int = 24
    val_workers: int = 12
    prefetch_factor: int = 8
    use_amp: bool = False
    max_grad_norm: Optional[float] = None
    early_stopping_patience: int = 10
    checkpoint_dir: str = "checkpoints"
    save_checkpoint: str = "best"
    wandb_project: Optional[str] = None
    init_checkpoint: Optional[str] = None
    gpu_ids: Optional[List[int]] = None
    if_train: bool = True
    if_test: bool = False
    port: int = 29427


class DistributedTrainer:
    def __init__(
            self,
            config: TrainingConfig,
            create_model: Callable,
            train_dataset: Dataset,
            val_dataset: Optional[Dataset] = None,
            test_dataset: Optional[Dataset] = None,
            criterion: Optional[torch.nn.Module] = None,
            create_optimizer: Optional[Callable] = None,
            create_scheduler: Optional[Callable] = None
    ):
        self.config = config
        self.create_model = create_model
        self.ddp_model = None
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.criterion = criterion
        self.create_optimizer = create_optimizer
        self.optimizer = None
        self.create_scheduler = create_scheduler
        self.scheduler = None

        if self.config.gpu_ids is None:
            self.config.gpu_ids = list(range(torch.cuda.device_count()))
        self.world_size = len(self.config.gpu_ids)

        self.rank_id = None
        self.gpu_id = None

        self.metrics = []
        self.epoch = 0
        self.current_lr = None

    def _setup(self):
        """Initialize the process group"""
        torch.cuda.empty_cache()
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(self.config.port)
        self.gpu_id = self.config.gpu_ids[self.rank_id]
        torch.cuda.set_device(self.gpu_id)
        dist.init_process_group("nccl", rank=self.rank_id, world_size=self.world_size)
        dist.barrier(device_ids=[self.gpu_id])

    def _cleanup(self):
        """Cleanup the distributed environment"""
        if dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier(device_ids=[self.gpu_id])
            dist.destroy_process_group()

        if self.rank_id == 0 and wandb.run is not None:
            wandb.finish()

        self.rank_id = None
        self.gpu_id = None
        self.ddp_model = None
        torch.cuda.empty_cache()

    def _load_checkpoint(self, checkpoint_name: str, model_only: bool = False, load_scheduler=True):
        base_path = os.path.join(self.config.checkpoint_dir, checkpoint_name)

        self.ddp_model.module.load_state_dict(torch.load(f"{base_path}_model.pth",
                                                         map_location=f"cuda:{self.gpu_id}", weights_only=True))
        if not model_only:
            self.optimizer.load_state_dict(torch.load(f"{base_path}_optimizer.pth",
                                                      map_location=f"cuda:{self.gpu_id}", weights_only=True))
            if load_scheduler and self.scheduler is not None:
                scheduler_path = f"{base_path}_scheduler.pth"
                if os.path.exists(scheduler_path):
                    self.scheduler.load_state_dict(torch.load(scheduler_path,
                                                              map_location=f"cuda:{self.gpu_id}", weights_only=True))

            metrics_path = f"{base_path}.json"
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    self.metrics = json.load(f)
                    if self.metrics:
                        self.epoch = self.metrics[-1].get("epoch", 1)
                        self.current_lr = self.metrics[-1].get("lr", None)

    def _save_checkpoint(self, checkpoint_name):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        base_path = os.path.join(self.config.checkpoint_dir, checkpoint_name)

        with open(f"{base_path}.json", 'w') as f:
            json.dump(self.metrics, f, indent=4)
        torch.save(self.ddp_model.module.state_dict(), f"{base_path}_model.pth")
        torch.save(self.optimizer.state_dict(), f"{base_path}_optimizer.pth")
        if self.scheduler is not None:
            torch.save(self.scheduler.state_dict(), f"{base_path}_scheduler.pth")

    def _train_pass(self, data_sampler, data_loader):
        if self.config.use_amp:
            scaler = GradScaler()

        # Set model to training mode
        self.ddp_model.train()
        # For shuffling to work properly
        data_sampler.set_epoch(self.epoch)

        # Make a pbar on rank 0
        if self.rank_id == 0:
            print()
            pbar = tqdm(
                total=len(data_loader),
                desc=f"Epoch: {self.epoch + 1}/{self.config.epochs}",
                position=0,
                leave=True
            )

        # Epoch metrics
        running_loss = torch.tensor(0.0, device=self.gpu_id)
        correct_predictions = torch.tensor(0, device=self.gpu_id)
        total_samples = torch.tensor(0, device=self.gpu_id)

        # Actual DDP training, loop through batches
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(self.gpu_id), labels.to(self.gpu_id)

            # Reset gradients
            self.optimizer.zero_grad(set_to_none=True)

            if self.config.use_amp:
                # Use autocast for mixed precision
                with autocast("cuda"):
                    outputs = self.ddp_model(inputs)
                    loss = self.criterion(outputs, labels)

                # Scale loss and do backward pass
                scaler.scale(loss).backward()

                if self.config.max_grad_norm:
                    scaler.unscale_(self.optimizer)
                    clip_grad_norm_(
                        self.ddp_model.parameters(),
                        self.config.max_grad_norm
                    )

                # Optimizer step with scaler
                scaler.step(self.optimizer)
                scaler.update()
            else:
                # Non-AMP forward/backward
                outputs = self.ddp_model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()

                if self.config.max_grad_norm:
                    clip_grad_norm_(
                        self.ddp_model.parameters(),
                        self.config.max_grad_norm
                    )

                self.optimizer.step()

            torch.cuda.synchronize()

            running_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            if self.rank_id == 0:
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

        if self.rank_id == 0:
            pbar.close()

        # Gather the training metrics from all processes
        dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_predictions, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

        return running_loss.item() / total_samples.item(), correct_predictions.item() / total_samples.item()

    def _test_pass(self, data_loader):
        """For validation or testing"""
        self.ddp_model.eval()

        with torch.no_grad():
            # Metrics
            running_loss = torch.tensor(0.0, device=self.gpu_id)
            correct_predictions = torch.tensor(0, device=self.gpu_id)
            total_samples = torch.tensor(0, device=self.gpu_id)

            # Actual validation, loop through all batches
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.gpu_id), labels.to(self.gpu_id)

                outputs = self.ddp_model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

            # Gather the validation metrics from all processes
            dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct_predictions, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

            # return avg_val_loss, avg_val_accuracy
            return running_loss.item() / total_samples.item(), correct_predictions.item() / total_samples.item()

    def _train_worker(self, rank_id: int):
        """Main loop for each worker"""
        try:
            # Initialize the process group
            self.rank_id = rank_id
            self._setup()

            wandb_run = None

            if self.config.if_train:
                # For logging
                if self.config.wandb_project is not None:
                    wandb.login()

                if self.config.wandb_project is not None and self.rank_id == 0:
                    wandb_run = wandb.init(
                        project=self.config.wandb_project,
                        config={
                            "epochs": self.config.epochs,
                            "batch_size": self.config.batch_size,
                            "train_workers": self.config.train_workers,
                            "val_workers": self.config.val_workers,
                            "prefetch_factor": self.config.prefetch_factor,
                            "use_amp": self.config.use_amp,
                            "max_grad_norm": self.config.max_grad_norm,
                            "early_stopping_patience": self.config.early_stopping_patience,
                            "gpu_ids": self.config.gpu_ids,
                            "init_checkpoint": self.config.init_checkpoint,
                            "save_checkpoint": self.config.save_checkpoint
                        }
                    )

                # Create samplers and dataloaders
                train_sampler = DistributedSampler(
                    self.train_dataset,
                    num_replicas=self.world_size,
                    rank=self.rank_id,
                    shuffle=True,
                    seed=29
                )

                train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=self.config.batch_size,
                    sampler=train_sampler,
                    num_workers=self.config.train_workers,
                    pin_memory=True,
                    persistent_workers=True,
                    prefetch_factor=self.config.prefetch_factor
                )

                if self.val_dataset:
                    val_sampler = DistributedSampler(
                        self.val_dataset,
                        num_replicas=self.world_size,
                        rank=self.rank_id,
                        shuffle=False
                    )

                    val_loader = DataLoader(
                        self.val_dataset,
                        batch_size=self.config.batch_size,
                        sampler=val_sampler,
                        num_workers=self.config.val_workers,
                        pin_memory=True
                    )

            if self.config.if_test and self.test_dataset:
                test_sampler = DistributedSampler(
                    self.test_dataset,
                    num_replicas=self.world_size,
                    rank=self.rank_id,
                    shuffle=False
                )

                test_loader = DataLoader(
                    self.test_dataset,
                    batch_size=self.config.batch_size,
                    sampler=test_sampler,
                    num_workers=self.config.val_workers,
                    pin_memory=True
                )

            # Move model
            model = self.create_model().to(self.gpu_id)

            if self.rank_id == 0:
                print(f"\nModel parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

            # Convert BatchNorm to SyncBatchNorm for distributed training
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

            # Initialize DDP model
            self.ddp_model = DDP(
                model,
                device_ids=[self.gpu_id],
                output_device=self.gpu_id,
                find_unused_parameters=False,
                gradient_as_bucket_view=True
            )

            if self.config.if_train:
                # Initialize criterion if not provided
                if self.criterion is None:
                    self.criterion = torch.nn.CrossEntropyLoss(
                        reduction="mean",
                        label_smoothing=0.1
                    )

                # Initialize optimizer
                self.optimizer = torch.optim.SGD(
                    self.ddp_model.parameters(),
                    lr=0.001,
                    momentum=0.9,
                    dampening=0,
                    weight_decay=1e-4,
                    nesterov=True
                ) if self.create_optimizer is None else self.create_optimizer(self.ddp_model)

                # Initialize scheduler if provided
                if self.create_scheduler is not None:
                    self.scheduler = self.create_scheduler(self.optimizer)
                    self.current_lr = self.scheduler.get_last_lr()[0]

            # Load checkpoint if provided
            if self.config.init_checkpoint:
                self._load_checkpoint(self.config.init_checkpoint, model_only=not self.config.if_train)

            if self.config.if_train:
                best_loss = float("inf")
                epochs_without_improvement = 0
                should_stop = torch.tensor(0, device=self.gpu_id)  # for early stopping

                # Training loop
                for _ in range(self.config.epochs):
                    # Train pass
                    avg_train_loss, avg_train_accuracy = self._train_pass(train_sampler, train_loader)

                    # Validation pass
                    avg_val_loss, avg_val_accuracy = None, None
                    if self.val_dataset:
                        avg_val_loss, avg_val_accuracy = self._test_pass(val_loader)

                    dist.barrier(device_ids=[self.gpu_id])

                    if self.epoch % 5 == 0:  # Clear every 5 epochs
                        torch.cuda.empty_cache()

                    # Log metrics, save model
                    if self.rank_id == 0:
                        print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}")
                        if avg_val_loss is not None and avg_val_accuracy is not None:
                            print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy:.4f}")

                        epoch_metrics = {
                            "epoch": self.epoch + 1,
                            "train_loss": avg_train_loss,
                            "train_accuracy": avg_train_accuracy,
                            "val_loss": avg_val_loss,
                            "val_accuracy": avg_val_accuracy
                        }
                        if self.current_lr is not None:
                            epoch_metrics["lr"] = self.current_lr

                        self.metrics.append(epoch_metrics)
                        if self.config.wandb_project is not None:
                            wandb.log(epoch_metrics)

                        # Check for early stopping
                        current_loss = (avg_val_loss if avg_val_loss is not None else avg_train_loss)
                        if current_loss < best_loss:
                            best_loss = current_loss
                            epochs_without_improvement = 0

                            # Save the best model
                            self._save_checkpoint(self.config.save_checkpoint)
                        else:
                            epochs_without_improvement += 1

                            # Early stopping condition
                            if epochs_without_improvement >= self.config.early_stopping_patience:
                                print(f"Early stopping at epoch {self.epoch + 1}")
                                should_stop = torch.tensor(1, device=self.gpu_id)

                    # Broadcast the early stopping decision to all processes
                    dist.barrier(device_ids=[self.gpu_id])
                    dist.broadcast(should_stop, src=self.config.gpu_ids[0])
                    # Early stopping
                    if should_stop.item():
                        break

                    # Step the scheduler
                    if self.scheduler is not None and avg_val_loss is not None:
                        self.scheduler.step(avg_val_loss)

                        # Get new lr on rank 0
                        if self.rank_id == 0:
                            new_lr = torch.tensor(self.scheduler.get_last_lr()[0], device=self.gpu_id)
                        else:
                            new_lr = torch.tensor(0., device=self.gpu_id)

                        # Broadcast new_lr from rank 0 to all processes
                        dist.broadcast(new_lr, src=self.config.gpu_ids[0])
                        new_lr = new_lr.item()

                        if abs(new_lr - self.current_lr) > 1e-10:
                            if self.epoch != 0:
                                if self.rank_id == 0:
                                    print(f"\nNew learning rate: {new_lr}")

                                # Make sure all processes finished broadcasting
                                dist.barrier(device_ids=[self.gpu_id])

                                # Load full checkpoint to maintain consistency
                                self._load_checkpoint(self.config.save_checkpoint, load_scheduler=False)

                                # Update the optimizer's learning rate after loading
                                for param_group in self.optimizer.param_groups:
                                    param_group['lr'] = new_lr

                                self.current_lr = new_lr

                    self.epoch += 1

            if self.config.if_test:
                avg_test_loss, avg_test_accuracy = self._test_pass(test_loader)
                if self.rank_id == 0:
                    print(f"\nTest Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.4f}\n")
        finally:
            if wandb_run is not None:
                wandb_run.finish()
            self._cleanup()

    def train(self):
        """Main function to start distributed training"""
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
            self._cleanup()
