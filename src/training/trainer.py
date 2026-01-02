"""Main training loop and trainer class."""

import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, List, Any
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from .metrics import MetricsTracker
from .losses import get_loss_function
from .schedulers import get_scheduler
from ..utils.helpers import (
    save_checkpoint, load_checkpoint, 
    get_device, format_time, ensure_dir
)
from ..utils.logger import get_logger


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Basic settings
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    
    # Optimizer
    optimizer: str = 'adamw'
    momentum: float = 0.9  # For SGD
    
    # Scheduler
    scheduler_type: str = 'cosine'
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.1
    scheduler_patience: int = 10
    warmup_epochs: int = 5
    
    # Loss function
    loss_function: str = 'cross_entropy'
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    
    # Early stopping
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001
    
    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    save_best_only: bool = True
    save_every_n_epochs: int = 10
    
    # Mixed precision
    use_amp: bool = True
    
    # Gradient clipping
    gradient_clip: float = 1.0
    
    # Logging
    log_interval: int = 10
    
    # Misc
    num_workers: int = 4
    device: Optional[str] = None
    
    @classmethod
    def from_config(cls, config: Dict) -> 'TrainingConfig':
        """Create TrainingConfig from configuration dictionary."""
        training_config = config.get('training', {})
        scheduler_config = training_config.get('scheduler', {})
        early_stopping_config = training_config.get('early_stopping', {})
        
        return cls(
            epochs=training_config.get('epochs', 100),
            batch_size=training_config.get('batch_size', 32),
            learning_rate=training_config.get('learning_rate', 0.001),
            weight_decay=training_config.get('weight_decay', 0.0001),
            optimizer=training_config.get('optimizer', 'adamw'),
            scheduler_type=scheduler_config.get('type', 'cosine'),
            scheduler_step_size=scheduler_config.get('step_size', 30),
            scheduler_gamma=scheduler_config.get('gamma', 0.1),
            scheduler_patience=scheduler_config.get('patience', 10),
            early_stopping_enabled=early_stopping_config.get('enabled', True),
            early_stopping_patience=early_stopping_config.get('patience', 15),
            early_stopping_min_delta=early_stopping_config.get('min_delta', 0.001),
            checkpoint_dir=training_config.get('checkpoint_dir', 'checkpoints'),
            save_best_only=training_config.get('save_best_only', True),
        )


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class Trainer:
    """
    Main trainer class for ISAR classifiers.
    
    Handles:
    - Training loop with validation
    - Mixed precision training
    - Checkpointing
    - Early stopping
    - Metrics tracking
    - Learning rate scheduling
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        class_names: Optional[List[str]] = None,
        callbacks: Optional[List[Callable]] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            class_names: List of class names
            callbacks: Optional list of callback functions
        """
        self.config = config
        self.class_names = class_names or []
        self.callbacks = callbacks or []
        self.logger = get_logger()
        
        # Device
        self.device = get_device(config.device)
        self.model = model.to(self.device)
        
        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Loss function
        self.criterion = get_loss_function(
            config.loss_function,
            label_smoothing=config.label_smoothing,
            focal_gamma=config.focal_gamma
        )
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = get_scheduler(
            self.optimizer,
            config.scheduler_type,
            epochs=config.epochs,
            steps_per_epoch=len(train_loader),
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma,
            patience=config.scheduler_patience
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config.use_amp else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta
        ) if config.early_stopping_enabled else None
        
        # Metrics tracker
        self.metrics_tracker = MetricsTracker(class_names=class_names)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.training_history = []
        
        # Ensure checkpoint directory exists
        ensure_dir(config.checkpoint_dir)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        params = self.model.parameters()
        
        if self.config.optimizer.lower() == 'adam':
            return torch.optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'adamw':
            return torch.optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'sgd':
            return torch.optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def train(self) -> Dict[str, Any]:
        """
        Run the full training loop.
        
        Returns:
            Dictionary containing training history and best metrics
        """
        self.logger.info(f"Starting training on {self.device}")
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        self.logger.info(f"Epochs: {self.config.epochs}, Batch size: {self.config.batch_size}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self._train_epoch()
            
            # Validation phase
            val_metrics = self._validate_epoch()
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Record history
            epoch_metrics = {
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            self.training_history.append(epoch_metrics)
            
            # Log progress
            self.logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}"
            )
            
            # Save checkpoints
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.best_val_acc = val_metrics['accuracy']
            
            if is_best or not self.config.save_best_only:
                self._save_checkpoint(is_best, val_metrics)
            
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(False, val_metrics, f"checkpoint_epoch_{epoch + 1}.pt")
            
            # Callbacks
            for callback in self.callbacks:
                callback(self, epoch_metrics)
            
            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping(val_metrics['loss']):
                    self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {format_time(total_time)}")
        self.logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
        
        # Save training history
        self._save_history()
        
        return {
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'total_time': total_time,
            'epochs_trained': self.current_epoch + 1,
            'history': self.training_history
        }
    
    def _train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        self.metrics_tracker.reset()
        
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.config.use_amp and self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                loss.backward()
                
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            preds = outputs.argmax(dim=1)
            self.metrics_tracker.update(targets.cpu(), preds.cpu())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / num_batches,
                'acc': self.metrics_tracker.get_accuracy()
            })
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': self.metrics_tracker.get_accuracy(),
            **self.metrics_tracker.compute_metrics()
        }
    
    @torch.no_grad()
    def _validate_epoch(self) -> Dict[str, float]:
        """Run validation epoch."""
        self.model.eval()
        self.metrics_tracker.reset()
        
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")
        
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            if self.config.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            preds = outputs.argmax(dim=1)
            self.metrics_tracker.update(targets.cpu(), preds.cpu())
            
            pbar.set_postfix({
                'loss': total_loss / num_batches,
                'acc': self.metrics_tracker.get_accuracy()
            })
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': self.metrics_tracker.get_accuracy(),
            **self.metrics_tracker.compute_metrics()
        }
    
    def _save_checkpoint(
        self,
        is_best: bool,
        metrics: Dict[str, float],
        filename: Optional[str] = None
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'metrics': metrics,
            'config': self.config.__dict__
        }
        
        if filename is None:
            filename = 'best_model.pt' if is_best else 'last_model.pt'
        
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        
        if is_best:
            self.logger.info(f"Saved best model with val_loss={metrics['loss']:.4f}")
    
    def _save_history(self):
        """Save training history to JSON."""
        history_path = os.path.join(self.config.checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint['best_val_acc']
        
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
