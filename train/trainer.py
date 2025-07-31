# train/trainer.py
# FINAL DIAGNOSTIC VERSION

import torch
import torch.nn as nn
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .losses import calculate_total_loss

class Trainer:
    """
    The Trainer class encapsulates the entire training and validation loop,
    now with comprehensive performance tracking and plotting.
    """
    def __init__(self, config, model, optimizer, train_loader, val_loader):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config.training.device
        
        # --- NEW: History tracker ---
        self.history = []

    def _run_epoch(self, epoch, is_training):
        """Helper function to run a single epoch of training or validation."""
        self.model.train(is_training)
        dataloader = self.train_loader if is_training else self.val_loader
        mode = "Train" if is_training else "Val"
        
        total_task_loss = 0
        total_aux_loss_val = 0 # To track MoE aux loss
        correct_predictions = 0
        total_samples = 0
        
        epoch_start_time = time.time()

        for i, batch in enumerate(dataloader):
            # A more robust check for tensors before moving to device
            batch_on_device = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_on_device[k] = v.to(self.device)
                else:
                    batch_on_device[k] = v # Keep non-tensor data as is
            
            with torch.set_grad_enabled(is_training):
                model_output = self.model(batch_on_device)
                
                if isinstance(model_output, tuple):
                    logits, aux_loss = model_output
                    total_aux_loss_val += aux_loss.item()
                else:
                    logits, aux_loss = model_output, 0.0

                labels = batch_on_device['answer_label']
                total_loss_batch, task_loss_batch = calculate_total_loss(
                    logits, labels, aux_loss, self.config.training.loss_alpha
                )

            if is_training:
                self.optimizer.zero_grad()
                total_loss_batch.backward()
                self.optimizer.step()

            total_task_loss += task_loss_batch.item()
            preds = torch.argmax(logits, dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

        # --- CONSOLIDATED METRICS FOR THE EPOCH ---
        epoch_duration = time.time() - epoch_start_time
        avg_loss = total_task_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        avg_aux_loss = total_aux_loss_val / len(dataloader) if total_aux_loss_val > 0 else 0
        
        print(
            f"Epoch {epoch+1:02d}/{self.config.training.epochs:02d} | [{mode}] "
            f"Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | "
            f"AuxLoss: {avg_aux_loss:.4f} | Time: {epoch_duration:.2f}s"
        )
        
        # --- NEW: Append results to history ---
        return {
            'epoch': epoch + 1, 'mode': mode, 'loss': avg_loss,
            'accuracy': accuracy, 'aux_loss': avg_aux_loss
        }

    def train(self):
        """ Main training loop with history tracking and plotting. """
        self.model.to(self.device)
        print(f"\n--- Starting Training for Run: {self.config.run_name} ---")
        print(f"Device: {self.device}")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable (Active) Parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

        for epoch in range(self.config.training.epochs):
            # Run train epoch and save history
            train_results = self._run_epoch(epoch, is_training=True)
            self.history.append(train_results)

            if self.val_loader:
                # Run validation epoch and save history
                val_results = self._run_epoch(epoch, is_training=False)
                self.history.append(val_results)
        
        # --- NEW: Plot history after training is complete ---
        self.plot_history()
        
    def plot_history(self):
        """ Plots training and validation curves for loss and accuracy. """
        if not self.history:
            print("No history to plot.")
            return
            
        df = pd.DataFrame(self.history)
        
        # Create a 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f'Training & Validation Analysis for: {self.config.run_name}', fontsize=16)
        
        # Plot 1: Loss curves
        sns.lineplot(data=df, x='epoch', y='loss', hue='mode', marker='o', ax=axes[0, 0])
        axes[0, 0].set_title('Task Loss vs. Epochs')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Cross-Entropy Loss')
        axes[0, 0].grid(True)

        # Plot 2: Accuracy curves
        sns.lineplot(data=df, x='epoch', y='accuracy', hue='mode', marker='o', ax=axes[0, 1])
        axes[0, 1].set_title('Accuracy vs. Epochs')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True)
        
        # Plot 3: Auxiliary Loss (only for MoE models)
        moe_df = df[df['aux_loss'] > 0]
        if not moe_df.empty:
            sns.lineplot(data=moe_df, x='epoch', y='aux_loss', hue='mode', marker='o', ax=axes[1, 0])
            axes[1, 0].set_title('MoE Auxiliary Loss vs. Epochs')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Router Load-Balancing Loss')
            axes[1, 0].grid(True)
        else:
            axes[1, 0].set_title('No Auxiliary Loss (Base Model)')
            axes[1, 0].axis('off')

        # Plot 4: Final Validation Accuracy Bar
        val_df = df[df['mode'] == 'Val']
        if not val_df.empty:
            best_acc = val_df['accuracy'].max()
            sns.barplot(x=['Best Validation Accuracy'], y=[best_acc], ax=axes[1, 1])
            axes[1, 1].set_title(f'Peak Validation Accuracy: {best_acc:.4f}')
            axes[1, 1].set_ylim(0, max(1.0, best_acc * 1.1)) # Adjust y-limit
            axes[1, 1].text(0, best_acc, f'{best_acc:.4f}', ha='center', va='bottom')
        else:
             axes[1, 1].axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the plot to the main project directory
        plot_filename = f"{self.config.run_name}_performance_plot.png"
        plt.savefig(plot_filename)
        print(f"\nPerformance plot saved to {plot_filename}")
        plt.show()