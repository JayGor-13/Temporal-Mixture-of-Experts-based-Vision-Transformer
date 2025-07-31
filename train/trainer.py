# train/trainer.py
# FINAL ANALYSIS VERSION

import torch
import torch.nn as nn
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from fvcore.nn import FlopCountAnalysis
from .losses import calculate_total_loss

class Trainer:
    """
    The Trainer class, now enhanced with comprehensive performance and complexity analysis,
    including FLOPs counting and MoE router gate visualization.
    """
    def __init__(self, config, model, optimizer, train_loader, val_loader):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config.training.device
        self.history = []
        
        # --- NEW: For MoE gate analysis ---
        self.gate_stats = {}
        self.hooks = []
        self.is_moe = 'moe' in self.config.run_name

    def _analyze_model_complexity(self):
        """ Calculates and prints FLOPs and parameter counts. """
        # Get a single batch to determine input shapes
        dummy_batch = next(iter(self.train_loader))
        dummy_batch_on_device = {k: v.to(self.device) for k, v in dummy_batch.items() if isinstance(v, torch.Tensor)}
        
        # Calculate FLOPs
        flops = FlopCountAnalysis(self.model, (dummy_batch_on_device,))
        total_flops = flops.total()

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\n--- Model Complexity Analysis: {self.config.run_name} ---")
        print(f"Total Parameters: {total_params/1e6:.2f}M")
        print(f"Trainable (Active) Parameters: {trainable_params/1e6:.2f}M ({100 * trainable_params / total_params:.2f}%)")
        print(f"FLOPs (per sample): {total_flops/1e9:.2f} GFLOPs")
        print("----------------------------------------------------\n")
        
        # Store for final plot
        self.complexity_stats = {
            'params': total_params, 'flops': total_flops
        }

    def _register_hooks(self):
        """ Registers forward hooks to capture router gating decisions. """
        if not self.is_moe: return
        
        def hook_fn(module, input, output):
            # The router output is (top_k_indices, routing_weights, aux_loss)
            top_k_indices = output[0]
            self.gate_stats[module].append(top_k_indices.detach().cpu())

        for i, layer in enumerate(self.model.modules()):
            if hasattr(layer, 'gate_layer'): # Identify router by unique attribute
                self.gate_stats[layer] = []
                self.hooks.append(layer.register_forward_hook(hook_fn))
        print(f"Registered {len(self.hooks)} hooks for MoE gate analysis.")

    def _remove_hooks(self):
        """ Removes all registered hooks. """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def _run_epoch(self, epoch, is_training):
        # (This function is identical to the previous "FINAL DIAGNOSTIC VERSION")
        # ... no changes needed here ...
        self.model.train(is_training)
        dataloader = self.train_loader if is_training else self.val_loader
        mode = "Train" if is_training else "Val"
        
        total_task_loss, total_aux_loss_val = 0, 0
        correct_predictions, total_samples = 0, 0
        epoch_start_time = time.time()

        for batch in dataloader:
            batch_on_device = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            with torch.set_grad_enabled(is_training):
                model_output = self.model(batch_on_device)
                logits, aux_loss = (model_output, 0.0) if isinstance(model_output, torch.Tensor) else model_output
                if self.is_moe: total_aux_loss_val += aux_loss.item()
                
                labels = batch_on_device['answer_label']
                total_loss_batch, task_loss_batch = calculate_total_loss(logits, labels, aux_loss, self.config.training.loss_alpha)

            if is_training:
                self.optimizer.zero_grad()
                total_loss_batch.backward()
                self.optimizer.step()

            total_task_loss += task_loss_batch.item()
            preds = torch.argmax(logits, dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

        epoch_duration = time.time() - epoch_start_time
        avg_loss = total_task_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        avg_aux_loss = total_aux_loss_val / len(dataloader) if self.is_moe else 0
        
        print(f"Epoch {epoch+1:02d}/{self.config.training.epochs:02d} | [{mode}] Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | AuxLoss: {avg_aux_loss:.4f} | Time: {epoch_duration:.2f}s")
        
        return {'epoch': epoch + 1, 'mode': mode, 'loss': avg_loss, 'accuracy': accuracy, 'aux_loss': avg_aux_loss}


    def train(self):
        """ Main training loop with integrated analysis. """
        self.model.to(self.device)
        print(f"\n--- Preparing Run: {self.config.run_name} ---")
        
        # 1. Analyze Complexity (FLOPs, Params)
        self._analyze_model_complexity()

        # 2. Register hooks for MoE models
        self._register_hooks()
        
        print(f"--- Starting Training on {self.device} ---")
        for epoch in range(self.config.training.epochs):
            self.history.append(self._run_epoch(epoch, is_training=True))
            if self.val_loader:
                self.history.append(self._run_epoch(epoch, is_training=False))
        
        # 3. Process collected stats and plot everything
        self.plot_analysis()
        
        # 4. Clean up hooks
        self._remove_hooks()

    def plot_analysis(self):
        """ Creates and saves a comprehensive 2x3 analysis plot. """
        if not self.history: return
        df = pd.DataFrame(self.history)
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle(f'Comprehensive Analysis for: {self.config.run_name}', fontsize=20)
        
        # Plot 1: Loss curves
        sns.lineplot(data=df, x='epoch', y='loss', hue='mode', marker='o', ax=axes[0, 0])
        axes[0, 0].set_title('A. Task Loss vs. Epochs')
        axes[0, 0].grid(True)

        # Plot 2: Accuracy curves
        sns.lineplot(data=df, x='epoch', y='accuracy', hue='mode', marker='o', ax=axes[0, 1])
        axes[0, 1].set_title('B. Accuracy vs. Epochs')
        axes[0, 1].grid(True)
        
        # Plot 3: Router Gate Analysis (Heatmap)
        if self.is_moe and self.gate_stats:
            num_layers = len(self.gate_stats.keys())
            num_experts = self.model.config.model.moe.num_experts
            gate_heatmap = np.zeros((num_layers, num_experts))
            
            for i, (module, stats) in enumerate(self.gate_stats.items()):
                all_indices = torch.cat(stats).flatten().numpy()
                counts = np.bincount(all_indices, minlength=num_experts)
                gate_heatmap[i] = counts / counts.sum() * 100
                
            sns.heatmap(gate_heatmap, annot=True, fmt=".1f", cmap="viridis", ax=axes[1, 0],
                        xticklabels=[f"E{i}" for i in range(num_experts)],
                        yticklabels=[f"L{i}" for i in range(num_layers)])
            axes[1, 0].set_title('C. MoE Gate Activation (%) per Layer')
            axes[1, 0].set_xlabel('Expert ID')
            axes[1, 0].set_ylabel('Transformer Layer')
        else:
            axes[1, 0].set_title('C. MoE Gate Analysis (N/A for Base Model)')
            axes[1, 0].axis('off')

        # Plot 4: Complexity vs. Performance
        val_df = df[df['mode'] == 'Val']
        best_acc = val_df['accuracy'].max() if not val_df.empty else 0
        
        # This plot is more meaningful when comparing multiple models,
        # but we can show the stats for the current one.
        stats_df = pd.DataFrame({
            'Metric': ['GFLOPs', 'Best Accuracy'],
            'Value': [self.complexity_stats['flops'] / 1e9, best_acc],
            'Model': [self.config.run_name, self.config.run_name]
        })
        sns.barplot(data=stats_df, x='Metric', y='Value', ax=axes[1, 1])
        axes[1, 1].set_title('D. Performance & Efficiency')
        axes[1, 1].set_ylabel('')
        for p in axes[1, 1].patches:
             axes[1, 1].annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                                 ha='center', va='center', xytext=(0, 9), textcoords='offset points')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_filename = f"{self.config.run_name}_full_analysis.png"
        plt.savefig(plot_filename)
        print(f"\nFull analysis plot saved to {plot_filename}")
        plt.show()