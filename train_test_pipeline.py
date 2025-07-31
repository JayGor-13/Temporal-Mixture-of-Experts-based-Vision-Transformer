# train_test_pipeline.py
# A special, NON-DESTRUCTIVE script to run a full pipeline test on a tiny data subset.
# FINAL CORRECTED VERSION - This harmonizes configs perfectly.

import yaml
import torch
from argparse import ArgumentParser
from dotmap import DotMap

# --- Imports the ORIGINAL, UNCHANGED modules ---
from data.dataset_factory import get_and_split_data, create_dataloader
from models.base_vit import BaseVit
from models.moe_vit_karm import TemporalMoEViT
from train.trainer import Trainer

if __name__ == '__main__':
    parser = ArgumentParser(description="Run a FAST pipeline test on a minimal data split.")
    parser.add_argument('--test_config', type=str, default='config/test_pipeline.yaml')
    args = parser.parse_args()
    
    with open(args.test_config) as f:
        config = DotMap(yaml.safe_load(f))

    print("\n" + "="*50)
    print("--- STARTING PIPELINE TEST ON MINIMAL DATA SPLIT ---")
    print(f"Using FULL Test Config: {args.test_config}")
    print("="*50)

    print("\n[Step 1] Loading full dataset metadata...")
    train_meta_full, val_meta_full, test_meta_full, vocab = get_and_split_data(config)
    print("Full metadata loaded successfully.")

    print("\n[Step 2] Slicing data into a 4-train, 2-val, 2-test split...")
    train_meta_test, val_meta_test = train_meta_full[:4], val_meta_full[:2]
    print(f"Data sliced: {len(train_meta_test)} train, {len(val_meta_test)} val samples.")
    
    print("\n[Step 3] Creating dataloaders for the tiny data split...")
    train_loader = create_dataloader(config, 'train', train_meta_test, vocab)
    val_loader = create_dataloader(config, 'val', val_meta_test, vocab)
    print("Dataloaders created successfully.")

    for model_type in ['base', 'moe']:
        print("\n" + "-"*50)
        print(f"--- TESTING MODEL TYPE: {model_type.upper()} ---")
        
        # --- NEW: Set a descriptive run name for plotting ---
        config.run_name = f"test_pipeline_{model_type}"
        
        if model_type == 'base': model = BaseVit(config)
        else: model = TemporalMoEViT(config)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
        # The new Trainer will automatically print complexity analysis and plot history
        trainer = Trainer(config, model, optimizer, train_loader, val_loader)
        trainer.train()

    print("\n" + "="*50)
    print("--- PIPELINE TEST FINISHED SUCCESSFULLY! ---")
    print("="*50)