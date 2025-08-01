# train_test_pipeline.py
# FINAL-FINAL VERSION with the device placement fix for testing.

import yaml
import torch
import os
from argparse import ArgumentParser
from dotmap import DotMap

# --- Imports the ORIGINAL, UNCHANGED modules ---
from data.dataset_factory import get_and_split_data, create_dataloader
from models.base_vit import BaseVit
from models.moe_vit_karm import TemporalMoEViT
from train.trainer import Trainer

# --- Helper function to save and load checkpoints for the test ---
def save_best_model(model, config, model_type):
    """Saves the model state to the test checkpoint directory."""
    checkpoint_dir = config.logging.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"test_pipeline_{model_type}_best.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nSaved best model checkpoint for '{model_type}' to: {checkpoint_path}")
    return checkpoint_path

if __name__ == '__main__':
    parser = ArgumentParser(description="Run a FAST pipeline test on a minimal data split.")
    parser.add_argument('--test_config', type=str, default='config/test_pipeline.yaml')
    args = parser.parse_args()
    
    with open(args.test_config) as f:
        config = DotMap(yaml.safe_load(f))

    print("\n" + "="*80)
    print("--- STARTING FULL (TRAIN-VALIDATE-TEST) PIPELINE TEST ON MINIMAL DATA ---")
    print(f"Using FULL Test Config: {args.test_config}")
    print("="*80)

    # 1. Load the FULL metadata and vocab
    print("\n[PHASE 1] Loading full dataset metadata...")
    train_meta_full, val_meta_full, test_meta_full, vocab = get_and_split_data(config)
    print("Full metadata loaded successfully.")

    # 2. Slice the data into a tiny 4-2-2 split
    print("\n[PHASE 2] Slicing data into a 4-train, 2-val, 2-test split...")
    train_meta_test = train_meta_full[:4]
    val_meta_test = val_meta_full[:2]
    test_meta_test = test_meta_full[:2] 
    print(f"Data sliced: {len(train_meta_test)} train, {len(val_meta_test)} val, {len(test_meta_test)} test samples.")
    
    # 3. Create dataloaders for all three splits
    print("\n[PHASE 3] Creating dataloaders for all splits...")
    train_loader = create_dataloader(config, 'train', train_meta_test, vocab)
    val_loader = create_dataloader(config, 'val', val_meta_test, vocab)
    test_loader = create_dataloader(config, 'test', test_meta_test, vocab)
    print("Dataloaders created successfully.")

    # 4. The Main Loop: Train, Validate, and finally Test each model
    for model_type in ['base', 'moe']:
        print("\n" + "="*50)
        print(f"--- PROCESSING MODEL TYPE: {model_type.upper()} ---")
        
        # --- 4A: TRAINING & VALIDATION ---
        print("\n--- Running Training & Validation Phase ---")
        config.run_name = f"test_pipeline_{model_type}"
        
        if model_type == 'base': model = BaseVit(config)
        else: model = TemporalMoEViT(config)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
        trainer = Trainer(config, model, optimizer, train_loader, val_loader)
        trainer.train()

        best_checkpoint_path = save_best_model(model, config, model_type)
        
        # --- 4B: FINAL TESTING ---
        print(f"\n--- Running Final Testing Phase for '{model_type}' ---")
        if model_type == 'base': test_model = BaseVit(config)
        else: test_model = TemporalMoEViT(config)
            
        test_model.load_state_dict(torch.load(best_checkpoint_path, map_location=config.training.device))
        print("Successfully loaded best model checkpoint for testing.")

        # --- THIS IS THE FIX ---
        # We must move the newly loaded model to the GPU before evaluation.
        test_model.to(config.training.device)
        
        evaluator = Trainer(config, test_model, optimizer=None, train_loader=None, val_loader=test_loader)
        test_results = evaluator._run_epoch(epoch=0, is_training=False)
        
        print("\n" + "-"*50)
        print("--- TEST SET PERFORMANCE ---")
        print(f"Model Type: {model_type.upper()}")
        print(f"Final Test Accuracy: {test_results['accuracy']:.4f}")
        print("-"*50)

    print("\n" + "="*80)
    print("--- FULL (TRAIN-VALIDATE-TEST) PIPELINE TEST FINISHED SUCCESSFULLY! ---")
    print("="*80)