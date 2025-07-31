# train/train.py
import yaml
import torch
from argparse import ArgumentParser
from dotmap import DotMap
from data.dataset_factory import get_and_split_data, create_dataloader
from models.base_vit import BaseVit
from models.moe_vit_karm import TemporalMoEViT
from train.trainer import Trainer

if __name__ == '__main__':
    parser = ArgumentParser(description="Train a model on the MSVD dataset.")
    parser.add_argument('--config', type=str, default='config/training_msvd.yaml')
    parser.add_argument('--model_type', type=str, required=True, choices=['moe', 'base'])
    args = parser.parse_args()
    with open(args.config) as f: config = DotMap(yaml.safe_load(f))
    
    # --- NEW: Set a descriptive run name for plotting ---
    config.run_name = f"full_project_{args.model_type}"

    print(f"\n--- Loading and Splitting Data for '{args.model_type.upper()}' ---")
    train_meta, val_meta, _, vocab = get_and_split_data(config)
    
    train_loader = create_dataloader(config, 'train', train_meta, vocab)
    val_loader = create_dataloader(config, 'val', val_meta, vocab)

    print(f"\n--- Initializing {args.model_type.upper()} model ---")
    if args.model_type == 'base': model = BaseVit(config)
    else: model = TemporalMoEViT(config)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)

    # The new Trainer will automatically print params and plot at the end
    trainer = Trainer(config, model, optimizer, train_loader, val_loader)
    trainer.train()
    
    print(f"\n--- Training for {args.model_type.upper()} finished successfully. ---")