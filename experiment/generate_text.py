#!/usr/bin/env python3
"""
Text generation script using trained language model.

This script loads a trained checkpoint and generates text using various sampling strategies
including temperature scaling and nucleus (top-p) sampling.

Supports multiple datasets with automatic configuration selection:
- TinyStories: Children's stories dataset (vocab_size=10000, simple vocabulary)
- OpenWebText: Web text dataset (vocab_size=32000, complex vocabulary)

The script automatically:
- Selects the correct Hydra configuration (model + data configs)
- Loads the appropriate tokenizer files
- Validates vocabulary size consistency

Usage:
    python generate_text.py --dataset tinystories
    python generate_text.py --dataset openwebtext --model_path /path/to/checkpoint.pt
"""

import sys
import torch
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import argparse

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.generation import generate_text
from cs336_basics.checkpoint import load_checkpoint


def load_trained_model(checkpoint_path: str, config_path: str = None, dataset: str = "tinystories"):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        config_path: Path to the config file (optional, will use default if not provided)
        dataset: Dataset type ('tinystories' or 'openwebtext') to determine tokenizer files
    
    Returns:
        model: Loaded model
        tokenizer: Tokenizer for encoding/decoding
        config: Configuration used for training
    """
    # Load configuration
    if config_path is None:
        config_path = project_root / "experiment" / "conf"
    
    # Create dataset-specific configuration overrides
    config_overrides = []
    if dataset.lower() == "tinystories":
        config_overrides = [
            "model=tinystories",
            "data=tinystories"
        ]
    elif dataset.lower() == "openwebtext":
        config_overrides = [
            "model=openwebtext", 
            "data=openwebtext"
        ]
    
    with hydra.initialize_config_dir(config_dir=str(config_path), version_base=None):
        cfg = hydra.compose(config_name="config", overrides=config_overrides)
    
    print(f"Using configuration for dataset: {dataset}")
    print(f"Model vocab size: {cfg.model.vocab_size}")
    print(f"Model config: {cfg.model}")
    print(f"Data config: {cfg.data}")
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.device = device
    
    # Initialize model using Hydra's instantiate
    model = hydra.utils.instantiate(cfg.model).to(device)
    
    # Initialize optimizer (needed for loading checkpoint, even though we won't use it)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    
    # Load checkpoint
    iteration = load_checkpoint(checkpoint_path, model, optimizer)
    print(f"Loaded checkpoint from iteration {iteration}")
    
    # Load tokenizer based on dataset
    data_dir = project_root / "data"
    
    # Determine tokenizer files based on dataset
    if dataset.lower() == "tinystories":
        vocab_file = data_dir / "TinyStoriesV2-GPT4-train-vocab.json"
        merges_file = data_dir / "TinyStoriesV2-GPT4-train-merges.txt"
    elif dataset.lower() == "openwebtext":
        vocab_file = data_dir / "owt_train-vocab.json"
        merges_file = data_dir / "owt_train-merges.txt"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Supported datasets: 'tinystories', 'openwebtext'")
    
    # Check if tokenizer files exist
    if not vocab_file.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")
    if not merges_file.exists():
        raise FileNotFoundError(f"Merges file not found: {merges_file}")
    
    tokenizer = Tokenizer.from_files(
        str(vocab_file),
        str(merges_file),
        special_tokens=["<|endoftext|>"]
    )
    
    # Verify tokenizer vocab size matches model vocab size
    tokenizer_vocab_size = len(tokenizer.vocab)
    model_vocab_size = cfg.model.vocab_size
    
    print(f"Tokenizer vocab size: {tokenizer_vocab_size}")
    print(f"Model vocab size: {model_vocab_size}")
    
    if tokenizer_vocab_size != model_vocab_size:
        print(f"WARNING: Vocabulary size mismatch!")
        print(f"  Tokenizer: {tokenizer_vocab_size}")
        print(f"  Model: {model_vocab_size}")
        print(f"  This may cause issues during generation.")
    else:
        print("✓ Vocabulary sizes match")
    
    return model, tokenizer, cfg


def generate_sample_text(model, tokenizer, cfg, prompt="Once upon a time", **generation_kwargs):
    """
    Generate text from a prompt using the trained model.
    
    Args:
        model: Trained language model
        tokenizer: Tokenizer for encoding/decoding
        cfg: Configuration object
        prompt: Text prompt to start generation
        **generation_kwargs: Additional arguments for generation
    
    Returns:
        generated_text: Generated text string
        input_length: Length of input prompt in tokens
        output_length: Length of generated tokens
    """
    # Set model to evaluation mode
    model.eval()
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=cfg.device)
    input_length = len(input_ids)
    
    print(f"Prompt: '{prompt}'")
    print(f"Input token IDs: {input_ids}")
    print(f"Input length: {input_length} tokens")
    
    # Generate text
    with torch.no_grad():
        # Get EOS token ID
        eos_token_id = None
        if "<|endoftext|>" in tokenizer.special_tokens:
            eos_token_id = tokenizer.encode("<|endoftext|>")[0]
        
        generated_tokens = generate_text(
            model=model,
            input_ids=input_tensor,
            eos_token_id=eos_token_id,
            **generation_kwargs
        )
    
    # Decode the generated tokens
    generated_ids = generated_tokens[0].cpu().tolist()
    generated_text = tokenizer.decode(generated_ids)
    output_length = len(generated_ids) - input_length
    
    print(f"Generated {output_length} new tokens")
    print(f"Total length: {len(generated_ids)} tokens")
    
    return generated_text, input_length, output_length


def run_examples(model, tokenizer, cfg, dataset="tinystories"):
    """Run example generations with different sampling strategies."""
    
    print("\n" + "="*80)
    print("EXAMPLE GENERATIONS WITH DIFFERENT SAMPLING STRATEGIES")
    print("="*80)
    
    # Test prompts based on dataset
    if dataset.lower() == "tinystories":
        prompts = [
            "Once upon a time",
            "The little girl"
        ]
    elif dataset.lower() == "openwebtext":
        prompts = [
            "The future of artificial intelligence",
            "In recent years, technology has"
        ]
    else:
        # Default prompts
        prompts = [
            "Once upon a time",
            "The little girl"
        ]
    
    # Generate text with different sampling strategies
    sampling_configs = [
        {
            "name": "Greedy Decoding",
            "params": {
                "max_new_tokens": 100,
                "temperature": 1.0,
                "do_sample": False
            }
        },
        {
            "name": "Temperature Sampling (T=0.8)",
            "params": {
                "max_new_tokens": 100,
                "temperature": 0.8,
                "do_sample": True
            }
        },
        {
            "name": "Nucleus Sampling (p=0.9, T=0.8)",
            "params": {
                "max_new_tokens": 100,
                "temperature": 0.8,
                "top_p": 0.9,
                "do_sample": True
            }
        }
    ]
    
    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"EXAMPLE PROMPT: '{prompt}'")
        print('='*60)
        
        for config in sampling_configs:
            print(f"\n--- {config['name']} ---")
            
            try:
                generated_text, input_len, output_len = generate_sample_text(
                    model, tokenizer, cfg, prompt, **config['params']
                )
                
                print(f"\nGenerated text ({output_len} tokens):")
                print("-" * 40)
                print(generated_text)
                print("-" * 40)
                
                # Check if we hit EOS token
                if "<|endoftext|>" in generated_text:
                    eos_pos = generated_text.find("<|endoftext|>")
                    actual_output = generated_text[len(prompt):eos_pos]
                    print(f"Note: Generation stopped at EOS token after {len(tokenizer.encode(actual_output))} tokens")
                
            except Exception as e:
                print(f"Error during generation: {e}")
                continue
            
            print("\n" + "-"*40)


def interactive_generation(model, tokenizer, cfg, args, dataset="tinystories"):
    """Interactive text generation with user input."""
    
    print(f"\n{'='*80}")
    print("INTERACTIVE TEXT GENERATION")
    print('='*80)
    print("Enter your own prompts and adjust parameters as needed.")
    print("Type 'quit' to exit, 'examples' to see example generations.")
    print('='*80)
    
    while True:
        print(f"\nCurrent parameters:")
        print(f"- Max new tokens: {args.max_tokens}")
        print(f"- Temperature: {args.temperature}")
        print(f"- Top-p: {args.top_p if args.top_p else 'None'}")
        print(f"- Sampling: {'Enabled' if args.do_sample else 'Greedy'}")
        
        print(f"\nOptions:")
        print("1. Generate text with current parameters")
        print("2. Change parameters")
        print("3. Show examples")
        print("4. Quit")
        
        choice = input("\nChoose an option (1-4): ").strip()
        
        if choice == "1":
            prompt = input("\nEnter your prompt: ").strip()
            if not prompt:
                print("Please enter a non-empty prompt.")
                continue
                
            print(f"\nGenerating text for prompt: '{prompt}'")
            print(f"Parameters: max_tokens={args.max_tokens}, temperature={args.temperature}, top_p={args.top_p}, sampling={args.do_sample}")
            
            try:
                generated_text, input_len, output_len = generate_sample_text(
                    model, tokenizer, cfg, prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p if args.top_p > 0 else None,
                    do_sample=args.do_sample
                )
                
                print(f"\n{'='*60}")
                print(f"GENERATED TEXT ({output_len} new tokens)")
                print('='*60)
                print(generated_text)
                print('='*60)
                
                # Check if we hit EOS token
                if "<|endoftext|>" in generated_text:
                    eos_pos = generated_text.find("<|endoftext|>")
                    actual_output = generated_text[len(prompt):eos_pos]
                    print(f"Note: Generation stopped at EOS token after {len(tokenizer.encode(actual_output))} tokens")
                    
            except Exception as e:
                print(f"Error during generation: {e}")
                
        elif choice == "2":
            print(f"\nCurrent parameters:")
            print(f"1. Max new tokens: {args.max_tokens}")
            print(f"2. Temperature: {args.temperature}")
            print(f"3. Top-p: {args.top_p if args.top_p else 'None'}")
            print(f"4. Sampling mode: {'Enabled' if args.do_sample else 'Greedy'}")
            
            param_choice = input("Which parameter to change (1-4)? ").strip()
            
            if param_choice == "1":
                try:
                    new_tokens = int(input(f"Enter max new tokens (current: {args.max_tokens}): "))
                    if new_tokens > 0:
                        args.max_tokens = new_tokens
                        print(f"Max tokens updated to {new_tokens}")
                    else:
                        print("Please enter a positive number.")
                except ValueError:
                    print("Please enter a valid integer.")
                    
            elif param_choice == "2":
                try:
                    new_temp = float(input(f"Enter temperature (current: {args.temperature}): "))
                    if new_temp > 0:
                        args.temperature = new_temp
                        print(f"Temperature updated to {new_temp}")
                    else:
                        print("Temperature must be positive.")
                except ValueError:
                    print("Please enter a valid float.")
                    
            elif param_choice == "3":
                try:
                    new_top_p = input(f"Enter top-p (current: {args.top_p if args.top_p else 'None'}, enter 0 for None): ")
                    if new_top_p.lower() == 'none' or new_top_p == '0':
                        args.top_p = 0
                        print("Top-p disabled")
                    else:
                        top_p_val = float(new_top_p)
                        if 0 < top_p_val <= 1:
                            args.top_p = top_p_val
                            print(f"Top-p updated to {top_p_val}")
                        else:
                            print("Top-p must be between 0 and 1.")
                except ValueError:
                    print("Please enter a valid float or 0.")
                    
            elif param_choice == "4":
                current_mode = "sampling" if args.do_sample else "greedy"
                new_mode = input(f"Current mode: {current_mode}. Enter 'sampling' or 'greedy': ").strip().lower()
                if new_mode in ['sampling', 'greedy']:
                    args.do_sample = (new_mode == 'sampling')
                    print(f"Mode updated to {new_mode}")
                else:
                    print("Please enter 'sampling' or 'greedy'.")
            else:
                print("Invalid choice.")
                
        elif choice == "3":
            run_examples(model, tokenizer, cfg, dataset)
            
        elif choice == "4":
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please enter 1-4.")


def main():
    """Main function with argument parsing and interactive generation."""
    
    parser = argparse.ArgumentParser(
        description="Text generation with trained language model",
        epilog="""
Examples:
  # Use TinyStories model with default checkpoint
  python generate_text.py --dataset tinystories
  
  # Use OpenWebText model with custom checkpoint
  python generate_text.py --dataset openwebtext --model_path /path/to/owt_checkpoint.pt
  
  # Run only examples without interactive mode
  python generate_text.py --dataset tinystories --examples_only
  
  # Use greedy decoding instead of sampling
  python generate_text.py --dataset openwebtext --greedy
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="tinystories", choices=["tinystories", "openwebtext"], 
                        help="Dataset type for tokenizer selection (default: tinystories)")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum new tokens to generate (default: 100)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling (default: 0.8)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for nucleus sampling (default: 0.9, set to 0 to disable)")
    parser.add_argument("--do_sample", action="store_true", default=True, help="Enable sampling (default: True)")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of sampling")
    parser.add_argument("--examples_only", action="store_true", help="Only run examples, no interactive mode")
    
    args = parser.parse_args()
    
    # Handle greedy flag
    if args.greedy:
        args.do_sample = False
    
    # Determine checkpoint path
    if args.model_path:
        checkpoint_path = Path(args.model_path)
    else:
        # Use default paths - try to find recent checkpoints
        outputs_dir = project_root / "outputs"
        checkpoint_path = None
        
        # Look for checkpoints in recent output directories
        if outputs_dir.exists():
            # Get most recent directories
            output_dirs = sorted([d for d in outputs_dir.iterdir() if d.is_dir()], reverse=True)
            for output_dir in output_dirs:
                # Look for subdirectories (timestamp directories)
                timestamp_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()], reverse=True)
                for timestamp_dir in timestamp_dirs:
                    # Look for checkpoint files
                    checkpoints = list(timestamp_dir.glob("checkpoint_*.pt"))
                    if checkpoints:
                        # Use the highest numbered checkpoint
                        checkpoint_path = sorted(checkpoints, key=lambda x: int(x.stem.split('_')[1]))[-1]
                        break
                if checkpoint_path:
                    break
        
        if not checkpoint_path or not checkpoint_path.exists():
            print(f"No trained checkpoint found for dataset '{args.dataset}'.")
            print("Please specify --model_path or train a model first.")
            print(f"Example usage: python generate_text.py --dataset {args.dataset} --model_path /path/to/checkpoint.pt")
            return
    
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return
        
    print(f"Using checkpoint: {checkpoint_path}")
    
    # Load model and tokenizer
    print(f"Loading model and tokenizer for dataset: {args.dataset}")
    model, tokenizer, cfg = load_trained_model(str(checkpoint_path), dataset=args.dataset)
    print("Model loaded successfully!")
    
    # Run examples if requested or start interactive mode
    if args.examples_only:
        run_examples(model, tokenizer, cfg, args.dataset)
    else:
        interactive_generation(model, tokenizer, cfg, args, args.dataset)


if __name__ == "__main__":
    main()
