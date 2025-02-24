from datasets import load_from_disk, Dataset, concatenate_datasets
from collections import Counter
import re
from typing import Dict, List, Tuple
from tqdm import tqdm
import random
import os
from glob import glob
from pathlib import Path

def clean_text(text: str) -> str:
    """Removing extra whitespace."""
    return re.sub(r'\s+', ' ', text).strip()

def build_character_vocabulary(dataset: Dataset) -> set:
    """Generate set of unique characters"""
    unique_chars = set()
    
    # Collect unique characters across the dataset
    for i in tqdm(range(len(dataset)), desc="Building character vocabulary"):
        text = dataset[i]['text']
        chars = set(clean_text(text))  # convert text to set of characters
        unique_chars.update(chars)
    
    return unique_chars

def create_train_val_split(dataset: Dataset, 
                          val_size: float = 0.1,
                          seed: int = 42) -> Tuple[Dataset, Dataset]:
    # Shuffle and split the dataset
    dataset = dataset.shuffle(seed=seed)
    val_size = int(len(dataset) * val_size)
    
    train_dataset = dataset.select(range(len(dataset) - val_size))
    val_dataset = dataset.select(range(len(dataset) - val_size, len(dataset)))
    
    return train_dataset, val_dataset

def save_dataset_to_txt(dataset: Dataset, filename: str):
    """Save dataset to a txt file"""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in tqdm(dataset, desc=f"Saving {filename}"):
            f.write(clean_text(item['text']) + '\n')

def main():
    # Create openwebtext folder if it doesn't exist
    output_dir = Path("openwebtext")
    output_dir.mkdir(exist_ok=True)
    
    # DEFINE THE PATH TO YOUR DATA HERE
    path = "./data"
    
    print("Loading dataset...")
    
    # all arrow files
    arrow_files = glob(os.path.join(path, "*.arrow"))
    if not arrow_files:
        raise ValueError(f"No Arrow files found in {path}")
    
    # concatenate all Arrow files
    datasets_list = []
    for arrow_file in tqdm(arrow_files, desc="Loading Arrow files"):
        dataset = Dataset.from_file(arrow_file)
        datasets_list.append(dataset)
    
    # concat all datasets
    dataset = concatenate_datasets(datasets_list)
    print(f"Loaded dataset with {len(dataset)} examples")
    
    # build vocab
    print("Building character vocabulary...")
    vocabulary = build_character_vocabulary(dataset)
    
    # save vocab to txt
    print("Saving vocabulary...")
    vocab_path = output_dir / 'vocab.txt'
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for char in sorted(vocabulary):
            f.write(f"{char}\n")
    
    # train-val split (90-10)
    print("Creating train-validation split (90-10)...")
    train_dataset, val_dataset = create_train_val_split(dataset, val_size=0.1)
    
    # save splits to txt files
    print("Saving splits to txt files...")
    save_dataset_to_txt(train_dataset, output_dir / "train_split.txt")
    save_dataset_to_txt(val_dataset, output_dir / "val_split.txt")
    
    # debugging stats
    print("\nDataset Statistics:")
    print(f"Total examples: {len(dataset)}")
    print(f"Character vocabulary size: {len(vocabulary)}")
    print(f"\nFiles saved:")
    print(f"- Vocabulary: {os.path.abspath(vocab_path)}")
    print(f"- Training data: {os.path.abspath(output_dir / 'train_split.txt')}")
    print(f"- Validation data: {os.path.abspath(output_dir / 'val_split.txt')}")

if __name__ == "__main__":
    main()