import os
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
import tiktoken

def prepare_data(dataset_name="tdooms/TinyStories", split="train", separator_token="<|endoftext|>", output_file="train.bin"):
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    separator_token_id = enc.encode_single_token(separator_token)

    def process(example):
        ids = enc.encode_ordinary(example['text'])  # encode_ordinary ignores any special tokens
        ids.append(separator_token_id)  # Add separator token at the end of each example
        return {'ids': ids, 'len': len(ids)}

    # Load and process the dataset
    if not os.path.exists(output_file):
        print(f"Processing {dataset_name} dataset...")
        ds = load_dataset(dataset_name, split=split)

        tokenized = ds.map(
            process,
            remove_columns=['text'],
            desc="Tokenizing the dataset",
            num_proc=8,
        )

        # Concatenate all ids into one large file
        arr_len = np.sum(tokenized['len'], dtype=np.uint64)
        dtype = np.uint16  # Can use uint16 since enc.max_token_value == 50256 is < 2**16
        arr = np.memmap(output_file, dtype=dtype, mode='w+', shape=(arr_len,))

        total_batches = 1024
        idx = 0

        for batch_idx in tqdm(range(total_batches), desc=f'Writing {output_file}'):
            # Batch together samples for faster write
            batch = tokenized.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)

        arr.flush()
        print(f"Dataset processed and saved to {output_file}")
    else:
        print(f"{output_file} already exists. Skipping processing.")

    # Load the processed data
    data = np.memmap(output_file, dtype=np.uint16, mode='r')
    return data, enc
