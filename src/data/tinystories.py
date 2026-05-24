import os
import numpy as np
from datasets import load_dataset
from transformers import GPT2TokenizerFast

def prepare_tinystories():
    print("正在下载并加载 TinyStories 数据集...")
    # 自动从 HuggingFace 下载
    dataset = load_dataset("roneneldan/TinyStories")

    print("正在加载 GPT-2 Tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def process(example):
        # 将文本转为 token id，并在末尾加上结束符 <|endoftext|>
        ids = tokenizer(example['text'])['input_ids']
        ids.append(tokenizer.eos_token_id)
        return {'ids': ids, 'len': len(ids)}

    print("正在进行分词处理 (可能需要几分钟)...")
    tokenized = dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=8, # 使用 8 进程加速，可根据你的 CPU 核心数调整
    )

    # 确定输出目录，统一放在 src/data/datasets/tinystories 下
    data_dir = os.path.dirname(__file__)
    out_dir = os.path.join(data_dir, "datasets", "tinystories")
    os.makedirs(out_dir, exist_ok=True)

    for split, dset in tokenized.items():
        # TinyStories 默认的验证集名字叫 'validation'，我们把它改成框架习惯的 'val'
        if split not in ['train', 'validation']:
            continue
        out_split = 'val' if split == 'validation' else split
        
        arr_len = np.sum(dset['len'])
        filename = os.path.join(out_dir, f'{out_split}.bin')
        
        # GPT2 词表大小不到 65535，使用 uint16 最省显存和硬盘
        dtype = np.uint16 
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
        print(f"正在写入 {out_split} 到 {filename} (共 {arr_len} 个 tokens)...")
        idx = 0
        for example in dset:
            length = example['len']
            arr[idx : idx + length] = example['ids']
            idx += length
        arr.flush()
        print(f"{out_split} 写入成功！")

if __name__ == '__main__':
    prepare_tinystories()