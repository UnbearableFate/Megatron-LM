import os
import json
from datasets import load_dataset
os.environ["HF_DATASETS_CACHE"] = "/work/NBB/share/datasets"
os.environ["TRANSFORMERS_CACHE"] = "/work/NBB/share/pre-trained-models"
os.environ["HF_DATASETS_OFFLINE"] = "1"  # 如果你是在计算节点上运行

def export_split_to_jsonl(dataset, split, out_file, num_proc=8, batch_size=1000):
    """
    1) 并行地把每条记录映射成 JSON 字符串；
    2) 批量写出到单个 .jsonl 文件。
    """
    # 第一步：并行 map，生成新的 column "json"（每条都是已经序列化好的 JSON）
    ds = dataset[split]
    print(f"[{split}] mapping to JSON strings with {num_proc} processes ...")
    ds_json = ds.map(
        lambda ex: {"json": json.dumps({"text": ex["text"]}, ensure_ascii=False)},
        batched=False,
        num_proc=num_proc,
        remove_columns=ds.column_names
    )
    # 第二步：顺序写文件（I/O 基本是线性的，但 JSON 序列化已经在多进程中完成）
    print(f"[{split}] writing out to {out_file} ...")
    with open(out_file, "w", encoding="utf-8") as fout:
        for line in ds_json["json"]:
            fout.write(line + "\n")
    print(f"[{split}] done.")

def main():
    # —— 可选：调整缓存目录 和 离线模式 ——
    print("Loading HuggingFace wikitext-103-raw-v1 …")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

    out_dir = "/work/NBB/share/datasets/wikitext/magatron-wikitext103"
    os.makedirs(out_dir, exist_ok=True)
    # 并行导出 train & validation
    export_split_to_jsonl(dataset, "train",
                          os.path.join(out_dir, "wikitext103-train.jsonl"),
                          num_proc=8)

    export_split_to_jsonl(dataset, "validation",
                          os.path.join(out_dir, "wikitext103-validation.jsonl"),
                          num_proc=4)

if __name__ == "__main__":
    main()