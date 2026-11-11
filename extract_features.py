"""
Extract DINO or OpenLLaMA features on WIT-1024 / WIT-1M from HuggingFace.

Features are saved at all transformer layers so that downstream alignment can
choose any layer pair at runtime.

Dataset: askoepke/wit_1m_recaptioned  (configs: wit_1024, wit_1m)

Output format (matches the format consumed by align_nested_wit1m.py):
  {
    "feats": torch.float32 tensor of shape (N, num_layers, feat_dim),
    "num_params": int,
    "layer_indices": list[int]  (optional),
    "shard_id": int             (only for sharded wit_1m),
    "num_shards": int           (only for sharded wit_1m),
  }

Vision tap points match cave_final: `blocks.{i}.add_1` (post-residual-add in
each ViT block), CLS token.

Language pooling matches cave_final: avg-pooled over non-pad positions, with
hidden states from all layers (embedding layer included).  Tokenization uses
left-padding "longest" over the *entire* per-shard text set at once, matching
cave_final's batching.

Usage:
    # DINO (vision) on WIT-1024
    python extract_features.py --modality vision --config wit_1024 \
        --output_dir results/features/wit_1024

    # OpenLLaMA (language) on WIT-1M shard 0
    python extract_features.py --modality language --config wit_1m \
        --shard_id 0 --num_shards 100 \
        --output_dir results/features/wit_1m/shards
"""
import argparse
import io
import os

import torch
from tqdm import tqdm, trange

import paths

DATASET_ID = "askoepke/wit_1m_recaptioned"
DEFAULT_VISION_MODEL = "vit_base_patch14_dinov2.lvd142m"
DEFAULT_LANGUAGE_MODEL = "openlm-research/open_llama_3b"
DEFAULT_NUM_SHARDS_WIT1M = 100


def _safe_name(name):
    return name.replace("/", "_")


def check_bfloat16_support():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 7


# --------------------------------------------------------------------- dataset

def _row_to_image(img_field):
    from PIL import Image
    if isinstance(img_field, Image.Image):
        img = img_field
    elif isinstance(img_field, dict):
        img = Image.open(io.BytesIO(img_field["bytes"]))
    else:
        img = Image.open(io.BytesIO(img_field))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _load_rows_local(files, start, end, column):
    """Rows [start, end) of `column` from a local parquet copy (files in HF
    iteration order), reading only the files and the column that are needed."""
    import pyarrow.parquet as pq
    out, offset = [], 0
    for f in files:
        n = pq.ParquetFile(f).metadata.num_rows
        lo, hi = max(start, offset), min(end, offset + n)
        if lo < hi:
            col = pq.read_table(f, columns=[column]).column(column)
            out.extend(col.slice(lo - offset, hi - lo).to_pylist())
        offset += n
        if offset >= end:
            break
    if len(out) != end - start:
        raise RuntimeError(f"wanted rows [{start}, {end}), got {len(out)} from {len(files)} files")
    return out


def load_rows(config, start, end, modality, caption):
    """Load rows [start, end). Vision rows come back as raw image fields and are
    decoded per batch in extract_vision (decoding a whole shard up front costs
    tens of GB of RAM). Reads $PLATO_DATA_ROOT/askoepke--wit_1m_recaptioned if
    present, else streams from the HF Hub so a small slice doesn't pull down
    entire parquet shards."""
    field = "image" if modality == "vision" else (
        "gemini_caption" if caption == "gemini" else "original_caption")
    files = paths.local_parquets(DATASET_ID, config)
    if files is not None:
        print(f"Reading {field} from {len(files)} local parquet files in {os.path.dirname(files[0])}")
        return _load_rows_local(files, start, end, field)

    from datasets import load_dataset
    ds = load_dataset(
        DATASET_ID, config, split="train", streaming=True,
        cache_dir="data/wit1m/shards",
    )
    # Skip `start` rows, take (end - start) rows.
    if start > 0:
        ds = ds.skip(start)
    ds = ds.take(end - start)
    return [row[field] for row in ds]


# Known lengths so we don't need to materialize the dataset just to count rows.
_KNOWN_LENGTHS = {"wit_1024": 1024, "wit_1m": 1_000_000}


def get_dataset_length(config):
    if config in _KNOWN_LENGTHS:
        return _KNOWN_LENGTHS[config]
    files = paths.local_parquets(DATASET_ID, config)
    if files is not None:
        import pyarrow.parquet as pq
        return sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    from datasets import load_dataset
    return len(load_dataset(
        DATASET_ID, config, split="train", cache_dir="data/wit1m/shards",
    ))


# --------------------------------------------------------------------- vision

class _ImageDataset(torch.utils.data.Dataset):
    """Decode + transform one image at a time (in DataLoader workers)."""

    def __init__(self, fields, transform):
        self.fields, self.transform = fields, transform

    def __len__(self):
        return len(self.fields)

    def __getitem__(self, i):
        return self.transform(_row_to_image(self.fields[i]))


def extract_vision(images, model_name, batch_size, layer_indices, num_workers=8):
    """Match cave_final's extract_dino.py exactly: `blocks.{i}.add_1`
    hook points, CLS token, float32 output, all blocks by default.
    `images` are raw image fields (bytes or PIL); they are decoded and
    transformed in DataLoader workers, in order."""
    import timm
    from timm.data import resolve_data_config, create_transform
    from torchvision.models.feature_extraction import create_feature_extractor

    print(f"\nmodel:       \t{model_name}")
    print(f"num_samples: \t{len(images)}")
    local = paths.resolve_timm(model_name)
    overlay = dict(file=local) if local else None
    vision_model = timm.create_model(
        model_name, pretrained=True, pretrained_cfg_overlay=overlay,
    ).cuda().eval()
    num_params = sum(p.numel() for p in vision_model.parameters())
    print(f"num_params:  \t{num_params:,}")

    transform = create_transform(
        **resolve_data_config(vision_model.pretrained_cfg, model=vision_model)
    )
    print(transform)

    num_blocks = len(vision_model.blocks)
    block_ids = layer_indices if layer_indices is not None else list(range(num_blocks))
    print(f"Extracting layers: {block_ids} (of {num_blocks} total)")

    return_nodes = [f"blocks.{i}.add_1" for i in block_ids]
    vision_model = create_feature_extractor(vision_model, return_nodes=return_nodes)

    loader = torch.utils.data.DataLoader(
        _ImageDataset(images, transform), batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    all_feats = []
    for xs in tqdm(loader, desc=model_name):
        xs = xs.cuda(non_blocking=True)
        with torch.no_grad():
            out = vision_model(xs)
            feats = [v[:, 0, :] for v in out.values()]   # CLS token
            feats = torch.stack(feats).permute(1, 0, 2)  # [B, L, D]
            all_feats.append(feats.cpu())

    all_feats = torch.cat(all_feats, dim=0)
    print(f"Features shape: {all_feats.shape}")
    return all_feats, num_params, block_ids


# ------------------------------------------------------------------- language

def load_llm(model_name, max_gpu_memory=None):
    """Match cave_final's models.load_llm: bf16 where supported, fp32 fallback.
    `max_gpu_memory` (e.g. "9GiB") caps the weights accelerate puts on each GPU
    and offloads the rest to CPU; only needed on small pre-Ampere GPUs, where
    the fp32 fallback doesn't fit."""
    from transformers import AutoModelForCausalLM
    dtype = torch.bfloat16 if check_bfloat16_support() else torch.float32
    print(f"torch_dtype:\t{dtype}")
    kwargs = {}
    if max_gpu_memory:
        kwargs["max_memory"] = {i: max_gpu_memory for i in range(torch.cuda.device_count())}
        kwargs["max_memory"]["cpu"] = "512GiB"
    model = AutoModelForCausalLM.from_pretrained(
        paths.resolve_llm(model_name),
        device_map="auto",
        torch_dtype=dtype,
        output_hidden_states=True,
        **kwargs,
    ).eval()
    return model


def load_tokenizer(model_name):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(paths.resolve_llm(model_name))
    if "huggyllama" in model_name:
        tok.pad_token = "[PAD]"
    elif tok.pad_token is None:
        tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    return tok


def extract_language(texts, model_name, batch_size, max_length, pool, layer_indices,
                     max_gpu_memory=None):
    """Match cave_final's extract_llm.py: tokenize the entire text set at once
    with padding='longest', left-pad, avg-pool over attention mask, all layers."""
    print(f"\nmodel:       \t{model_name}")
    print(f"num_samples: \t{len(texts)}")
    model = load_llm(model_name, max_gpu_memory)
    num_params = sum(p.numel() for p in model.parameters())
    tok = load_tokenizer(model_name)
    print(f"num_params:  \t{num_params:,}")

    tokens = tok(
        texts, padding="longest", truncation=True,
        max_length=max_length, return_tensors="pt",
    )
    print(f"Token shape: {tokens['input_ids'].shape}")

    device = next(model.parameters()).device
    all_feats = []

    for i in trange(0, len(texts), batch_size, desc=model_name):
        tok_in = {
            k: v[i:i + batch_size].to(device).long()
            for k, v in tokens.items()
        }
        with torch.no_grad():
            out = model(
                input_ids=tok_in["input_ids"],
                attention_mask=tok_in["attention_mask"],
            )
            if pool == "avg":
                feats = torch.stack(out["hidden_states"]).permute(1, 0, 2, 3)  # (B, L, T, D)
                mask = tok_in["attention_mask"].unsqueeze(-1).unsqueeze(1)      # (B, 1, T, 1)
                feats = (feats * mask).sum(2) / mask.sum(2)
            else:  # 'last'
                feats = [v[:, -1, :] for v in out["hidden_states"]]
                feats = torch.stack(feats).permute(1, 0, 2)

            if layer_indices is not None:
                feats = feats[:, layer_indices, :]
            # Keep compute dtype (bf16 on A100) to match cave_final's save format.
            all_feats.append(feats.cpu())

    all_feats = torch.cat(all_feats, dim=0)
    print(f"Features shape: {all_feats.shape}")
    return all_feats, num_params, tokens["attention_mask"].cpu()


# ------------------------------------------------------------------- pipeline

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modality", choices=["vision", "language"], required=True)
    p.add_argument("--config", choices=["wit_1024", "wit_1m"], required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model", default=None,
                   help="Model name. Defaults: "
                        f"vision={DEFAULT_VISION_MODEL}, language={DEFAULT_LANGUAGE_MODEL}")
    p.add_argument("--caption", choices=["original", "gemini"], default="original")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Default: 64 for vision, 4 for language (matches cave_final)")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--pool", choices=["avg", "last"], default="avg")
    p.add_argument("--layer_indices", nargs="*", type=int, default=None)
    p.add_argument("--max_gpu_memory", default=None,
                   help='Cap LLM weights per GPU, e.g. "9GiB" (offload rest to CPU)')
    p.add_argument("--num_workers", type=int, default=8,
                   help="DataLoader workers for image decoding (vision only)")
    # sharding (wit_1m)
    p.add_argument("--shard_id", type=int, default=None)
    p.add_argument("--num_shards", type=int, default=DEFAULT_NUM_SHARDS_WIT1M)
    # testing
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N rows (for quick tests).")
    p.add_argument("--force_remake", action="store_true")
    args = p.parse_args()

    if args.model is None:
        args.model = DEFAULT_VISION_MODEL if args.modality == "vision" else DEFAULT_LANGUAGE_MODEL
    if args.batch_size is None:
        args.batch_size = 64 if args.modality == "vision" else 4

    # Figure out row range
    if args.config == "wit_1024" or args.shard_id is None:
        total = get_dataset_length(args.config)
        start, end = 0, total
    else:
        total = get_dataset_length(args.config)
        per_shard = total // args.num_shards
        start = args.shard_id * per_shard
        end = total if args.shard_id == args.num_shards - 1 else start + per_shard
        print(f"Shard {args.shard_id}: samples [{start}, {end})")

    if args.limit is not None:
        end = min(end, start + args.limit)
        print(f"--limit {args.limit} active: using samples [{start}, {end})")

    # Resolve save path. Gemini-caption language features get a "_gemini" tag
    # so they don't overwrite the original-caption ones (upstream computed the
    # tag but never used it).
    pool_tag = "pool-cls" if args.modality == "vision" else f"pool-{args.pool}"
    model_dir_name = _safe_name(args.model)
    cap_tag = "_gemini" if (args.modality == "language" and args.caption == "gemini") else ""

    if args.config == "wit_1024" or args.shard_id is None:
        save_path = os.path.join(args.output_dir, f"{model_dir_name}_{pool_tag}{cap_tag}.pt")
    else:
        save_path = os.path.join(
            args.output_dir, f"{model_dir_name}{cap_tag}", f"shard_{args.shard_id:04d}.pt"
        )

    if os.path.exists(save_path) and not args.force_remake:
        print(f"file exists: {save_path}. skipping")
        return

    # Load rows
    print(f"Loading {args.modality} rows from {DATASET_ID}/{args.config} [{start}:{end})...")
    rows = load_rows(args.config, start, end, args.modality, args.caption)

    # Extract
    if args.modality == "vision":
        feats, num_params, block_ids = extract_vision(
            rows, args.model, args.batch_size, args.layer_indices, args.num_workers,
        )
        save_dict = {"feats": feats, "num_params": num_params}
        if args.layer_indices is not None:
            save_dict["layer_indices"] = args.layer_indices
    else:
        feats, num_params, mask = extract_language(
            rows, args.model, args.batch_size, args.max_length,
            args.pool, args.layer_indices, args.max_gpu_memory,
        )
        save_dict = {
            "feats": feats,
            "num_params": num_params,
            "mask": mask,
        }
        if args.layer_indices is not None:
            save_dict["layer_indices"] = args.layer_indices

    if args.shard_id is not None and args.config == "wit_1m":
        save_dict["shard_id"] = args.shard_id
        save_dict["num_shards"] = args.num_shards

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(save_dict, save_path)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
