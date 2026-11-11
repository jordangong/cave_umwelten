"""
Resolve datasets and checkpoints to local pools, so nothing streams from the
HF Hub when a local copy exists. Same env vars as platonic-rep's paths.py:

    PLATO_DATA_ROOT (default ~/datasets)
        <root>/askoepke--wit_1m_recaptioned/<config>/train-*.parquet
    PLATO_CKPT_ROOT (default ~/checkpoints)
        <root>/gen/<modality-tag>/<Vendor>/[<Family>/]<model>/   (HF causal LMs)
        <root>/timm/<timm-model-name>/model.safetensors        (timm models)

Anything not found locally falls back to the Hub, as upstream does.
"""
import glob
import os

TIMM_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def data_root():
    return os.path.expanduser(os.environ.get("PLATO_DATA_ROOT", "~/datasets"))


def ckpt_root():
    return os.path.expanduser(os.environ.get("PLATO_CKPT_ROOT", "~/checkpoints"))


def local_parquets(dataset_id, config):
    """Sorted parquet files of a local copy of `dataset_id`/`config`, or None.
    Sorted order is the order HF `load_dataset` iterates them in, so global row
    indices (e.g. gallery_indices_*.npy) mean the same thing either way."""
    d = os.path.join(data_root(), dataset_id.replace("/", "--"), config)
    files = sorted(glob.glob(os.path.join(d, "*.parquet")))
    return files or None


def resolve_llm(model_name):
    """HF id -> local model dir if found in the pool, else the id."""
    if os.path.isdir(model_name):
        return model_name
    base = model_name.split("/")[-1]
    root = ckpt_root()
    hits = []
    for pattern in (f"{root}/gen/*/*/*/{base}", f"{root}/gen/*/*/{base}"):
        hits += [d for d in glob.glob(pattern) if os.path.isfile(os.path.join(d, "config.json"))]
    if len(hits) > 1:
        txt_hits = [d for d in hits if f"{root}/gen/txt/" in d]
        hits = txt_hits if len(txt_hits) == 1 else hits
    if len(hits) > 1:
        raise RuntimeError(f"ambiguous local checkpoints for {model_name}: {hits}")
    if hits:
        print(f"resolved {model_name} -> {hits[0]}")
        return hits[0]
    print(f"{model_name} not in {root}, falling back to HF hub")
    return model_name


def resolve_timm(model_name):
    """timm name -> local weight file if found in the pool, else None."""
    for fn in TIMM_WEIGHT_FILES:
        path = os.path.join(ckpt_root(), "timm", model_name, fn)
        if os.path.isfile(path):
            print(f"resolved {model_name} -> {path}")
            return path
    print(f"{model_name} not in {ckpt_root()}/timm, falling back to HF hub")
    return None
