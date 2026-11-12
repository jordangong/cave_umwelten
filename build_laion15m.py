"""
Build the LAION-15M gallery of Koepke et al. (App. G.1.4, Table 6) from the
jp1924 LAION-400M parquets (images stored as lossless JPEG-XL recompressions of
the original JPEGs).

Recipe (paper): random 20M of LAION-400M -> image dedup (pHash, Hamming <= 2,
against the WIT-1024 queries and within the pool, first occurrence kept) ->
caption dedup (exact string, against WIT-1024 captions and within the pool,
first occurrence kept) -> random 15M of what is left.

Stages, one per call, each resumable (finished outputs are skipped):

  sample   20M rows uniformly from all jp1924 rows. The draw order is the
           "occurrence order" that decides which duplicate is kept.
           -> work/pool.parquet (order, file_idx, row)
  hash     (array) per source file: JXL -> original JPEG (bit-exact), pHash,
           and keep the pool rows with their JPEG bytes.
           -> work/pool/<split>-<file>.parquet
  dedup    image + caption dedup, then a random 15M; pos = rank in that draw.
           -> work/final.parquet (pos, order, file_idx, row), stats.json
  scatter  (array) per pool file: rows that made the 15M, split by pos bucket.
           -> work/buckets/bNNN/<split>-<file>.parquet
  pack     (array) per bucket: sort by pos, write 10k-row shards.
           -> laion_15m/train-NNNNN-of-01500.parquet
  finalize check all shards (row counts, pos = 0..N-1) and write the nested
           gallery indices: rows are in random order, so gallery n = rows [0, n).
           -> nested_indices/gallery_indices_<n>.npy

Layout under $PLATO_DATA_ROOT (default ~/datasets):

  jp1924--Laion400m-{1,2,3,4}/data/train-*.parquet   source
  askoepke--wit_1m_recaptioned/wit_1024/*.parquet    WIT-1024 queries
  jordangong--laion-15m/                             output (OUT)

Shard schema: pos, id (LAION SAMPLE_ID), caption, image (JPEG bytes, or PNG
for the rare non-JPEG source), img_w, img_h, width, height (jp1924 columns =
LAION metadata), license, nsfw, similarity, phash (64-bit pHash as int64),
src_file, src_row.
"""
import argparse
import glob
import io
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import paths

POOL_SIZE = 20_000_000
FINAL_SIZE = 15_000_000
SHARD_ROWS = 10_000
BUCKET_ROWS = 250_000          # pack works on one bucket (25 shards) at a time
HAMMING = 2
NESTED_SIZES = [10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000, 15_000_000]
SRC_COLS = ["id", "image", "caption", "height", "width", "license", "nsfw", "similarity"]
RG_CHUNK = 20                  # source row groups (100 rows each) read at a time


# ----------------------------------------------------------------- paths

def src_files():
    root = paths.data_root()
    files = glob.glob(os.path.join(root, "jp1924--Laion400m-[1-4]", "data", "train-*.parquet"))
    key = lambda f: (int(re.search(r"Laion400m-(\d)", f).group(1)), os.path.basename(f))
    files = sorted(files, key=key)
    if len(files) != 1773:
        raise RuntimeError(f"expected 1773 jp1924 parquets, found {len(files)}")
    return files


def src_tag(f):
    split = re.search(r"Laion400m-(\d)", f).group(1)
    num = re.search(r"train-(\d+)-of", f).group(1)
    return f"{split}-{num}"


def out_root(args):
    return os.path.expanduser(args.out) if args.out else os.path.join(paths.data_root(), "jordangong--laion-15m")


def wpath(args, *p):
    return os.path.join(out_root(args), "work", *p)


def write_atomic(table, path, **kw):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    pq.write_table(table, tmp, **kw)
    os.replace(tmp, path)


def my_items(n, args):
    """Items of this array task: round-robin over n."""
    task = args.task if args.task is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    ntasks = args.ntasks if args.ntasks is not None else int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    return list(range(task, n, ntasks))


def ncpu():
    return int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))


# ----------------------------------------------------------------- image ops

def _pil():
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None       # trusted dataset images, as in extract_features
    return Image


def phash_int(img):
    """imagehash.phash (grayscale, 32x32 Lanczos, 2D DCT, top-left 8x8 > median)
    packed MSB-first into a signed int64."""
    import imagehash
    bits = imagehash.phash(img).hash.flatten()
    return int.from_bytes(np.packbits(bits).tobytes(), "big", signed=True)


def jxl_to_image_bytes(jxl, decoder):
    """Return (bytes, fmt). JPEG-recompressed JXL gives back the original JPEG
    bit for bit; anything else is decoded and stored as PNG."""
    is_jpeg, info, data, icc, boxes = decoder(jxl)
    if is_jpeg:
        return data, "jpeg"
    Image = _pil()
    import pillow_jxl  # noqa: F401  (registers the JXL plugin)
    buf = io.BytesIO()
    Image.open(io.BytesIO(jxl)).save(buf, format="PNG")
    return buf.getvalue(), "png"


# ----------------------------------------------------------------- sample

def stage_sample(args):
    out = wpath(args, "pool.parquet")
    if os.path.exists(out):
        print(f"{out} exists"); return
    files = src_files()
    nrows = np.array([pq.ParquetFile(f).metadata.num_rows for f in files], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(nrows)])
    total = int(offsets[-1])
    print(f"{len(files)} files, {total:,} rows")
    rng = np.random.default_rng(args.seed)
    gidx = rng.choice(total, size=POOL_SIZE, replace=False)       # random order = occurrence order
    file_idx = np.searchsorted(offsets, gidx, side="right") - 1
    row = gidx - offsets[file_idx]
    t = pa.table({
        "order": pa.array(np.arange(POOL_SIZE, dtype=np.int32)),
        "file_idx": pa.array(file_idx.astype(np.int16)),
        "row": pa.array(row.astype(np.int32)),
    })
    write_atomic(t, out)
    meta = {"seed": args.seed, "total_rows": total, "pool_size": POOL_SIZE,
            "files": [os.path.relpath(f, paths.data_root()) for f in files], "rows_per_file": nrows.tolist()}
    with open(wpath(args, "sources.json"), "w") as fh:
        json.dump(meta, fh)
    print(f"wrote {out}")


# ----------------------------------------------------------------- hash

POOL_SCHEMA = pa.schema([
    ("order", pa.int32()), ("file_idx", pa.int16()), ("row", pa.int32()),
    ("id", pa.int64()), ("caption", pa.string()),
    ("image", pa.binary()), ("fmt", pa.string()), ("img_w", pa.int32()), ("img_h", pa.int32()),
    ("width", pa.int32()), ("height", pa.int32()),
    ("license", pa.string()), ("nsfw", pa.string()), ("similarity", pa.float32()),
    ("phash", pa.int64()), ("err", pa.string()),
])


def hash_one_file(job):
    """Worker: one source file -> one pool parquet."""
    fi, path, rows, orders, out = job
    if os.path.exists(out):
        return fi, len(rows), 0, "skip"
    Image = _pil()
    from pillow_jxl import Decoder
    dec = Decoder(num_threads=0)
    pf = pq.ParquetFile(path)
    rg_rows = np.array([pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)])
    rg_start = np.concatenate([[0], np.cumsum(rg_rows)])
    tmp = out + ".tmp"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    writer = pq.ParquetWriter(tmp, POOL_SCHEMA)
    nerr = 0
    for g0 in range(0, len(rg_rows), RG_CHUNK):
        g1 = min(g0 + RG_CHUNK, len(rg_rows))
        lo, hi = np.searchsorted(rows, [rg_start[g0], rg_start[g1]])
        if lo == hi:
            continue
        t = pf.read_row_groups(list(range(g0, g1)), columns=SRC_COLS)
        sub = t.take(pa.array(rows[lo:hi] - rg_start[g0]))
        cols = {c: sub.column(c).to_pylist() for c in SRC_COLS if c != "image"}
        imgs = pc.struct_field(sub.column("image"), "bytes").to_pylist()
        rec = {k: [] for k in ("image", "fmt", "img_w", "img_h", "phash", "err")}
        for b in imgs:
            try:
                data, fmt = jxl_to_image_bytes(b, dec)
                im = Image.open(io.BytesIO(data))
                im.load()
                rec["image"].append(data); rec["fmt"].append(fmt)
                rec["img_w"].append(im.size[0]); rec["img_h"].append(im.size[1])
                rec["phash"].append(phash_int(im)); rec["err"].append(None)
            except Exception as e:  # noqa: BLE001  (record and keep going)
                nerr += 1
                for k in ("image", "fmt", "img_w", "img_h", "phash"):
                    rec[k].append(None)
                rec["err"].append(repr(e)[:200])
        batch = pa.table({
            "order": pa.array(orders[lo:hi], pa.int32()),
            "file_idx": pa.array(np.full(hi - lo, fi), pa.int16()),
            "row": pa.array(rows[lo:hi], pa.int32()),
            "id": cols["id"], "caption": cols["caption"],
            **rec,
            "width": cols["width"], "height": cols["height"], "license": cols["license"],
            "nsfw": cols["nsfw"], "similarity": cols["similarity"],
        }).select(POOL_SCHEMA.names).cast(POOL_SCHEMA)
        writer.write_table(batch, row_group_size=1000)
    writer.close()
    os.replace(tmp, out)
    return fi, len(rows), nerr, "done"


def stage_hash(args):
    from concurrent.futures import ProcessPoolExecutor
    files = src_files()
    pool = pq.read_table(wpath(args, "pool.parquet"))
    fidx = pool.column("file_idx").to_numpy()
    rows = pool.column("row").to_numpy()
    orders = pool.column("order").to_numpy()
    by_file = np.argsort(fidx, kind="stable")
    bounds = np.searchsorted(fidx[by_file], np.arange(len(files) + 1))
    jobs = []
    for fi in my_items(len(files), args):
        sel = by_file[bounds[fi]:bounds[fi + 1]]
        sel = sel[np.argsort(rows[sel])]
        out = wpath(args, "pool", src_tag(files[fi]) + ".parquet")
        jobs.append((fi, files[fi], rows[sel], orders[sel], out))
    print(f"{len(jobs)} files for this task, {ncpu()} workers")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=ncpu()) as ex:
        for i, (fi, n, nerr, st) in enumerate(ex.map(hash_one_file, jobs)):
            print(f"[{i + 1}/{len(jobs)}] {src_tag(files[fi])}: {n} rows, {nerr} errors, {st} "
                  f"({time.time() - t0:.0f}s)", flush=True)


# ----------------------------------------------------------------- dedup

def bitcount64(x):
    return np.bitwise_count(x.view(np.uint64))


def near_pairs(u, maxd=HAMMING):
    """All index pairs (a, b), a != b, of distinct 64-bit hashes `u` with
    Hamming distance <= maxd. Pigeonhole: with maxd + 1 blocks, one block is
    identical, so candidates are pairs sharing a block value. Within each
    block's sort order, compare element j with j + o for o = 1, 2, ...; after
    each step keep only groups larger than o, so the work is sum(m^2) over
    groups, vectorized."""
    uu = u.view(np.uint64)
    nb = maxd + 1
    widths = [64 // nb + (1 if i < 64 % nb else 0) for i in range(nb)]
    shifts = np.cumsum([0] + widths[:-1])
    pa_, pb_ = [], []
    for w, s in zip(widths, shifts):
        key = (uu >> np.uint64(s)) & np.uint64((1 << w) - 1)
        idx = np.argsort(key, kind="stable")
        k = key[idx]
        o = 1
        while len(k) > o:
            same = k[o:] == k[:-o]
            if not same.any():
                break
            a, b = idx[:-o][same], idx[o:][same]
            d = bitcount64(u[a] ^ u[b])
            m = d <= maxd
            pa_.append(a[m]); pb_.append(b[m])
            o += 1
            if o in (8, 64, 512, 4096, 32768):       # drop groups that are done
                starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
                sizes = np.diff(np.r_[starts, len(k)])
                keep = np.repeat(sizes > o, sizes)
                k, idx = k[keep], idx[keep]
        print(f"  block shift {s} width {w}: {sum(len(x) for x in pa_):,} pairs so far", flush=True)
    a = np.concatenate(pa_) if pa_ else np.zeros(0, np.int64)
    b = np.concatenate(pb_) if pb_ else np.zeros(0, np.int64)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    pairs = np.unique(np.stack([lo, hi], 1), axis=0) if len(lo) else np.zeros((0, 2), np.int64)
    return pairs


def image_dedup_keep(h):
    """Keep mask over hashes `h` (in occurrence order): an image is dropped if
    an earlier *kept* image is within Hamming <= HAMMING (greedy keep-first).
    Exact repeats of a hash always go; near pairs are found among distinct
    hashes and resolved in order of each hash's first occurrence."""
    uniq, first, inv = np.unique(h, return_index=True, return_inverse=True)
    print(f"{len(h):,} images, {len(uniq):,} distinct pHashes; finding pairs within Hamming {HAMMING}")
    pairs = near_pairs(uniq)
    print(f"{len(pairs):,} near pairs among distinct hashes")
    fa, fb = first[pairs[:, 0]], first[pairs[:, 1]]
    early = np.where(fa < fb, pairs[:, 0], pairs[:, 1])
    late = np.where(fa < fb, pairs[:, 1], pairs[:, 0])
    o = np.argsort(first[late], kind="stable")
    early, late = early[o], late[o]
    removed = np.zeros(len(uniq), bool)
    for e, l in zip(early.tolist(), late.tolist()):
        if not removed[e]:          # final: every edge into e sorts before this one
            removed[l] = True
    removed_any = np.zeros(len(uniq), bool)      # alt. reading: any earlier near dup, kept or not
    removed_any[late] = True
    is_first = np.zeros(len(h), bool)
    is_first[first] = True
    keep = is_first & ~removed[inv]
    info = {
        "distinct_phash": int(len(uniq)),
        "near_pairs": int(len(pairs)),
        "img_dup_within_exact": int((~is_first).sum()),
        "img_dup_within_near": int((is_first & removed[inv]).sum()),
        "img_dup_within": int((~keep).sum()),
        "img_dup_within_if_drop_any_earlier": int((~(is_first & ~removed_any[inv])).sum()),
    }
    return keep, info


def wit1024_queries():
    files = paths.local_parquets("askoepke/wit_1m_recaptioned", "wit_1024")
    t = pa.concat_tables([pq.read_table(f, columns=["image", "original_caption"]) for f in files])
    Image = _pil()
    hashes = []
    for b in t.column("image").to_pylist():
        b = b["bytes"] if isinstance(b, dict) else b
        hashes.append(phash_int(Image.open(io.BytesIO(b))))
    return np.array(hashes, dtype=np.int64), t.column("original_caption").to_pylist()


def stage_dedup(args):
    out = wpath(args, "final.parquet")
    if os.path.exists(out):
        print(f"{out} exists"); return
    files = sorted(glob.glob(wpath(args, "pool", "*.parquet")))
    if len(files) != 1773:
        raise RuntimeError(f"hash stage incomplete: {len(files)}/1773 pool files")
    cols = ["order", "file_idx", "row", "caption", "phash", "err"]
    t = pa.concat_tables([pq.read_table(f, columns=cols) for f in files])
    t = t.sort_by("order")
    if t.num_rows != POOL_SIZE or not np.array_equal(t.column("order").to_numpy(), np.arange(POOL_SIZE)):
        raise RuntimeError("pool files don't cover order 0..POOL_SIZE-1 exactly once")
    st = {"raw_pool": POOL_SIZE}

    ok = pc.is_valid(t.column("phash")).to_numpy(zero_copy_only=False)
    st["decode_failed"] = int((~ok).sum())
    errs = pc.value_counts(pc.drop_null(t.column("err"))).to_pylist()
    st["decode_errors_top"] = sorted(([e["values"], e["counts"]] for e in errs), key=lambda x: -x[1])[:10]
    t = t.filter(pa.array(ok))
    h = t.column("phash").to_numpy()
    print(f"pool {POOL_SIZE:,}, decode failures {st['decode_failed']:,}")

    # image dedup vs WIT-1024
    qh, qcap = wit1024_queries()
    hit = np.zeros(len(h), bool)
    for i in range(0, len(h), 20_000):
        x = h[i:i + 20_000, None] ^ qh[None, :]
        hit[i:i + 20_000] = (bitcount64(x) <= HAMMING).any(1)
    st["img_dup_wit1024"] = int(hit.sum())
    t, h = t.filter(pa.array(~hit)), h[~hit]
    print(f"image dups with WIT-1024: {st['img_dup_wit1024']:,}")

    # image dedup within the pool, first occurrence kept (t is in occurrence order)
    keep, info = image_dedup_keep(h)
    st.update(info)
    t = t.filter(pa.array(keep))
    st["after_image_dedup"] = t.num_rows
    print(f"image dups within pool: {st['img_dup_within']:,} -> {t.num_rows:,}")

    # caption dedup vs WIT-1024 (exact string)
    capq = pc.is_in(t.column("caption"), value_set=pa.array(qcap))
    capq = pc.fill_null(capq, False).to_numpy(zero_copy_only=False)
    st["cap_dup_wit1024"] = int(capq.sum())
    t = t.filter(pa.array(~capq))
    nullcap = pc.is_null(t.column("caption")).to_numpy(zero_copy_only=False)
    st["cap_null"] = int(nullcap.sum())
    t = t.filter(pa.array(~nullcap))
    # within the pool, first occurrence kept (t is still in occurrence order)
    g = t.group_by("caption", use_threads=False).aggregate([("order", "min")])
    firsts = pc.is_in(t.column("order"), value_set=g.column("order_min")).to_numpy(zero_copy_only=False)
    st["cap_dup_within"] = int((~firsts).sum())
    t = t.filter(pa.array(firsts))
    st["final_pool"] = t.num_rows
    print(f"caption dups: WIT-1024 {st['cap_dup_wit1024']:,}, null {st['cap_null']:,}, "
          f"within {st['cap_dup_within']:,} -> final pool {t.num_rows:,}")
    if t.num_rows < FINAL_SIZE:
        raise RuntimeError(f"final pool {t.num_rows:,} < {FINAL_SIZE:,}")

    # random 15M; pos = rank in the draw, so any prefix is a uniform sample
    rng = np.random.default_rng(args.seed + 1)
    pick = rng.choice(t.num_rows, size=FINAL_SIZE, replace=False)
    f = t.take(pa.array(pick)).select(["order", "file_idx", "row"])
    f = f.append_column("pos", pa.array(np.arange(FINAL_SIZE, dtype=np.int32)))
    write_atomic(f, out)
    st["final"] = FINAL_SIZE
    st["seed"] = args.seed
    st["commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                  text=True, cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    with open(os.path.join(out_root(args), "stats.json"), "w") as fh:
        json.dump(st, fh, indent=1)
    print(json.dumps(st, indent=1))


# ----------------------------------------------------------------- scatter / pack

def stage_scatter(args):
    files = sorted(glob.glob(wpath(args, "pool", "*.parquet")))
    final = pq.read_table(wpath(args, "final.parquet"))
    order_to_pos = np.full(POOL_SIZE, -1, np.int64)
    order_to_pos[final.column("order").to_numpy()] = final.column("pos").to_numpy()
    nb = -(-FINAL_SIZE // BUCKET_ROWS)
    for i in my_items(len(files), args):
        tag = os.path.basename(files[i])[:-len(".parquet")]
        done = wpath(args, "buckets", f".done-{tag}")
        if os.path.exists(done):
            continue
        t = pq.read_table(files[i])
        pos = order_to_pos[t.column("order").to_numpy()]
        sel = pos >= 0
        t = t.filter(pa.array(sel))
        pos = pos[sel]
        t = t.append_column("pos", pa.array(pos.astype(np.int32)))
        b = pos // BUCKET_ROWS
        for bi in np.unique(b):
            write_atomic(t.filter(pa.array(b == bi)), wpath(args, "buckets", f"b{bi:03d}", tag + ".parquet"))
        open(done, "w").close()
        print(f"{tag}: {len(pos)} rows into {len(np.unique(b))} of {nb} buckets", flush=True)


SHARD_COLS = ["pos", "id", "caption", "image", "img_w", "img_h", "width", "height",
              "license", "nsfw", "similarity", "phash", "src_file", "src_row"]


def stage_pack(args):
    nb = -(-FINAL_SIZE // BUCKET_ROWS)
    nshards = FINAL_SIZE // SHARD_ROWS
    names = json.load(open(wpath(args, "sources.json")))["files"]
    out_dir = os.path.join(out_root(args), "laion_15m")
    for bi in my_items(nb, args):
        s0, s1 = bi * BUCKET_ROWS // SHARD_ROWS, min((bi + 1) * BUCKET_ROWS, FINAL_SIZE) // SHARD_ROWS
        outs = [os.path.join(out_dir, f"train-{s:05d}-of-{nshards:05d}.parquet") for s in range(s0, s1)]
        if all(os.path.exists(o) for o in outs):
            continue
        parts = sorted(glob.glob(wpath(args, "buckets", f"b{bi:03d}", "*.parquet")))
        t = pa.concat_tables([pq.read_table(p) for p in parts])
        # a bucket's images are ~15 GB: sorting concatenates the column, which
        # overflows binary's 32-bit offsets, so sort as large_binary
        t = t.set_column(t.schema.get_field_index("image"), "image",
                         t.column("image").cast(pa.large_binary())).sort_by("pos")
        pos = t.column("pos").to_numpy()
        lo = bi * BUCKET_ROWS
        if not np.array_equal(pos, np.arange(lo, lo + len(outs) * SHARD_ROWS)):
            raise RuntimeError(f"bucket {bi}: pos not contiguous ({len(pos)} rows from {len(parts)} parts)")
        if pc.any(pc.is_null(t.column("image"))).as_py():
            raise RuntimeError(f"bucket {bi}: null image in final rows")
        t = t.append_column("src_file", pa.array([names[i] for i in t.column("file_idx").to_numpy()]))
        t = t.append_column("src_row", t.column("row")).select(SHARD_COLS)
        for j, o in enumerate(outs):
            if not os.path.exists(o):
                # take (not slice): a slice keeps large_binary's absolute 64-bit
                # offsets, which don't fit binary past the first ~2 GB
                s = t.take(pa.array(np.arange(j * SHARD_ROWS, (j + 1) * SHARD_ROWS)))
                s = s.set_column(s.schema.get_field_index("image"), "image", s.column("image").cast(pa.binary()))
                write_atomic(s, o, row_group_size=1000)
        print(f"bucket {bi}: shards {s0}..{s1 - 1}", flush=True)


def stage_finalize(args):
    out_dir = os.path.join(out_root(args), "laion_15m")
    shards = sorted(glob.glob(os.path.join(out_dir, "train-*.parquet")))
    nshards = FINAL_SIZE // SHARD_ROWS
    if len(shards) != nshards:
        raise RuntimeError(f"{len(shards)}/{nshards} shards")
    for s, f in enumerate(shards):
        m = pq.ParquetFile(f).metadata
        stats = m.row_group(0).column(0).statistics
        if m.num_rows != SHARD_ROWS or stats.min != s * SHARD_ROWS:
            raise RuntimeError(f"{f}: {m.num_rows} rows, first pos {stats.min}")
        last = m.row_group(m.num_row_groups - 1).column(0).statistics.max
        if last != (s + 1) * SHARD_ROWS - 1:
            raise RuntimeError(f"{f}: last pos {last}")
    idx_dir = os.path.join(out_root(args), "nested_indices")
    os.makedirs(idx_dir, exist_ok=True)
    for n in NESTED_SIZES:
        np.save(os.path.join(idx_dir, f"gallery_indices_{n}.npy"), np.arange(n, dtype=np.int64))
    print(f"{len(shards)} shards ok; nested indices for {NESTED_SIZES} in {idx_dir}")


# ----------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["sample", "hash", "dedup", "scatter", "pack", "finalize"])
    p.add_argument("--out", default=None, help="output root (default $PLATO_DATA_ROOT/jordangong--laion-15m)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--task", type=int, default=None, help="array task id (default $SLURM_ARRAY_TASK_ID)")
    p.add_argument("--ntasks", type=int, default=None, help="array size (default $SLURM_ARRAY_TASK_COUNT)")
    args = p.parse_args()
    globals()[f"stage_{args.stage}"](args)


if __name__ == "__main__":
    sys.exit(main())
