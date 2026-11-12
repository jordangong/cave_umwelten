"""
Build the two 100k galleries of Koepke et al. Sec. 3.2.2 / Fig. 6 (text-audio,
text-video). The paper gives only "a 100k subset of LAION-Audio-630k" and
"PVD-100k, a 100k subset of PVD"; everything else here is our choice
(docs/wit.md, "cave_umwelten: text-audio and text-video"):

  laion_audio  LAION-Audio-630k from its public parts: Meranti/CLAP_freesound
               (Freesound full, webdataset tars) + the CLAPv2 parquet repos of
               BBC, Epidemic, Audiostock, Sonniss, WeSound, Paramount. Free To
               Use Sounds was never released. One clip per source clip
               (CLAPv2 duplicates dropped), caption = the clip's first caption
               (Freesound title, Epidemic title, else CLAPv2 `text`), empty
               captions dropped, exact caption duplicates dropped (first
               occurrence in a seeded random order, as App. G.1.2 does for
               WIT/LAION), then a fresh seeded order of what is left and its
               first 100k clips that decode.
               Audio is stored decoded: mono, 16 kHz, first 30 s, FLAC.
  pvd          facebook/PE-Video human-captioned clips (train + test, 118,862),
               caption = human_caption, empty dropped, exact duplicates dropped,
               first 100k of a fresh seeded order that decode. Videos stay in
               the source tars; the manifest holds (tar, offset, size) and the
               decoded frame count.

Rows of both are in that random order, so the first 1000 rows are the fixed
query set Q and the gallery of total size n (Q included, App. C) is rows
[0, n). Gallery ids in nested_indices/ follow knn_sets.py's convention for
queries kept apart from the gallery shards: gallery row g = pos 1000 + g, and
gallery_indices_<n>.npy = arange(n - 1000) (knn_sets --size_includes_queries).

Stages (one per call, each resumable; array stages split items round-robin over
$SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT):

  laion_audio:
    index     (array) Freesound tars -> work/fs_index/<dir>-<tar>.parquet
              (member offsets + json), and (task 0) CLAPv2 parquet metadata
              -> work/clap_index.parquet
    pool      dedup + caption choice + seeded order -> work/pool.parquet, stats
    pack      (array) decode/resample the first CANDIDATES pool clips:
              Freesound in order-chunks, CLAPv2 grouped by source parquet
              -> work/packed/{oNNNN,pNNNN}.parquet
    finalize  first 100k decoded clips -> laion_audio_100k/train-000NN-of-00020.parquet,
              nested_indices/, stats.json
  pvd:
    index     (array) PE-Video train/test tars -> work/index/<split>-<tar>.parquet
    pool      -> work/pool.parquet, stats
    pack      (array) full PyAV decode of the first CANDIDATES clips (frame count)
              -> work/probed/oNNNN.parquet
    finalize  -> pvd_100k/manifest-*.parquet, nested_indices/, stats.json

Layout under $PLATO_DATA_ROOT (default ~/datasets): sources
Meranti--CLAP_freesound/, CLAPv2--*/, facebook--PE-Video/; outputs
jordangong--laion-audio-100k/ and jordangong--pvd-100k/. Paths stored in
manifests are relative to $PLATO_DATA_ROOT.
"""
import argparse
import collections
import glob
import io
import json
import os
import sys
import tarfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import paths

FINAL = 100_000
N_QUERY = 1000
SHARD_ROWS = {"laion_audio": 5_000, "pvd": 10_000}  # audio shards ~1 GB (binary offsets cap a column chunk at 2 GB)
CANDIDATES = 102_000        # 2% reserve for clips that fail to decode
CHUNK = 2000
NESTED_SIZES = [5_000, 10_000, 50_000, 100_000]   # total gallery sizes, queries included

AUDIO_SR = 16_000
AUDIO_MAX_S = 30.0
AUDIO_MIN_S = 0.1
VIDEO_MIN_FRAMES = 16

CLAP_SOURCES = {            # dir under data root -> source name
    "CLAPv2--BBCSoundEffects": "bbc",
    "CLAPv2--audiostock": "audiostock",
    "CLAPv2--epidemic_sound_effects_t5_debiased": "epidemic",
    "CLAPv2--paramount_motion": "paramount",
    "CLAPv2--sonniss_game_effects": "sonniss",
    "CLAPv2--wesoundeffects": "wesound",
}
FS_DIR = "Meranti--CLAP_freesound/freesound"
PVD_DIR = "facebook--PE-Video"

OUT_NAME = {"laion_audio": "jordangong--laion-audio-100k", "pvd": "jordangong--pvd-100k"}
CONFIG_NAME = {"laion_audio": "laion_audio_100k", "pvd": "pvd_100k"}


# ----------------------------------------------------------------- helpers

def root():
    return paths.data_root()


def out_root(args):
    return os.path.expanduser(args.out) if args.out else os.path.join(root(), OUT_NAME[args.dataset])


def wpath(args, *p):
    return os.path.join(out_root(args), "work", *p)


def write_atomic(table, path, **kw):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    pq.write_table(table, tmp, **kw)
    os.replace(tmp, path)


def write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(path + ".tmp", path)


def my_items(n, args):
    task = args.task if args.task is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    ntasks = args.ntasks if args.ntasks is not None else int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    return list(range(task, n, ntasks))


def ncpu():
    return int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))


def run_pool(fn, jobs, procs):
    if procs <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(procs) as pool:
        return pool.map(fn, jobs, chunksize=1)


def read_tar_bytes(relpath, offset, size):
    with open(os.path.join(root(), relpath), "rb") as f:
        f.seek(offset)
        b = f.read(size)
    if len(b) != size:
        raise IOError(f"short read {relpath}@{offset}: {len(b)}/{size}")
    return b


def word_stats(texts):
    w = np.array([len(t.split()) for t in texts])
    return {"words_mean": round(float(w.mean()), 2), "words_median": float(np.median(w)),
            "words_p90": float(np.quantile(w, 0.9))}


def order_and_dedup(table, key_col, seed):
    """Seeded random order; keep the first row per value of `key_col`."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(table.num_rows)
    table = table.take(pa.array(perm))
    keys = table.column(key_col).to_pylist()
    seen, keep = set(), []
    for i, k in enumerate(keys):
        if k not in seen:
            seen.add(k)
            keep.append(i)
    return table.take(pa.array(keep)), table.num_rows - len(keep)


# ----------------------------------------------------------------- tar index

def scan_tar(relpath, want_ext):
    """Members of one webdataset tar: {stem: {ext: (offset_data, size)}} plus
    the parsed json of each stem."""
    members, meta = collections.defaultdict(dict), {}
    with tarfile.open(os.path.join(root(), relpath), "r:") as tf:
        for ti in tf:
            if not ti.isfile():
                continue
            name = ti.name[2:] if ti.name.startswith("./") else ti.name
            stem, ext = os.path.splitext(name)
            ext = ext.lstrip(".")
            if ext == "json":
                meta[stem] = json.loads(tf.extractfile(ti).read())
            elif ext == want_ext:
                members[stem][ext] = (ti.offset_data, ti.size)
    return members, meta


def _fs_index_one(job):
    relpath, out = job
    if os.path.exists(out):
        return out, 0
    members, meta = scan_tar(relpath, "flac")
    rows = collections.defaultdict(list)
    for stem, m in members.items():
        if "flac" not in m or stem not in meta:
            continue
        j = meta[stem]
        od = j.get("original_data", {}) or {}
        texts = [t.strip() for t in (j.get("text") or []) if isinstance(t, str)]
        rows["key"].append("/".join(stem.split("/")[-2:]))
        rows["tar"].append(relpath)
        rows["offset"].append(m["flac"][0])
        rows["size"].append(m["flac"][1])
        rows["captions"].append(texts)
        rows["freesound_id"].append(int(od["id"]) if od.get("id") is not None else -1)
        rows["username"].append(str(od.get("username", "")))
    write_atomic(pa.table(rows) if rows else pa.table({"key": pa.array([], pa.string())}), out)
    return out, len(rows["key"])


def clap_index():
    rows = collections.defaultdict(list)
    for d, src in CLAP_SOURCES.items():
        files = sorted(glob.glob(os.path.join(root(), d, "data", "*", "*.parquet")))
        if not files:
            raise FileNotFoundError(f"no parquets under {root()}/{d}")
        for f in files:
            pf = pq.ParquetFile(f)
            cols = ["index", "text", "audio_len"] + (["text_augment_all"] if src == "epidemic" else [])
            t = pf.read(columns=cols).to_pylist()
            rel = os.path.relpath(f, root())
            for r, row in enumerate(t):
                text = (row["text"] or "").strip()
                caps = [text]
                if src == "epidemic":
                    aug = [a.strip() for a in (row.get("text_augment_all") or []) if a and a.strip()]
                    caps = aug or caps
                rows["source"].append(src)
                rows["file"].append(rel)
                rows["row"].append(r)
                rows["index"].append(row["index"])
                rows["text"].append(text)
                rows["audio_len"].append(float(row["audio_len"]))
                rows["captions"].append(caps)
        print(f"  {src}: {len(files)} files, {sum(1 for s in rows['source'] if s == src)} rows", flush=True)
    return pa.table(rows)


# ----------------------------------------------------------------- laion_audio

def audio_index(args):
    tars = sorted(glob.glob(os.path.join(root(), FS_DIR, "*", "*.tar")))
    if len(tars) != 1008:
        raise RuntimeError(f"expected 1008 freesound tars, found {len(tars)}")
    jobs = []
    for t in tars:
        rel = os.path.relpath(t, root())
        tag = f"{os.path.basename(os.path.dirname(t))}-{os.path.basename(t)[:-4]}"
        jobs.append((rel, wpath(args, "fs_index", f"{tag}.parquet")))
    mine = [jobs[i] for i in my_items(len(jobs), args)]
    t0 = time.time()
    done = run_pool(_fs_index_one, mine, ncpu())
    print(f"freesound: {len(done)} tars, {sum(n for _, n in done)} new rows, {time.time() - t0:.0f}s")
    task = args.task if args.task is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    out = wpath(args, "clap_index.parquet")
    if task == 0 and not os.path.exists(out):
        write_atomic(clap_index(), out)
        print(f"wrote {out}")


def audio_pool(args):
    fs_files = sorted(glob.glob(wpath(args, "fs_index", "*.parquet")))
    if len(fs_files) != 1008:
        raise RuntimeError(f"{len(fs_files)}/1008 freesound index files; run index first")
    fs = pa.concat_tables([pq.read_table(f) for f in fs_files if pq.ParquetFile(f).metadata.num_rows])
    clap = pq.read_table(wpath(args, "clap_index.parquet"))
    stats = {"seed": args.seed, "sources": {}}

    rows = collections.defaultdict(list)
    # Freesound: one row per freesound id (a few ids appear under two keys)
    seen = set()
    fs_rows = fs.to_pylist()
    for r in fs_rows:
        fid = r["freesound_id"]
        if fid >= 0 and fid in seen:
            continue
        seen.add(fid)
        caps = r["captions"]
        rows["source"].append("freesound")
        rows["source_key"].append(f"{r['key']}|fsid={fid}|{r['username']}")
        rows["caption"].append(caps[0] if caps else "")
        rows["captions"].append(caps)
        rows["file"].append(r["tar"])
        rows["offset"].append(r["offset"])
        rows["size"].append(r["size"])
        rows["row"].append(-1)
    stats["sources"]["freesound"] = {"members": len(fs_rows), "unique": len(seen)}
    # CLAPv2: one row per (source, text, length); the uploads repeat rows under two index names.
    # This also merges distinct clips with equal text and length (Epidemic/Audiostock series);
    # the caption dedup below would drop all but one of those anyway.
    ckeys = set()
    for r in clap.to_pylist():
        k = (r["source"], r["text"], round(r["audio_len"], 2))
        st = stats["sources"].setdefault(r["source"], {"rows": 0, "unique_text_len": 0})
        st["rows"] += 1
        if k in ckeys:
            continue
        ckeys.add(k)
        st["unique_text_len"] += 1
        rows["source"].append(r["source"])
        rows["source_key"].append(f"{r['index']}|len={r['audio_len']:.2f}")
        rows["caption"].append(r["captions"][0] if r["captions"] else "")
        rows["captions"].append(r["captions"])
        rows["file"].append(r["file"])
        rows["offset"].append(-1)
        rows["size"].append(-1)
        rows["row"].append(r["row"])
    t = pa.table(rows)
    finish_pool(args, t, stats)


def finish_pool(args, t, stats):
    caps = t.column("caption").to_pylist()
    nonempty = [i for i, c in enumerate(caps) if c and c.strip()]
    stats["clips"] = t.num_rows
    stats["empty_caption"] = t.num_rows - len(nonempty)
    t = t.take(pa.array(nonempty))
    t, ndup = order_and_dedup(t, "caption", args.seed)
    # Fresh random order of the deduped pool (App. G.1: "randomly sampled from the final
    # pool"). The dedup order itself is biased: a caption with c copies keeps its earliest
    # copy, so heavily duplicated captions sit near the front and a prefix over-samples them.
    t = t.take(pa.array(np.random.default_rng(args.seed + 1).permutation(t.num_rows)))
    stats["caption_duplicates"] = ndup
    stats["pool"] = t.num_rows
    t = t.append_column("order", pa.array(np.arange(t.num_rows, dtype=np.int64)))
    if "source" in t.column_names:
        stats["pool_by_source"] = dict(collections.Counter(t.column("source").to_pylist()))
    stats["pool_caption"] = word_stats(t.column("caption").to_pylist())
    write_atomic(t, wpath(args, "pool.parquet"))
    write_json(stats, wpath(args, "pool_stats.json"))
    print(json.dumps(stats, indent=2))


def _decode_audio(b):
    import soundfile as sf
    import soxr
    with sf.SoundFile(io.BytesIO(b)) as f:
        sr, ch, frames = f.samplerate, f.channels, f.frames
        x = f.read(frames=min(frames, int(AUDIO_MAX_S * sr) + sr), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    y = soxr.resample(x, sr, AUDIO_SR, quality="HQ") if sr != AUDIO_SR else x
    y = np.clip(y[: int(AUDIO_MAX_S * AUDIO_SR)], -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, y, AUDIO_SR, format="FLAC", subtype="PCM_16")
    return buf.getvalue(), len(y), sr, ch, frames / sr


def _audio_pack_one(job):
    _name, rows, out = job
    if os.path.exists(out):
        return out, None
    # CLAPv2 rows: read each source file's audio column once, decode, drop it
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["file"] if r["row"] >= 0 else None].append(r)
    res = {}
    for f, rs in groups.items():
        col = pq.read_table(os.path.join(root(), f), columns=["audio"]).column("audio") if f else None
        for r in rs:
            err, audio, n, sr, ch, secs = "", b"", 0, 0, 0, 0.0
            try:
                b = col[r["row"]].as_py()["bytes"] if f else read_tar_bytes(r["file"], r["offset"], r["size"])
                audio, n, sr, ch, secs = _decode_audio(b)
                if n < AUDIO_MIN_S * AUDIO_SR:
                    err = f"short ({n} samples)"
            except Exception as e:  # recorded; the clip is skipped at finalize
                err = f"{type(e).__name__}: {e}"[:300]
            res[r["order"]] = (audio if not err else b"", n, sr, ch, float(secs), err)
        del col
    out_rows = collections.defaultdict(list)
    for r in rows:
        audio, n, sr, ch, secs, err = res[r["order"]]
        for k in ("order", "source", "source_key", "caption", "captions"):
            out_rows[k].append(r[k])
        for k, v in zip(("audio", "num_samples", "orig_sr", "orig_channels", "orig_seconds", "error"),
                        (audio, n, sr, ch, secs, err)):
            out_rows[k].append(v)
    write_atomic(pa.table(out_rows), out)
    return out, sum(1 for e in out_rows["error"] if e)


def _video_pack_one(job):
    _name, rows, out = job
    if os.path.exists(out):
        return out, None
    import av
    out_rows = collections.defaultdict(list)
    for r in rows:
        err, n, w, h, rate = "", 0, 0, 0, 0.0
        try:
            b = read_tar_bytes(r["tar"], r["offset"], r["size"])
            with av.open(io.BytesIO(b)) as c:
                s = c.streams.video[0]
                s.thread_type = "AUTO"
                w, h = s.codec_context.width, s.codec_context.height
                rate = float(s.average_rate) if s.average_rate else 0.0
                for _ in c.decode(s):
                    n += 1
            if n < VIDEO_MIN_FRAMES:
                err = f"only {n} frames"
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:300]
        out_rows["order"].append(r["order"])
        out_rows["decoded_frames"].append(n)
        out_rows["dec_width"].append(w)
        out_rows["dec_height"].append(h)
        out_rows["dec_fps"].append(rate)
        out_rows["error"].append(err)
    write_atomic(pa.table(out_rows), out)
    return out, sum(1 for e in out_rows["error"] if e)


def pack_units(args, cand):
    """Work units of the first CANDIDATES pool rows: Freesound / PVD rows in
    order-chunks of CHUNK (offset reads from the tars), CLAPv2 rows grouped by
    source parquet (one row group per file, so each file is read once), files
    packed into units of <= CHUNK rows."""
    sub = "packed" if args.dataset == "laion_audio" else "probed"
    units = []
    offs = [r for r in cand if r.get("row", -1) < 0]
    for c in range(0, len(offs), CHUNK):
        units.append((f"o{c // CHUNK:04d}", offs[c:c + CHUNK]))
    by_file = collections.defaultdict(list)
    for r in cand:
        if r.get("row", -1) >= 0:
            by_file[r["file"]].append(r)
    cur, cid = [], 0
    for f in sorted(by_file):
        if cur and len(cur) + len(by_file[f]) > CHUNK:
            units.append((f"p{cid:04d}", cur))
            cur, cid = [], cid + 1
        cur = cur + by_file[f]
    if cur:
        units.append((f"p{cid:04d}", cur))
    return [(name, rows, wpath(args, sub, f"{name}.parquet")) for name, rows in units]


def stage_pack(args):
    pool = pq.read_table(wpath(args, "pool.parquet"))
    cand = pool.slice(0, min(CANDIDATES, pool.num_rows)).to_pylist()
    jobs = pack_units(args, cand)
    mine = [jobs[i] for i in my_items(len(jobs), args)]
    fn = _audio_pack_one if args.dataset == "laion_audio" else _video_pack_one
    t0 = time.time()
    procs = ncpu() if args.dataset == "laion_audio" else max(1, ncpu() // 2)  # PyAV decodes with 2 threads
    for out, nerr in run_pool(fn, mine, procs):
        print(f"  {os.path.basename(out)}: {'skipped (exists)' if nerr is None else f'{nerr} errors'}", flush=True)
    print(f"pack: {len(mine)}/{len(jobs)} units, {time.time() - t0:.0f}s")


def nested_indices(args):
    d = os.path.join(out_root(args), "nested_indices")
    os.makedirs(d, exist_ok=True)
    for n in NESTED_SIZES:
        np.save(os.path.join(d, f"gallery_indices_{n}.npy"), np.arange(n - N_QUERY, dtype=np.int64))
    write_json({"n_query": N_QUERY, "sizes_include_queries": True, "sizes": [N_QUERY] + NESTED_SIZES,
                "query_rows": [0, N_QUERY], "gallery_row_of_id": f"pos = {N_QUERY} + id"},
               os.path.join(d, "meta.json"))


def finalize_common(args, done, extra_stats):
    """`done`: candidate table (sorted by order) with an `error` column."""
    errs = done.column("error").to_pylist()
    ok = [i for i, e in enumerate(errs) if not e]
    if len(ok) < FINAL:
        raise RuntimeError(f"only {len(ok)} decodable candidates of {done.num_rows}")
    sel = done.take(pa.array(ok[:FINAL]))
    last_order = sel.column("order")[FINAL - 1].as_py()
    failed = [(done.column("order")[i].as_py(), errs[i]) for i in range(done.num_rows)
              if errs[i] and done.column("order")[i].as_py() <= last_order]
    sel = sel.append_column("pos", pa.array(np.arange(FINAL, dtype=np.int64)))
    stats = dict(extra_stats)
    stats.update({"final": FINAL, "n_query": N_QUERY, "candidates_used_up_to_order": last_order,
                  "skipped_undecodable": len(failed), "skipped_examples": failed[:20],
                  "final_caption": word_stats(sel.column("caption").to_pylist()),
                  "query_caption": word_stats(sel.column("caption").to_pylist()[:N_QUERY])})
    return sel, stats


def stage_finalize(args):
    pool = pq.read_table(wpath(args, "pool.parquet"))
    units = pack_units(args, pool.slice(0, min(CANDIDATES, pool.num_rows)).to_pylist())
    files = [out for _, _, out in units]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise RuntimeError(f"{len(missing)}/{len(files)} pack units missing, e.g. {missing[0]}")
    pool_stats = json.load(open(wpath(args, "pool_stats.json")))
    cfg_dir = os.path.join(out_root(args), CONFIG_NAME[args.dataset])
    rows_per = SHARD_ROWS[args.dataset]
    nshards = FINAL // rows_per
    if args.dataset == "laion_audio":
        # 19 GB of FLAC: select on metadata only, then gather audio per output shard
        # (a take/sort over one binary column > 2 GB overflows its 32-bit offsets)
        tabs = [pq.read_table(f) for f in files]
        meta = pa.concat_tables([
            t.drop_columns(["audio"]).append_column("_f", pa.array(np.full(t.num_rows, i, np.int32)))
             .append_column("_r", pa.array(np.arange(t.num_rows, dtype=np.int32))) for i, t in enumerate(tabs)])
        sel, stats = finalize_common(args, meta.sort_by("order"), {"pool": pool_stats})
        sel = sel.drop_columns(["error"])
        stats["final_by_source"] = dict(collections.Counter(sel.column("source").to_pylist()))
        stats["query_by_source"] = dict(collections.Counter(sel.column("source").to_pylist()[:N_QUERY]))
        secs = np.array(sel.column("num_samples").to_pylist()) / AUDIO_SR
        orig = np.array(sel.column("orig_seconds").to_pylist())
        stats["stored_seconds"] = {"mean": round(float(secs.mean()), 2), "median": float(np.median(secs)),
                                   "frac_full_30s": round(float((secs >= AUDIO_MAX_S - 1e-3).mean()), 4)}
        stats["orig_seconds"] = {"median": float(np.median(orig)), "p90": float(np.quantile(orig, .9)),
                                 "frac_over_10s": round(float((orig > 10.08).mean()), 4)}
        cols = ["pos", "order", "source", "source_key", "caption", "captions", "audio", "num_samples",
                "orig_sr", "orig_channels", "orig_seconds"]
        prefix = "train"
    else:
        probed = pa.concat_tables([pq.read_table(f) for f in files]).sort_by("order")
        cand = pool.slice(0, probed.num_rows)
        assert cand.column("order").to_pylist() == probed.column("order").to_pylist()
        for c in probed.column_names:
            if c != "order":
                cand = cand.append_column(c, probed.column(c))
        sel, stats = finalize_common(args, cand, {"pool": pool_stats})
        sel = sel.drop_columns(["error"])
        stats["final_by_split"] = dict(collections.Counter(sel.column("split").to_pylist()))
        fr = np.array(sel.column("decoded_frames").to_pylist())
        stats["decoded_frames"] = {"min": int(fr.min()), "median": float(np.median(fr)), "max": int(fr.max())}
        cols = ["pos", "order", "video_id", "split", "caption", "model_caption", "description", "keywords",
                "category", "tar", "offset", "size", "decoded_frames", "dec_width", "dec_height", "dec_fps",
                "fps", "duration", "width", "height"]
        prefix = "manifest"
    for s in range(nshards):
        part = sel.slice(s * rows_per, rows_per)
        if args.dataset == "laion_audio":
            fi, ri = part.column("_f").to_pylist(), part.column("_r").to_pylist()
            audio = [tabs[f].column("audio")[r].as_py() for f, r in zip(fi, ri)]
            part = part.append_column("audio", pa.array(audio, pa.binary()))
        out = os.path.join(cfg_dir, f"{prefix}-{s:05d}-of-{nshards:05d}.parquet")
        write_atomic(part.select(cols), out, row_group_size=500)
    nested_indices(args)
    write_json(stats, os.path.join(out_root(args), "stats.json"))
    print(json.dumps({k: v for k, v in stats.items() if k != "pool"}, indent=2))
    print(f"wrote {nshards} shards to {cfg_dir}")


# ----------------------------------------------------------------- pvd

def _pvd_index_one(job):
    relpath, split, out = job
    if os.path.exists(out):
        return out, 0
    members, meta = scan_tar(relpath, "mp4")
    rows = collections.defaultdict(list)
    for stem, m in members.items():
        if "mp4" not in m or stem not in meta:
            continue
        j = meta[stem]
        rows["video_id"].append(int(j["video_id"]))
        rows["split"].append(split)
        rows["tar"].append(relpath)
        rows["offset"].append(m["mp4"][0])
        rows["size"].append(m["mp4"][1])
        for k in ("human_caption", "model_caption", "description", "keywords", "category_name"):
            v = j.get(k)
            rows[k].append(v.strip() if isinstance(v, str) else "")
        for k in ("fps", "video_duration_in_s", "num_frames", "width", "height"):
            v = j.get(k)
            rows[k].append(float(v) if v is not None else float("nan"))
    write_atomic(pa.table(rows), out)
    return out, len(rows["video_id"])


def pvd_index(args):
    jobs = []
    for split in ("train", "test"):
        for t in sorted(glob.glob(os.path.join(root(), PVD_DIR, split, "*.tar"))):
            jobs.append((os.path.relpath(t, root()), split,
                         wpath(args, "index", f"{split}-{os.path.basename(t)[:-4]}.parquet")))
    if len(jobs) != 238:
        raise RuntimeError(f"expected 208 + 30 PE-Video tars, found {len(jobs)}")
    mine = [jobs[i] for i in my_items(len(jobs), args)]
    t0 = time.time()
    done = run_pool(_pvd_index_one, mine, ncpu())
    print(f"pvd: {len(done)} tars, {sum(n for _, n in done)} new rows, {time.time() - t0:.0f}s")


def pvd_pool(args):
    files = sorted(glob.glob(wpath(args, "index", "*.parquet")))
    if len(files) != 238:
        raise RuntimeError(f"{len(files)}/238 index files; run index first")
    t = pa.concat_tables([pq.read_table(f) for f in files])
    vids = t.column("video_id").to_pylist()
    stats = {"seed": args.seed, "indexed": t.num_rows, "unique_video_id": len(set(vids)),
             "by_split": dict(collections.Counter(t.column("split").to_pylist()))}
    if len(set(vids)) != t.num_rows:
        raise RuntimeError("duplicate video ids across tars")
    t = t.rename_columns(["caption" if c == "human_caption" else
                          "category" if c == "category_name" else
                          "duration" if c == "video_duration_in_s" else c for c in t.column_names])
    finish_pool(args, t, stats)


# ----------------------------------------------------------------- main

def stage_index(args):
    (audio_index if args.dataset == "laion_audio" else pvd_index)(args)


def stage_pool(args):
    (audio_pool if args.dataset == "laion_audio" else pvd_pool)(args)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=["laion_audio", "pvd"])
    p.add_argument("stage", choices=["index", "pool", "pack", "finalize"])
    p.add_argument("--out", default=None, help="output root (default $PLATO_DATA_ROOT/<jordangong--...>)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--task", type=int, default=None, help="array task id (default $SLURM_ARRAY_TASK_ID)")
    p.add_argument("--ntasks", type=int, default=None, help="array size (default $SLURM_ARRAY_TASK_COUNT)")
    args = p.parse_args()
    globals()[f"stage_{args.stage}"](args)


if __name__ == "__main__":
    sys.exit(main())
