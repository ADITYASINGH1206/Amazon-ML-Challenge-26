#!/usr/bin/env python3
"""
EDA for the Business Entity Resolution challenge.

Reads student_resource/dataset/ next to this script by default (works from any cwd):
    python eda.py                                   # all sections
    python eda.py --sections 8,9 --out eda_report_8_9.txt
    python eda.py --data-dir path/to/dataset --probe-n 3000 --workers 16

Requires : pandas, numpy
Optional : scikit-learn  (TF-IDF blocking probe, sections 8b and 9b)
           torch         (GPU top-K search for 8b/9b; falls back to CPU torch if no CUDA)
           pyarrow       (faster TSV loading)
           rapidfuzz     (extra fuzzy-match scores in section 6)
    pip install pandas numpy scikit-learn rapidfuzz torch pyarrow

Sections 8/9 stream the pool through a process pool (normalise + TF-IDF in parallel) and
score chunks on the GPU against the probe queries, keeping a running top-K. The full pool
matrix is never held in memory, and only the n-gram columns the queries use are kept
(cosine scores are unchanged by that, since rows are L2-normalised over the full vocabulary).

Everything printed is also written to the --out report file -- send that file back.
"""
import argparse
import multiprocessing as mp
import os
import random
import re
import sys
import time
import unicodedata
import warnings
from collections import Counter, defaultdict, deque
from functools import lru_cache

import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz
    HAVE_RF = True
except ImportError:
    HAVE_RF = False
try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import normalize
    HAVE_SK = True
except ImportError:
    HAVE_SK = False

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 300)


# ----------------------------------------------------------------------------- output helpers
class Tee:
    """Write to console and to a UTF-8 report file (console errors on odd chars are tolerated)."""

    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.out = sys.__stdout__

    def write(self, s):
        self.f.write(s)
        try:
            self.out.write(s)
        except UnicodeEncodeError:
            enc = self.out.encoding or "ascii"
            self.out.write(s.encode(enc, "replace").decode(enc))

    def flush(self):
        self.f.flush()
        self.out.flush()


T0 = time.time()
KEY = {}  # headline numbers, printed at the end


def section(title):
    print("\n" + "=" * 100)
    print(f"## {title}    [t={time.time() - T0:.0f}s]")
    print("=" * 100)


def sub(title):
    print(f"\n--- {title} ---")


def progress(msg):
    """Console-only progress line (kept out of the report)."""
    sys.__stderr__.write(f"  [t={time.time() - T0:.0f}s] {msg}\n")
    sys.__stderr__.flush()


def short(s, n=80):
    s = str(s)
    return s if len(s) <= n else s[: n - 3] + "..."


def fmt(items):
    return ", ".join(f"{t}:{n}" for t, n in items)


# ----------------------------------------------------------------------------- text helpers
def _norm(s):
    s = unicodedata.normalize("NFKD", str(s).lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^\w\s]", " ", s).replace("_", " ")
    return re.sub(r"\s+", " ", s).strip()


norm = lru_cache(maxsize=None)(_norm)


@lru_cache(maxsize=None)
def toks(s):
    return frozenset(norm(s).split())


@lru_cache(maxsize=None)
def tri(s):
    t = f" {norm(s)} "
    return frozenset(t[i : i + 3] for i in range(len(t) - 2))


POST_RE = re.compile(r"(?<!\d)(\d{5,6})(?!\d)")


@lru_cache(maxsize=None)
def posts(s):
    """5-digit (US ZIP / FR postcode) or 6-digit (IN PIN) codes; '700 001' is joined to '700001'."""
    s2 = re.sub(r"(?<!\d)(\d{3}) (\d{3})(?!\d)", r"\1\2", str(s))
    return frozenset(POST_RE.findall(s2))


@lru_cache(maxsize=None)
def nums(s):
    return frozenset(re.findall(r"\d+", str(s)))


def jacc(a, b):
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def load(path):
    if not os.path.exists(path):
        print(f"  [MISSING] {path}")
        return None
    try:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, engine="pyarrow")
    except Exception:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].str.strip()
    with open(path, "rb") as f:
        n = sum(1 for line in f if line.strip()) - 1
    flag = "" if n == len(df) else f"   <-- WARNING: {n} data lines but {len(df)} rows parsed (quote chars?)"
    print(f"  {path}: {len(df):,} rows, cols={list(df.columns)}{flag}")
    return df


def q(x, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    return [round(float(v), 3) for v in np.quantile(x, qs)] if len(x) else []


# ----------------------------------------------------------------------------- parallel TF-IDF
# char_wb 2-4 grams, min_df=2, sublinear tf, smooth idf, L2 rows -- same as TfidfVectorizer
# fitted on the pool, but the document frequencies are counted in parallel worker processes.
NGRAM = dict(analyzer="char_wb", ngram_range=(2, 4))
_W = {}  # per-worker state set by _tx_init


def _view_texts(names, addrs, views):
    nn = [_norm(x) for x in names]
    out = {}
    if "name" in views:
        out["name"] = nn
    if "name+addr" in views:
        out["name+addr"] = [f"{a} {_norm(b)}".strip() for a, b in zip(nn, addrs)]
    return nn, out


def _df_task(names, addrs, views):
    _, texts = _view_texts(names, addrs, views)
    res = {}
    for v in views:
        cv = CountVectorizer(binary=True, dtype=np.int32, **NGRAM)
        try:
            X = cv.fit_transform(texts[v])
        except ValueError:  # empty vocabulary
            res[v] = pd.Series(dtype=np.int64)
            continue
        res[v] = pd.Series(np.bincount(X.indices, minlength=X.shape[1]), index=cv.get_feature_names_out())
    return res


def _tfidf(cv, idf, texts):
    X = cv.transform(texts)
    np.log(X.data, out=X.data)
    X.data += 1
    X.data *= idf[X.indices]
    return normalize(X, norm="l2", copy=False)


def _tx_init(specs, probe_tokens):
    warnings.filterwarnings("ignore")
    _W["specs"] = {v: (CountVectorizer(vocabulary=vocab, dtype=np.float32, **NGRAM), idf, used)
                   for v, (vocab, idf, used) in specs.items()}
    _W["tok"] = probe_tokens


def _tx_task(start, names, addrs):
    specs = _W["specs"]
    nn, texts = _view_texts(names, addrs, list(specs))
    mats = {v: _tfidf(cv, idf, texts[v])[:, used] for v, (cv, idf, used) in specs.items()}
    post = {}
    tok = _W["tok"]
    if tok:
        d = defaultdict(list)
        for i, s in enumerate(nn):
            for t in set(s.split()):
                if t in tok:
                    d[t].append(start + i)
        post = {t: np.array(L, dtype=np.int64) for t, L in d.items()}
    return start, mats, post


def parallel_map(pool, fn, tasks, window):
    """Ordered apply_async with at most `window` tasks in flight (bounds result memory)."""
    pending = deque()
    for t in tasks:
        pending.append(pool.apply_async(fn, t))
        if len(pending) >= window:
            yield pending.popleft().get()
    while pending:
        yield pending.popleft().get()


def fit_vocab(workers, names, addrs, views, chunk):
    n = len(names)
    acc = {v: [] for v in views}
    tasks = ((names[i : i + chunk], addrs[i : i + chunk], views) for i in range(0, n, chunk))
    done = 0
    with mp.Pool(workers) as mpp:
        for res in parallel_map(mpp, _df_task, tasks, workers + 2):
            for v, s in res.items():
                acc[v].append(s)
                if len(acc[v]) >= 12:
                    acc[v] = [pd.concat(acc[v]).groupby(level=0).sum()]
            done += chunk
            progress(f"doc-freq pass {min(done, n):,}/{n:,}")
    out = {}
    for v in views:
        df = pd.concat(acc[v]).groupby(level=0).sum()
        df = df[df >= 2].sort_index()
        vocab = dict(zip(df.index, range(len(df))))
        idf = (np.log((1 + n) / (1 + df.values.astype(np.float64))) + 1).astype(np.float32)
        out[v] = (vocab, idf)
        print(f"  {v:<9} vocabulary: {len(vocab):,} n-grams (min_df=2 over {n:,} pool docs)")
    return out


def query_matrix(vocab, idf, texts):
    X = _tfidf(CountVectorizer(vocabulary=vocab, dtype=np.float32, **NGRAM), idf, texts)
    used = np.unique(X.indices)
    return X[:, used], used


class TopK:
    """Running exact top-K cosine per query over streamed pool chunks (GPU if available)."""

    def __init__(self, Xq, K, qc, filters=(False, True), block_bytes=256 << 20):
        import torch
        warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support is in beta.*")
        self.t = torch
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.XqT = torch.from_numpy(np.ascontiguousarray(Xq.T.toarray())).to(self.dev)
        self.qc = torch.from_numpy(np.asarray(qc, dtype=np.int64)).to(self.dev)
        self.K = K
        self.R = max(1024, block_bytes // (4 * Xq.shape[0]))
        self.state = {f: None for f in filters}

    def add(self, M, start, pc):
        t = self.t
        for r0 in range(0, M.shape[0], self.R):
            B = M[r0 : r0 + self.R]
            n = B.shape[0]
            if B.nnz:
                A = t.sparse_csr_tensor(t.from_numpy(B.indptr.astype(np.int64)),
                                        t.from_numpy(B.indices.astype(np.int64)),
                                        t.from_numpy(B.data), size=B.shape).to(self.dev)
                S = t.sparse.mm(A, self.XqT)
            else:
                S = t.zeros((n, self.XqT.shape[1]), device=self.dev)
            k = min(self.K, n)
            if False in self.state:
                self._merge(False, S, k, start + r0)
            if True in self.state:
                pcb = t.from_numpy(np.asarray(pc[r0 : r0 + n], dtype=np.int64)).to(self.dev)
                S.masked_fill_(pcb[:, None] != self.qc[None, :], -1.0)
                self._merge(True, S, k, start + r0)

    def _merge(self, f, S, k, base):
        t = self.t
        v, i = t.topk(S, k, dim=0)
        i = i + base
        st = self.state[f]
        if st is not None:
            v, i = t.cat([st[0], v]), t.cat([st[1], i])
            v, o = t.topk(v, min(self.K, v.shape[0]), dim=0)
            i = t.gather(i, 0, o)
        self.state[f] = (v, i)

    def result(self, f):
        v, i = self.state[f]
        return i.T.cpu().numpy(), v.T.cpu().numpy()


def stream_pool(workers, names, addrs, specs, probe_tokens, chunk, on_chunk):
    n = len(names)
    tasks = ((i, names[i : i + chunk], addrs[i : i + chunk]) for i in range(0, n, chunk))
    with mp.Pool(workers, initializer=_tx_init, initargs=(specs, probe_tokens)) as mpp:
        for start, mats, post in parallel_map(mpp, _tx_task, tasks, workers + 4):
            on_chunk(start, mats, post)
            progress(f"TF-IDF + GPU scoring {min(start + chunk, n):,}/{n:,}")


# ----------------------------------------------------------------------------- main
def parse_sections(s):
    if s.strip().lower() == "all":
        return set(range(11))
    out = set()
    for part in s.split(","):
        a, _, b = part.strip().partition("-")
        out |= set(range(int(a), int(b or a) + 1))
    deps = {4: {3}, 5: {3}, 6: {3, 4}, 7: {2}}
    for s_ in list(out):
        out |= deps.get(s_, set())
    return out | {0, 10}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "student_resource", "dataset"))
    ap.add_argument("--out", default="eda_report.txt")
    ap.add_argument("--sections", default="all", help="e.g. 'all', '8,9', '2-7'")
    ap.add_argument("--samples", type=int, default=6, help="match groups shown per country")
    ap.add_argument("--probe-n", type=int, default=2000, help="S1 records used in blocking probes")
    ap.add_argument("--workers", type=int, default=min(16, max(1, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--chunk", type=int, default=50000, help="pool rows per worker task (sections 8/9)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    want = parse_sections(args.sections).__contains__

    sys.stdout = Tee(args.out)
    rng = random.Random(args.seed)
    try:
        import torch
        gpu = f"torch={torch.__version__} cuda={torch.cuda.is_available()}"
        if torch.cuda.is_available():
            gpu += f" ({torch.cuda.get_device_name(0)})"
        HAVE_TORCH = True
    except ImportError:
        gpu, HAVE_TORCH = "torch=False", False
    print(f"EDA run {time.strftime('%Y-%m-%d %H:%M')} | pandas {pd.__version__} | numpy {np.__version__} "
          f"| sklearn={HAVE_SK} rapidfuzz={HAVE_RF} | {gpu} | workers={args.workers} "
          f"| sections={sorted(parse_sections(args.sections))}")
    TFIDF_OK = HAVE_SK and HAVE_TORCH

    # ------------------------------------------------------------------ 0
    section("0. FILES")
    F = {}
    splits = ("train", "test") if (want(1) or want(2)) else ("train",)
    for split in splits:
        for s in (1, 2, 3):
            F[(split, s)] = load(os.path.join(args.data_dir, split, f"{split}_source{s}.tsv"))
    gt = load(os.path.join(args.data_dir, "train", "train_ground_truth.tsv"))
    if gt is None or any(F[("train", s)] is None for s in (1, 2, 3)):
        print("Training files missing -- check --data-dir")
        return
    s1, s2, s3 = F[("train", 1)], F[("train", 2)], F[("train", 3)]
    pool = pd.concat([s2, s3], ignore_index=True)
    n_train = {s: len(F[("train", s)]) for s in (1, 2, 3)}
    train_countries = set(s1.country) | set(s2.country) | set(s3.country)

    # ------------------------------------------------------------------ 1
    if want(1):
        section("1. PER-FILE OVERVIEW")
        for (split, s), df in F.items():
            if df is None:
                continue
            print(f"\n[{split} source{s}] rows={len(df):,}  unique ids={df.entity_id.nunique():,}  "
                  f"id prefix ok={df.entity_id.str.startswith(f'S{s}-').mean():.2%}")
            for c in ("business_name", "business_address", "country"):
                if c in df:
                    print(f"   empty {c:<17}: {(df[c] == '').mean():.2%}")
            print(f"   duplicate name rows            : {df.duplicated('business_name').mean():.2%}")
            print(f"   duplicate name+address rows    : {df.duplicated(['business_name', 'business_address']).mean():.2%}")
            print(f"   country values: {df.country.value_counts().to_dict()}")

    # ------------------------------------------------------------------ 2
    if want(2):
        section("2. TEXT CHARACTERISTICS by split / source / country")
        A = pd.concat([df.assign(split=sp, src=f"S{s}") for (sp, s), df in F.items() if df is not None],
                      ignore_index=True)
        A["name_tok"] = A.business_name.map(lambda x: len(norm(x).split()))
        A["addr_tok"] = A.business_address.map(lambda x: len(norm(x).split()))
        A["non_ascii"] = (A.business_name + A.business_address).map(lambda x: any(ord(c) > 127 for c in x))
        A["zip5"] = A.business_address.str.contains(r"(?<!\d)\d{5}(?!\d)", regex=True)
        A["pin6"] = A.business_address.str.contains(r"(?<!\d)\d{6}(?!\d)", regex=True)
        A["pin_3_3"] = A.business_address.str.contains(r"(?<!\d)\d{3} \d{3}(?!\d)", regex=True)
        A["name_digit"] = A.business_name.str.contains(r"\d", regex=True)
        A["landmark"] = A.business_address.str.lower().str.contains(
            r"\b(?:near|nr|opp|opposite|behind|beside|next to|pres de|en face)\b", regex=True)
        A["commas"] = A.business_address.str.count(",")
        g = A.groupby(["split", "src", "country"]).agg(
            n=("entity_id", "size"),
            name_tok_med=("name_tok", "median"),
            addr_tok_med=("addr_tok", "median"),
            addr_tok_p90=("addr_tok", lambda x: x.quantile(0.9)),
            commas=("commas", "mean"),
            non_ascii=("non_ascii", "mean"),
            zip5=("zip5", "mean"),
            pin6=("pin6", "mean"),
            pin_3_3=("pin_3_3", "mean"),
            landmark=("landmark", "mean"),
            name_digit=("name_digit", "mean"),
        )
        print(g.round(3).to_string())

    # ------------------------------------------------------------------ 3
    if want(3):
        section("3. GROUND TRUTH STRUCTURE (train)")
        REC = {r.entity_id: (r.business_name, r.business_address, r.country)
               for r in pd.concat([s1, pool]).itertuples(index=False)}

        gt["m"] = gt.matched_entity_ids.map(lambda x: [t.strip() for t in x.split(",") if t.strip()])
        gt["k"] = gt.m.map(len)
        gt["k2"] = gt.m.map(lambda L: sum(t.startswith("S2-") for t in L))
        gt["k3"] = gt.m.map(lambda L: sum(t.startswith("S3-") for t in L))
        gt["country"] = gt.source1_entity_id.map(dict(zip(s1.entity_id, s1.country)))
        GTM = dict(zip(gt.source1_entity_id, gt.m))

        s1ids, gtids = set(s1.entity_id), set(gt.source1_entity_id)
        print(f"GT rows={len(gt):,} unique S1 ids={len(gtids):,} | S1 records={len(s1):,}")
        print(f"S1 ids missing from GT: {len(s1ids - gtids):,} | GT ids not in S1 file: {len(gtids - s1ids):,}")
        sing = (gt.k == 0).mean()
        KEY["train singleton rate (= score of predicting all-empty)"] = f"{sing:.4f}"
        print(f"Singleton rate: {sing:.2%}")
        print("\nMatches per S1 entity (share of entities, 10 = 10+):")
        print((gt.k.clip(upper=10).value_counts().sort_index() / len(gt)).round(4).to_string())
        nz = gt[gt.k > 0]
        if len(nz):
            print(f"\nNon-singletons: mean matches={nz.k.mean():.2f} median={nz.k.median():.0f} max={nz.k.max()}")
            comp = np.select([(nz.k2 > 0) & (nz.k3 > 0), nz.k2 > 0], ["S2+S3", "S2 only"], "S3 only")
            print("Composition:", pd.Series(comp).value_counts(normalize=True).round(3).to_dict())
            print(f"Entities with >1 match from the SAME source: S2 {(nz.k2 > 1).mean():.2%}, S3 {(nz.k3 > 1).mean():.2%}")
        print("\nPer S1 country:")
        pc = gt.groupby("country").agg(
            n=("k", "size"),
            singleton=("k", lambda x: (x == 0).mean()),
            mean_k=("k", "mean"),
            mean_k_nonzero=("k", lambda x: x[x > 0].mean()),
            mean_s2=("k2", "mean"),
            mean_s3=("k3", "mean"),
        )
        print(pc.round(3).to_string())

        allm = [t for L in gt.m for t in L]
        pool_ids = set(pool.entity_id)
        print(f"\nTotal true pairs: {len(allm):,} | matched IDs not found in S2/S3 files: {sum(t not in pool_ids for t in allm)}")
        print(f"Lists containing duplicate IDs: {sum(len(L) != len(set(L)) for L in gt.m)}")
        cnt = Counter(allm)
        multi = {t: c for t, c in cnt.items() if c > 1}
        print(f"S2/S3 records assigned to >1 S1 entity: {len(multi):,} "
              f"(max={max(multi.values()) if multi else 0})   <- tests the one-to-one assumption")
        KEY["S2/S3 records matched to >1 S1 entity"] = len(multi)
        matched = set(allm)
        for nm, df in (("S2", s2), ("S3", s3)):
            cov = df.entity_id.isin(matched)
            print(f"{nm}: {cov.mean():.2%} of records matched to some S1 ({(~cov).sum():,} unmatched distractors); "
                  f"by country {df.assign(c=cov).groupby('country').c.mean().round(3).to_dict()}")
            KEY[f"{nm} fraction of records that match some S1"] = f"{cov.mean():.3f}"
        print(f"Pool size ratio (S2+S3)/S1: {len(pool) / len(s1):.2f}")

    # ------------------------------------------------------------------ 4
    if want(4):
        section("4. COUNTRY CONSISTENCY on true pairs")
        P = pd.DataFrame([(a, b) for a, L in zip(gt.source1_entity_id, gt.m) for b in L], columns=["s1", "m"])
        P = P[P.s1.isin(s1ids) & P.m.isin(pool_ids)].reset_index(drop=True)
        P["c1"] = P.s1.map(lambda x: REC[x][2])
        P["c2"] = P.m.map(lambda x: REC[x][2])
        eq = (P.c1 == P.c2).mean() if len(P) else float("nan")
        print(f"Same country label on true pairs: {eq:.2%}")
        KEY["true pairs with same country label"] = f"{eq:.4f}"
        if len(P) and eq < 1:
            print(pd.crosstab(P.c1, P.c2).to_string())

    def show(eid, prefix=""):
        n, a, c = REC[eid]
        print(f"{prefix}{eid} [{c}] {short(n, 55)}  |  {short(a, 100)}")

    # ------------------------------------------------------------------ 5
    if want(5):
        section("5. SAMPLE MATCH GROUPS (read these closely)")
        for c in sorted(gt.country.dropna().unique()):
            g_c = gt[(gt.country == c) & (gt.k > 0)]
            pick = list(g_c.sample(min(len(g_c), args.samples), random_state=args.seed).source1_entity_id)
            big = g_c[g_c.k >= 3]
            if len(big):
                pick += list(big.sample(min(len(big), 2), random_state=args.seed).source1_entity_id)
            print(f"\n########## {c}")
            for e in dict.fromkeys(pick):
                show(e)
                for t in GTM[e]:
                    if t in REC:
                        show(t, "    -> ")
                print()
            sg = gt[(gt.country == c) & (gt.k == 0)]
            print(f"  singleton S1 records ({c}):")
            for e in sg.sample(min(len(sg), 4), random_state=args.seed).source1_entity_id:
                show(e, "    ")
            um = pool[(pool.country == c) & (~pool.entity_id.isin(matched))]
            print(f"  unmatched S2/S3 records ({c}):")
            for e in um.sample(min(len(um), 4), random_state=args.seed).entity_id:
                show(e, "    ")

    # ------------------------------------------------------------------ 6
    if want(6):
        section("6. PAIR-LEVEL SIGNALS: true pairs vs RANDOM same-country pairs (random negatives are easy!)")
        Ppos = P.sample(min(len(P), 30000), random_state=args.seed).reset_index(drop=True)
        pos_set = set(zip(P.s1, P.m))
        pool_by_c = {c: list(gp.entity_id) for c, gp in pool.groupby("country")}
        s1_list = list(s1.entity_id)
        neg, tries = [], 0
        while len(neg) < len(Ppos) and tries < 10 * len(Ppos) + 100:
            tries += 1
            a = rng.choice(s1_list)
            cands = pool_by_c.get(REC[a][2])
            if not cands:
                continue
            b = rng.choice(cands)
            if (a, b) not in pos_set:
                neg.append((a, b))

        def feats(a, b):
            na, aa, _ = REC[a]
            nb, ab, _ = REC[b]
            pa, pb = posts(aa), posts(ab)
            ua, ub = nums(aa), nums(ab)
            d = dict(
                name_tok_jacc=jacc(toks(na), toks(nb)),
                name_tri_jacc=jacc(tri(na), tri(nb)),
                addr_tok_jacc=jacc(toks(aa), toks(ab)),
                addr_tri_jacc=jacc(tri(aa), tri(ab)),
                name_exact=float(norm(na) == norm(nb)),
                share_name_tok=float(bool(toks(na) & toks(nb))),
                post_both=float(bool(pa and pb)),
                post_eq=float(bool(pa & pb)) if (pa and pb) else np.nan,
                num_share=float(bool(ua & ub)) if (ua and ub) else np.nan,
            )
            if HAVE_RF:
                d["name_tsr"] = fuzz.token_set_ratio(norm(na), norm(nb)) / 100
                d["addr_tsr"] = fuzz.token_set_ratio(norm(aa), norm(ab)) / 100
            return d

        rows = [dict(feats(a, b), label="pos", country=REC[a][2]) for a, b in zip(Ppos.s1, Ppos.m)]
        rows += [dict(feats(a, b), label="neg", country=REC[a][2]) for a, b in neg]
        Fd = pd.DataFrame(rows)
        cont = [c for c in ["name_tok_jacc", "name_tri_jacc", "addr_tok_jacc", "addr_tri_jacc", "name_tsr", "addr_tsr"]
                if c in Fd]
        binf = ["name_exact", "share_name_tok", "post_both", "post_eq", "num_share"]
        print(f"pos pairs={len(Ppos):,}  neg pairs={len(neg):,}")
        print("Continuous features, quantiles p10/p25/p50/p75/p90:")
        for f in cont:
            for lab in ("pos", "neg"):
                print(f"  {f:<14} {lab}: {q(Fd.loc[Fd.label == lab, f].dropna())}")
        print("\nBinary feature rates by country (post_eq / num_share only where both sides have one):")
        print(Fd.groupby(["country", "label"])[binf].mean().round(3).to_string())
        print("\nMedian continuous features by country:")
        print(Fd.groupby(["country", "label"])[cont].median().round(3).to_string())

        sub("Hardest TRUE pairs (lowest name+address trigram overlap) -- the noise we must survive")
        hp = Fd.iloc[: len(Ppos)].assign(s1=Ppos.s1.values, m=Ppos.m.values)
        hp["score"] = hp.name_tri_jacc + hp.addr_tri_jacc
        for r in hp.nsmallest(15, "score").itertuples():
            show(r.s1, "  ")
            show(r.m, "    == ")
            print()

    # ------------------------------------------------------------------ 7
    if want(7):
        section("7. FREQUENT TOKENS (train, all sources) -- legal suffixes & address vocabulary")
        tr = A[A.split == "train"]
        for c, gc in tr.groupby("country"):
            names = [norm(x) for x in gc.business_name]
            print(f"\n########## {c}")
            print("  name tokens   :", fmt(Counter(t for x in names for t in x.split()).most_common(40)))
            print("  last name tok :", fmt(Counter(x.split()[-1] for x in names if x).most_common(25)))
            print("  addr tokens   :", fmt(Counter(t for x in gc.business_address for t in norm(x).split()
                                                   if not t.isdigit()).most_common(40)))

    # ------------------------------------------------------------------ 8
    if want(8):
        section("8. BLOCKING PROBES (train: sampled S1 vs full S2+S3 pool)")
        probe = s1.sample(min(len(s1), args.probe_n), random_state=args.seed).reset_index(drop=True)
        pool_arr = pool.entity_id.values
        gt8 = gt[gt.source1_entity_id.isin(set(probe.entity_id))]
        GTM8 = {a: [t.strip() for t in m.split(",") if t.strip()]
                for a, m in zip(gt8.source1_entity_id, gt8.matched_entity_ids)}
        n_true_probe = sum(len(GTM8.get(e, [])) for e in probe.entity_id)
        print(f"probe S1 records={len(probe):,} (true pairs among them={n_true_probe:,}); pool={len(pool):,}")
        pos_index = pd.Index(pool_arr)
        true_idx = [pos_index.get_indexer(GTM8.get(e, [])) for e in probe.entity_id]  # -1 = not in pool

        views = ["name", "name+addr"] if TFIDF_OK else []
        probe_tokens = frozenset().union(*(toks(n) for n in probe.business_name))
        names, addrs = pool.business_name.tolist(), pool.business_address.tolist()
        codes = {c: i for i, c in enumerate(sorted(set(pool.country) | set(s1.country)))}
        pcode = pool.country.map(codes).values.astype(np.int64)
        qcode = probe.country.map(codes).values.astype(np.int64)

        specs, scorers, Xq = {}, {}, {}
        if views:
            fitted = fit_vocab(args.workers, names, addrs, views, args.chunk)
            _, ptexts = _view_texts(probe.business_name.tolist(), probe.business_address.tolist(), views)
            for v in views:
                vocab, idf = fitted[v]
                Xq[v], used = query_matrix(vocab, idf, ptexts[v])
                specs[v] = (vocab, idf, used)
                scorers[v] = TopK(Xq[v], 100, qcode)
            del fitted
        postings = defaultdict(list)

        def on_chunk(start, mats, post):
            for v, M in mats.items():
                scorers[v].add(M, start, pcode[start : start + M.shape[0]])
            for t, arr in post.items():
                postings[t].append(arr)

        stream_pool(args.workers, names, addrs, specs, probe_tokens, args.chunk, on_chunk)
        postings = {t: np.concatenate(L) for t, L in postings.items()}
        del names, addrs

        sub("8a. Inverted index on NAME tokens, dropping tokens with doc-freq above a cap")
        for frac in (0.001, 0.005, 0.02):
            cap = max(20, int(frac * len(pool)))
            sizes, hit, tot, empty = [], 0, 0, 0
            for j, e in enumerate(probe.entity_id):
                kept = [t for t in toks(probe.business_name[j]) if 0 < len(postings.get(t, ())) <= cap]
                if not kept:
                    empty += 1
                cand = np.unique(np.concatenate([postings[t] for t in kept])) if kept else np.empty(0, np.int64)
                sizes.append(len(cand))
                T = true_idx[j]
                if len(T):
                    hit += int(np.isin(T, cand).sum())
                    tot += len(T)
            s = np.array(sizes)
            print(f"  df cap={cap:>6} ({frac:.1%} of pool): pair recall={hit / max(tot, 1):.3f} | cand/entity "
                  f"mean={s.mean():.1f} p50={np.median(s):.0f} p90={np.percentile(s, 90):.0f} max={s.max()} "
                  f"| no usable token={empty / len(probe):.2%}")

        if views:
            sub("8b. TF-IDF char_wb n-grams (2-4), exact nearest neighbours: pair recall@k")
            Ks = [1, 3, 5, 10, 20, 50, 100]
            is_single = np.array([len(GTM8.get(e, [])) == 0 for e in probe.entity_id])
            for vname in views:
                for filt in (False, True):
                    idx, sc = scorers[vname].result(filt)
                    rec = []
                    for k in Ks:
                        hit = tot = 0
                        for j, e in enumerate(probe.entity_id):
                            T = GTM8.get(e, [])
                            if T:
                                got = set(pool_arr[idx[j, :k]])
                                hit += sum(t in got for t in T)
                                tot += len(T)
                        rec.append(hit / max(tot, 1))
                    print(f"  {vname:<9} country_filter={str(filt):<5}: "
                          + "  ".join(f"R@{k}={r:.3f}" for k, r in zip(Ks, rec)))
                    if vname == "name+addr" and filt:
                        KEY["TF-IDF name+addr (country-filtered) recall@10 / @50"] = f"{rec[3]:.3f} / {rec[5]:.3f}"
                        sub("8c. Singleton detection signal: top-1 similarity (name+addr, country-filtered)")
                        gap = sc[:, 0] - sc[:, 1] if sc.shape[1] > 1 else sc[:, 0]
                        print(f"  top1 score  singletons    : {q(sc[is_single, 0])}")
                        print(f"  top1 score  non-singletons: {q(sc[~is_single, 0])}")
                        print(f"  top1-top2   singletons    : {q(gap[is_single])}")
                        print(f"  top1-top2   non-singletons: {q(gap[~is_single])}")
                        top1_true = [pool_arr[idx[j, 0]] in set(GTM8.get(e, []))
                                     for j, e in enumerate(probe.entity_id) if not is_single[j]]
                        print(f"  non-singletons whose top-1 neighbour is a true match: {np.mean(top1_true):.3f}")
            del scorers, Xq
        else:
            print("  (scikit-learn or torch not installed -- TF-IDF probe skipped)")

    # ------------------------------------------------------------------ 9
    if want(9):
        section("9. TEST SET & UNSEEN COUNTRIES")
        del s1, s2, s3, pool, gt
        for s in (1, 2, 3):
            F.pop(("train", s), None)
            if ("test", s) not in F:
                F[("test", s)] = load(os.path.join(args.data_dir, "test", f"test_source{s}.tsv"))
        te_parts = [F[("test", s)].assign(src=f"S{s}") for s in (1, 2, 3) if F[("test", s)] is not None]
        if te_parts:
            te = pd.concat(te_parts, ignore_index=True)
            print("test records by source x country:")
            print(pd.crosstab(te.src, te.country).to_string())
            for s in (1, 2, 3):
                if F[("test", s)] is not None:
                    print(f"  test/train size ratio S{s}: {len(F[('test', s)]) / n_train[s]:.2f}")
            new = sorted(set(te.country) - train_countries)
            print("countries in test but not in train:", new)
            KEY["unseen test countries"] = new
            for c in new:
                gc = te[te.country == c]
                for src in ("S1", "S2", "S3"):
                    gg = gc[gc.src == src]
                    print(f"\n  {c} {src} samples ({len(gg):,} records):")
                    for r in gg.sample(min(len(gg), 10 if src == "S1" else 6), random_state=args.seed).itertuples():
                        print(f"    {r.entity_id} {short(r.business_name, 55)}  |  {short(r.business_address, 100)}")
                names = [_norm(x) for x in gc.business_name]
                print(f"\n  {c} name tokens   :", fmt(Counter(t for x in names for t in x.split()).most_common(40)))
                print(f"  {c} last name tok :", fmt(Counter(x.split()[-1] for x in names if x).most_common(25)))
                print(f"  {c} addr tokens   :", fmt(Counter(t for x in gc.business_address for t in _norm(x).split()
                                                         if not t.isdigit()).most_common(40)))
            del te

            if TFIDF_OK and all(F[("test", s)] is not None for s in (1, 2, 3)):
                sub("9b. Test: top-1 similarity per country (name+addr TF-IDF, country-filtered) -- compare with 8c")
                t1 = F[("test", 1)]
                tpool = pd.concat([F[("test", 2)], F[("test", 3)]], ignore_index=True)
                tprobe = t1.sample(min(len(t1), args.probe_n * 2), random_state=args.seed).reset_index(drop=True)
                codes = {c: i for i, c in enumerate(sorted(set(tpool.country) | set(t1.country)))}
                names, addrs = tpool.business_name.tolist(), tpool.business_address.tolist()
                tpcode = tpool.country.map(codes).values.astype(np.int64)
                fitted = fit_vocab(args.workers, names, addrs, ["name+addr"], args.chunk)
                vocab, idf = fitted.pop("name+addr")
                _, ptexts = _view_texts(tprobe.business_name.tolist(), tprobe.business_address.tolist(), ["name+addr"])
                Xq9, used = query_matrix(vocab, idf, ptexts["name+addr"])
                scorer = TopK(Xq9, 2, tprobe.country.map(codes).values, filters=(True,))
                stream_pool(args.workers, names, addrs, {"name+addr": (vocab, idf, used)}, None, args.chunk,
                            lambda start, mats, post: scorer.add(mats["name+addr"], start,
                                                                 tpcode[start : start + mats["name+addr"].shape[0]]))
                _, sc = scorer.result(True)
                for c in sorted(tprobe.country.unique()):
                    m = (tprobe.country == c).values
                    print(f"  {c:<10} n={m.sum():>5}  top1={q(sc[m, 0])}  top1-top2={q(sc[m, 0] - sc[m, 1])}")
        else:
            print("  test files not found -- skipped")

    # ------------------------------------------------------------------ 10
    section("10. HEADLINE NUMBERS")
    for k, v in KEY.items():
        print(f"  {k}: {v}")
    print(f"\nDone in {time.time() - T0:.0f}s. Report saved to {args.out}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
