#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_make_field_map.py — builds field_id for Illusion 5-B (byte→field SHAP) (based on tshark PDML)
─────────────────────────────────────────────────────────────────────────────
Matching the 06 output (x_data) of the 5 byte-based models (2dcnn/etbert/yatc/netmamba/trafficformer)
with 'the same layout, same masking (ip+port), same L2 removal', builds a field_id per byte (token) position.
(rf/xgboost need no field_id since feature_columns.json already holds 1,213 feature names)

Parser = tshark (PDML): for each byte, the 'most specific' field name (ip.id, tcp.window_size,
tls.handshake.extensions_server_name, dns.qry.name, http.request.uri …).
Datasets using non-standard ports can be force-decoded (-d) per class (task3) via FORCE_DECODING_RULES (empty for the 8 public datasets).

Layout/constants (confirmed against the original shaping code):
  2dcnn        : L3 concatenation 784B → 28×28, masking/padding value = 257 (shaping_2dcnn.MASK_VAL)
  etbert       : L3 concatenation → 128 bigram tokens, field of token i = leading byte i
  yatc/netmamba: 5 packets×(header80=ip_pos..pay_pos + payload240)=1600 → 40×40, masking/padding=0
  trafficformer: 5 packets×([SEP]+63 bigram, first 64B of L3), IP/port/TS randomized → no byte verify (fields only)
  masking = by field name (ip.src/dst, ipv6.src/dst, *.srcport/dstport → "#masked")
           = same bytes as dpkt addr_span/port_span

whole mode (5 attack datasets): instead of rescanning the monolithic pcap per session, the streams needed per pcap are
  grouped for a 1-pass extraction (-Y "tcp.stream==a||…" -w subset) → frame→stream map from the subset
  → parse PDML once → distribute per stream. (scans = 1 to a few per pcap)

--verify : compares tshark-reconstructed bytes with 06 x_data at every position (match rate). 2dcnn/yatc/netmamba only.
--report : field coverage report (field_report.csv) — ratio of specific fields over all bytes,
           top fields, detection of key fields (ip.id/tls sni/dns.qry/http/ssh/mysql/mongo).

Output: 01_dataset/{ds}/{variant}/{model}/{split}_{mode}/field_id.npy  (aligned to files.csv row order)
      01_dataset/{ds}/{variant}/field_vocab.csv , field_report_{split}_{mode}.csv

Usage (server):
  python3 08_make_field_map.py --dataset vpn16 --variant strat --split test --workers 24 --verify --report
"""
import argparse, csv, os, subprocess, sys, tempfile
from collections import Counter, defaultdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

CODE = Path(__file__).resolve().parent
WORK = CODE.parent
LIST_BASE = WORK / "01_dataset"
sys.path.insert(0, str(CODE))
from lib import datasets as ds
from lib.parser import tshark_pdml as tp
from lib.parser.field_vocab import FieldVocab, PAD_FIELD, SEP_FIELD

VARIANT_FL = {"full": "00_filelist", "sizectrl": "00_filelist_sm", "strat": "00_filelist_strat"}
DATASET_WHOLE = {"ustc16", "cic17", "cic18", "iot23"}
MODELS = ["2dcnn", "etbert", "yatc", "netmamba", "trafficformer"]

# ── Layout constants (confirmed against the shaping_* originals) ──
N_2DCNN, MASK2D = 784, 257                 # shaping_2dcnn: MASK_VAL=257, /256 normalization
ETB_TOK = 128
MFR_NPKT, MFR_H, MFR_P = 5, 80, 240; MFR_L = MFR_NPKT * (MFR_H + MFR_P)   # masking/padding=0
TF_NPKT, TF_SEL = 5, 64
PKT_CAP = 60                               # max packets parsed per session (enough for 784B/1600B)

MASK_IP = {"ip.src", "ip.dst", "ipv6.src", "ipv6.dst"}
MASK_PORT = {"tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport"}
GENERIC = {"payload", "ip", "ipv6", "tcp", "udp", "data", "tcp.payload",
           "udp.payload", "data.data", PAD_FIELD, SEP_FIELD}
KEY_FIELDS = ["ip.id", "ip.ttl", "ip.dsfield", "ip.checksum", "tcp.window_size",
              "tcp.seq", "tls.handshake.extensions_server_name", "dns.qry.name",
              "http.request.uri", "http.host", "ssh.protocol", "mysql", "mongo"]

# ── Per-class (task3) forced Decode As: {class name: [(display filter, protocol), ...]} ──
FORCE_DECODING_RULES = {
    "AdminPanelDiscovery":   [("tcp.port==8080", "tls"), ("tcp.port==8888", "tls")],
    "BruteForceMongoDB":     [("tcp.port==27017", "mongo")],
    "BruteForceMySQL":       [("tcp.port==3306",  "mysql")],
    "BruteForceSSH":         [("tcp.port==2222",  "ssh")],
    "BruteForceWEB":         [("tcp.port==8080",  "tls")],
    "DirectNetworkFloodSSH": [("tcp.port==2222",  "ssh")],
    "ForbiddenBypass":       [("tcp.port==8080", "tls"), ("tcp.port==8888", "tls")],
    "GETFlood":              [("tcp.port==8080",  "tls")],
    "HTTPCC":                [("tcp.port==8080",  "tls")],
    "JWT":                   [("tcp.port==8888",  "tls")],
    "SlowLoris":             [("tcp.port==8080",  "tls")],
    "SQLInjection":          [("tcp.port==443",   "tls")],
    "SSTI":                  [("tcp.port==8080",  "tls")],
}
TSHARK_OPTS = ["-o", "tcp.desegment_tcp_streams:TRUE",
               "-o", "tls.desegment_ssl_records:TRUE"]
STREAMS_PER_PASS = 250                     # whole mode: streams per pass (filter length limit)


def _is_ip_pkt(p):
    """The dpkt path only puts packets parsed as IP into samples → same filter on the tshark side."""
    return p.ip_pos < len(p.raw) and p.labels[p.ip_pos].startswith(("ip", "ipv6"))


def _mask(lbl, val, mask_val):
    if lbl in MASK_IP or lbl in MASK_PORT:
        return lbl + "#masked", mask_val
    return lbl, val


# ── Per-model (bytes, fields, pkt_ids) builders — assembled identically to shaping ──
#    pkt_ids = packet number of each position (1-based, padding=0).
#    The 98 (SII) style 'k-th byte of a field' (sni-1, sni-2 …) is derived at analysis time
#    via byte_run_matrix(field_id, pkt_id) — required to avoid merging runs across packet boundaries.
def _l3_stream(pkts, mask_val):
    for pi, p in enumerate(pkts, 1):
        for j in range(p.ip_pos, len(p.raw)):
            lbl, val = _mask(p.labels[j], p.raw[j], mask_val)
            yield lbl, val, pi


def b_2dcnn(pkts):
    xb, fb, pb = [], [], []
    for lbl, val, pi in _l3_stream(pkts, MASK2D):
        xb.append(val); fb.append(lbl); pb.append(pi)
        if len(xb) >= N_2DCNN:
            break
    pad = N_2DCNN - len(xb[:N_2DCNN])
    xb = xb[:N_2DCNN] + [MASK2D] * pad
    fb = fb[:N_2DCNN] + [PAD_FIELD] * pad
    pb = pb[:N_2DCNN] + [0] * pad
    return xb, fb, pb


def b_mfr(pkts):
    xb, fb, pb = [], [], []
    for pi, p in enumerate(pkts[:MFR_NPKT], 1):
        hx, hf = [], []
        for j in range(p.ip_pos, min(p.pay_pos, len(p.raw))):
            lbl, val = _mask(p.labels[j], p.raw[j], 0); hx.append(val); hf.append(lbl)
        nh = len(hx[:MFR_H])
        hx = hx[:MFR_H] + [0] * (MFR_H - nh)
        hf = hf[:MFR_H] + [PAD_FIELD] * (MFR_H - nh)
        hp = [pi] * nh + [0] * (MFR_H - nh)
        px, pf = [], []
        for j in range(p.pay_pos, len(p.raw)):
            lbl, val = _mask(p.labels[j], p.raw[j], 0); px.append(val); pf.append(lbl)
        np_ = len(px[:MFR_P])
        px = px[:MFR_P] + [0] * (MFR_P - np_)
        pf = pf[:MFR_P] + [PAD_FIELD] * (MFR_P - np_)
        pp = [pi] * np_ + [0] * (MFR_P - np_)
        xb += hx + px; fb += hf + pf; pb += hp + pp
    pad = MFR_L - len(xb)
    xb += [0] * pad; fb += [PAD_FIELD] * pad; pb += [0] * pad
    return xb[:MFR_L], fb[:MFR_L], pb[:MFR_L]


def b_etbert(pkts):
    fb, pb = [], []
    for lbl, _, pi in _l3_stream(pkts, 0):
        fb.append(lbl); pb.append(pi)
        if len(fb) > ETB_TOK + 1:
            break
    bf = fb[:-1] if len(fb) >= 2 else []          # bigram i = bytes i,i+1 → field of the leading byte
    bp = pb[:-1] if len(pb) >= 2 else []
    pad = ETB_TOK - len(bf[:ETB_TOK])
    bf = bf[:ETB_TOK] + [PAD_FIELD] * pad
    bp = bp[:ETB_TOK] + [0] * pad
    return None, bf, bp


def b_trafficformer(pkts):
    """Assembled identically to shaping_trafficformer._to_text (fixed 2026-09-01):
    per packet [SEP] + (selected bytes−1) bigrams — short packets are concatenated without block padding,
    'variable length as is'. (The former fixed 64-tokens/packet layout diverged from the actual
    token stream, leaking attribution mass into <pad> — verified empirically)"""
    fb, pb = [], []
    for pi, p in enumerate(pkts[:TF_NPKT], 1):
        seg = []
        for j in range(p.ip_pos, min(p.ip_pos + TF_SEL, len(p.raw))):
            lbl, _ = _mask(p.labels[j], p.raw[j], 0); seg.append(lbl)
        fb.append(SEP_FIELD); pb.append(pi)
        nb = max(0, len(seg) - 1)                  # number of bigrams = selected bytes − 1
        fb += seg[:nb]                             # field of bigram i = leading byte i
        pb += [pi] * nb
    L = TF_NPKT * TF_SEL
    fb = fb[:L] + [PAD_FIELD] * max(0, L - len(fb))
    pb = pb[:L] + [0] * max(0, L - len(pb))
    return None, fb, pb


BUILD = {"2dcnn": b_2dcnn, "yatc": b_mfr, "netmamba": b_mfr,
         "etbert": b_etbert, "trafficformer": b_trafficformer}
FLEN = {"2dcnn": N_2DCNN, "etbert": ETB_TOK, "yatc": MFR_L, "netmamba": MFR_L,
        "trafficformer": TF_NPKT * TF_SEL}
FSHAPE = {"2dcnn": (28, 28), "yatc": (40, 40), "netmamba": (40, 40)}


# ═══════════ Session packet acquisition ═══════════
_W = {}


def _init_session(dataset, models):
    _W["models"] = models
    _W["pidx"] = tp.build_pcap_index(ds.session_dir(dataset))


def _work_session(task):
    i, fn, sid, gk = task
    pcap = _W["pidx"].get(fn)
    if pcap is None:
        return i, None
    deco = FORCE_DECODING_RULES.get(gk)
    pkts = tp.read_session_packets(pcap, cap=PKT_CAP, decode_as=deco,
                                   tshark_opts=TSHARK_OPTS)
    if not pkts:
        return i, None
    pkts = [p for p in pkts if _is_ip_pkt(p)]
    if not pkts:
        return i, None
    return i, {m: BUILD[m](pkts) for m in _W["models"]}


WHOLE_ERR_LOG = "/tmp/fmap_whole_err.log"


def _whole_err(msg):
    try:
        with open(WHOLE_ERR_LOG, "a", encoding="utf-8") as f:
            f.write(msg.rstrip() + "\n")
    except OSError:
        pass


def _work_whole_chunk(task):
    """(pcap path, deco, [(idx,fn,sid,proto,stream)…]) → {idx: {model:(xb,fb)}}
    Monolithic pcap 1-pass: -Y multi-stream filter -w subset → frame→stream map → PDML once."""
    pcap, deco, models, items = task
    filt = "||".join(f"{p}.stream=={s}" for _, _, _, p, s in items)
    tmp = tempfile.NamedTemporaryFile(suffix=".pcap", delete=False); tmp.close()
    out = {}
    try:
        # in one pass, write the subset + output stream numbers 'relative to the original file'.
        #   -F pcap  : the -w default (pcapng) cannot be read by dpkt.pcap.Reader, giving 0 packets
        #   -T fields: stream numbers are reassigned per file, so re-reading the subset gives 0,1,2…
        #              → they must be extracted in this pass, which reads the original, to stay aligned
        r = subprocess.run(["tshark", "-r", str(pcap), "-Y", filt,
                            "-w", tmp.name, "-F", "pcap",
                            "-T", "fields", "-e", "tcp.stream", "-e", "udp.stream"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=1800)
        _sz = os.path.getsize(tmp.name)
        if _sz <= 24:            # pcap header only = 0 packets extracted → real failure
            _whole_err(f"{pcap}: subset failed rc={r.returncode} size={_sz} "
                       f"stderr={r.stderr.decode('utf-8', 'replace')[:400]}")
            return out
        if r.returncode != 0:    # truncated pcap (CIC-2018 etc.): extracted part is valid → continue
            _whole_err(f"{pcap}: rc={r.returncode} (original pcap likely truncated) — "
                       f"continuing with extracted {_sz}B")
        frame_sid = []
        for line in r.stdout.decode("utf-8", "replace").splitlines():
            c = line.split("\t")
            t = c[0].strip() if c else ""
            u = c[1].strip() if len(c) > 1 else ""
            frame_sid.append(("tcp", t) if t else (("udp", u) if u else None))
        cap = max(2000, PKT_CAP * len(items))
        pkts = tp.read_session_packets(tmp.name, cap=cap, decode_as=deco,
                                       tshark_opts=TSHARK_OPTS)
        if not pkts:
            _whole_err(f"{pcap}: subset PDML parsed 0 packets "
                       f"(subset size={os.path.getsize(tmp.name)}, "
                       f"frame_sid={len(frame_sid)})")
            return out
        by_stream = defaultdict(list)
        for k, p in enumerate(pkts):
            if k < len(frame_sid) and frame_sid[k] and _is_ip_pkt(p):
                if len(by_stream[frame_sid[k]]) < PKT_CAP:
                    by_stream[frame_sid[k]].append(p)
        for idx, fn, sid, proto, stream in items:
            sp = by_stream.get((proto, stream))
            if sp:
                out[idx] = {m: BUILD[m](sp) for m in models}
        if not out:
            _whole_err(f"{pcap}: stream matches 0/{len(items)} — "
                       f"by_stream keys e.g. {list(by_stream)[:3]} vs "
                       f"requested e.g. {[(p, s) for _, _, _, p, s in items[:3]]}")
        return out
    except Exception as e:
        import traceback
        _whole_err(f"{pcap}: exception {e}\n{traceback.format_exc()}")
        return out
    finally:
        try: os.unlink(tmp.name)
        except OSError: pass


# ═══════════ verify / report ═══════════
def _verify_row(model, xrow, xb):
    if xb is None or xrow is None:
        return None
    flat = np.asarray(xrow).reshape(-1).astype(np.float64)
    xb = np.asarray(xb, dtype=np.float64)
    n = min(len(flat), len(xb))
    if model == "2dcnn":
        flat = np.rint(flat * 256.0)          # inverse of /256 normalization (incl. masking/padding=257)
    return float((xb[:n] == flat[:n]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", required=True)
    ap.add_argument("--model", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--variant", choices=["full", "sizectrl", "strat"], default="strat")
    ap.add_argument("--split", nargs="+", default=["test"], choices=["train", "test"])
    ap.add_argument("--mode", nargs="+", default=["noisy", "denoised"],
                    choices=["noisy", "denoised"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sample", type=int, default=0, help=">0: only the first N rows per split (for verification)")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if not tp.tshark_available():
        print("[FATAL] tshark not installed — apt install -y tshark"); return

    for dataset in a.dataset:
        whole = dataset in DATASET_WHOLE
        out_base = LIST_BASE / dataset / a.variant
        # inherit the existing vocab — so that when regenerating only some models, the field_id
        # id→name mapping of other models is not broken (load adds in saved order → ids preserved)
        _vp = out_base / "field_vocab.csv"
        vocab = FieldVocab.load(_vp) if _vp.exists() else FieldVocab()
        print(f"\n=== {dataset} ({a.variant}) · {'whole(batch extraction)' if whole else 'session'} ===",
              flush=True)
        for split in a.split:
            for mode in a.mode:
                sm = f"{split}_{mode}"
                # union of sessions (per-model files.csv; the same session is parsed only once)
                need = {}
                rows_by_model = {}
                for m in a.model:
                    fc = out_base / m / sm / "files.csv"
                    if not fc.exists():
                        continue
                    rows = list(csv.reader(open(fc, encoding="utf-8")))[1:]
                    if a.sample:
                        rows = rows[:a.sample]
                    rows_by_model[m] = rows
                    for r in rows:
                        need.setdefault((r[0], r[1]), r[2] if len(r) > 2 else "")
                if not rows_by_model:
                    print(f"  [{sm}] files.csv not found (06 not built) → skip"); continue
                keys = list(need)
                key2idx = {k: i for i, k in enumerate(keys)}
                store = {}
                if not whole:
                    tasks = [(i, fn, sid, need[(fn, sid)])
                             for i, (fn, sid) in enumerate(keys)]
                    with ProcessPoolExecutor(max_workers=a.workers,
                                             initializer=_init_session,
                                             initargs=(dataset, a.model)) as ex:
                        for i, res in ex.map(_work_session, tasks, chunksize=8):
                            if res:
                                store[i] = res
                else:
                    pidx = tp.build_pcap_index(ds.pcap_dir(dataset))
                    groups = defaultdict(list)          # (pcap, deco key) → items
                    skip = 0
                    for i, (fn, sid) in enumerate(keys):
                        pcap = pidx.get(fn)
                        if pcap is None or "_" not in sid:
                            skip += 1; continue
                        proto, stream = sid.split("_", 1)
                        if proto not in ("tcp", "udp"):
                            skip += 1; continue
                        gk = need[(fn, sid)]
                        deco = tuple(FORCE_DECODING_RULES.get(gk) or [])
                        groups[(str(pcap), deco)].append((i, fn, sid, proto, stream))
                    tasks = []
                    for (pcap, deco), items in groups.items():
                        for j in range(0, len(items), STREAMS_PER_PASS):
                            tasks.append((pcap, list(deco) or None, a.model,
                                          items[j:j + STREAMS_PER_PASS]))
                    print(f"  [{sm}] whole: pcap groups {len(groups)} · extraction passes {len(tasks)}"
                          f" · sessions {len(keys)} (skip={skip})", flush=True)
                    with ProcessPoolExecutor(max_workers=a.workers) as ex:
                        for res in ex.map(_work_whole_chunk, tasks):
                            store.update(res)

                # per-model save + verify + report
                for m, rows in rows_by_model.items():
                    arr = np.zeros((len(rows), FLEN[m]), dtype=np.uint16)
                    parr = np.zeros((len(rows), FLEN[m]), dtype=np.uint8)   # packet number (1-based, pad=0)
                    ok = miss = 0; vrates = []; fcount = Counter()
                    xdata = None
                    if a.verify and m in FSHAPE:
                        xp = out_base / m / sm / "x_data.npy"
                        xdata = np.load(xp, mmap_mode="r") if xp.exists() else None
                    for ri, r in enumerate(rows):
                        rec = store.get(key2idx.get((r[0], r[1])))
                        if not rec or m not in rec:
                            miss += 1; continue
                        xb, fb, pbk = rec[m]
                        arr[ri, :len(fb)] = [vocab.add(x) for x in fb[:FLEN[m]]]
                        parr[ri, :len(pbk)] = pbk[:FLEN[m]]
                        ok += 1
                        if a.report:
                            fcount.update(fb)
                        if xdata is not None and ri < len(xdata):
                            vr = _verify_row(m, xdata[ri], xb)
                            if vr is not None:
                                vrates.append(vr)
                    od = out_base / m / sm; od.mkdir(parents=True, exist_ok=True)
                    outarr = arr.reshape(len(rows), *FSHAPE[m]) if m in FSHAPE else arr
                    outpkt = parr.reshape(len(rows), *FSHAPE[m]) if m in FSHAPE else parr
                    np.save(od / "field_id.npy", outarr)
                    np.save(od / "pkt_id.npy", outpkt)      # for byte_run_matrix(fid,pkt)
                    msg = f"  [{m}/{sm}] field_id {outarr.shape} ok={ok} miss={miss}"
                    if vrates:
                        msg += f" · verify match {np.mean(vrates)*100:.2f}% (n={len(vrates)})"
                    if a.report and fcount:
                        tot = sum(fcount.values())
                        spec = sum(c for f, c in fcount.items() if f not in GENERIC
                                   and not f.endswith("#masked"))
                        msg += f" · specific fields {spec/tot*100:.1f}%"
                    print(msg, flush=True)
                    if a.report and fcount:
                        rp = out_base / f"field_report_{m}_{sm}.csv"
                        with open(rp, "w", newline="", encoding="utf-8-sig") as fp:
                            w = csv.writer(fp); w.writerow(["field", "bytes", "ratio(%)"])
                            tot = sum(fcount.values())
                            for f, c in fcount.most_common():
                                w.writerow([f, c, f"{c/tot*100:.3f}"])
                        hit = [k for k in KEY_FIELDS
                               if any(k in f for f in fcount)]
                        print(f"      key fields detected: {hit}", flush=True)
        if len(vocab) > 1:                       # save only if something was processed
            out_base.mkdir(parents=True, exist_ok=True)
            vocab.save(out_base / "field_vocab.csv")
            print(f"  field_vocab={len(vocab)} → {out_base/'field_vocab.csv'}", flush=True)


if __name__ == "__main__":
    main()
