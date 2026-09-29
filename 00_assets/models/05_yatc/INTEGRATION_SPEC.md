# YaTC (AAAI 2023) pipeline integration spec

> Analysis of 13 files of the original repo + paper (Section 3.1). Written 2026-07-26.

## 1. pcap → model input conversion (`data_process.py`)

| Item | Value |
|---|---|
| Packets used | first 5 of the flow |
| Header | from the IP layer (IP header + L4 header + options), 80 bytes (excess truncated / 0x00 padding) |
| Payload | L4 payload, 240 bytes (excess truncated / 0x00 padding) |
| Missing packets | filled with all-zero 320B packets |
| Serialization | per packet header(80B)+payload(240B) concatenated in order → 5×320B = 1600 bytes |
| Imaging | (40, 40) uint8 grayscale PNG |

- MFR structure: 8 rows per packet = header 2 rows (80B) + payload 6 rows (240B), 5 packets stacked vertically → 40×40.
- The original code is IPv4-only (crashes on a `packet['IP']` exception).
- **Paper vs code mismatch**: the paper specifies "port zeroing + IP randomization (direction preserved)", but the public code does not implement it.
  → In this framework it is handled via `--ip_mask`/`--port_mask` (masked bytes=0x00).

## 2. Normalization (fine-tune.py build_dataset)
Grayscale(1ch) → ToTensor() → Normalize(mean=[0.5], std=[0.5]) → [-1,1], shape (1,40,40).
Equivalent: `x = (uint8/255 - 0.5) / 0.5`

## 3. Fine-tune
- Input: torchvision ImageFolder (`train/{cls}/*.png`, `test/{cls}/*.png`), label = lexicographic index of the folder name.
- Entry point: `fine-tune.py --blr 2e-3 --epochs 200 --data_path ... --nb_classes N`
- Model: `models_YaTC.TraFormer_YaTC` (inherits timm 0.3.2 VisionTransformer; img 40, patch 2, embed 192, depth 4, heads 16). Per-packet attention → row pooling → flow-level attention → mean of 5 packet cls → head.
- Hyperparameters: batch 64, epochs 200, AdamW + layer-wise lr decay(0.75), lr=blr×batch/256 (blr 2e-3), warmup 20ep + cosine, wd 0.05, drop_path 0.1, label smoothing 0.1, AMP.
- Checkpoint loading: `torch.load(...)['model']`, head removed then strict=False, head re-initialized with trunc_normal_(std=2e-5).
- **fine-tune.py has no model-saving logic** → saving is done in our adapter (train_yatc.py).
- The macro_f1 in `engine.py evaluate()` is actually weighted F1.

## 4. Pre-trained checkpoint
- Required (scratch is possible without it, but not at paper performance). Not included in the repo.
- Google Drive: https://drive.google.com/file/d/1wWmZN87NgwujSd2-o5nm3HaQUIzWlv16/view
- Location: `00_assets/models/05_yatc/output_dir/pretrained-model.pth`

## 5. Dependency pitfalls
- `assert timm.__version__ == "0.3.2"` hardcoded (fine-tune.py, pre-train.py). timm 0.3.2 raises a `torch._six` error on torch≥1.10 → pin torch 1.9 or patch the assert/helper.
- `util/pos_embed.py:50` `dtype=np.float` → error on numpy≥1.24. Needs a patch.
- models_YaTC.py imports skimage (unused, but ImportError if missing).

## 6. dpkt session → sample conversion pseudocode
```python
def session_to_mfr(pkts: list[bytes]) -> np.ndarray:   # (40,40) uint8
    HEADER_LEN, PAYLOAD_LEN, N_PKT = 80, 240, 5
    rows = b''
    for buf in pkts[:N_PKT]:
        # header = IP header(+options)+L4 header(+options), payload = L4 payload
        header, payload = split_l3(buf)
        rows += header[:HEADER_LEN].ljust(HEADER_LEN, b'\x00')
        rows += payload[:PAYLOAD_LEN].ljust(PAYLOAD_LEN, b'\x00')
    rows = rows.ljust(N_PKT * 320, b'\x00')
    return np.frombuffer(rows, dtype=np.uint8).reshape(40, 40)
```
