# NetMamba (ICNP 2024) pipeline integration spec

> Based on analysis of the original repo + paper (§V, Table III). Written 2026-07-26.

## 1. Input representation (paper Table III = matches code)
M=5 packets, Nh=80 header bytes, Np=240 payload bytes, 1600 bytes total.
StrideEmbed(Conv1d kernel=4, stride=4) → 400 tokens + 1 cls = sequence of 401.
Stored as a 40×40 uint8 PNG, but semantically a 1-D byte sequence of 1600.

### Packet processing (`dataset/dataset_common.py` `raw_packet_to_string`)
1. Strip Ethernet, start from L3 (IP). No IP or parse failure → replaced by an all-zero 640B(?) slot (not skipped).
2. **Anonymization: ip.src = ip.dst = "0.0.0.0"** (IP only. Ports kept. No checksum recomputation).
3. header = IP header + L4 header (scapy Raw-stripping approach), payload = Raw.
4. header 80B / payload 240B crop + 0x00 padding. 320B concatenated per packet; if fewer than 5, zero-padded to 1600B.
- augment mode (sliding window 5, stride 1) is only for CrossPlatform·ISCXTor — not used by default.

## 2. Normalization
Grayscale → ToTensor(÷255) → Normalize(0.5, 0.5) → [-1,1]. Equivalent to `(uint8/255-0.5)/0.5`.

## 3. Fine-tune
- Input: ImageFolder — **all 3 splits train/valid/test are required**. Labels = folder names in lexicographic order.
- Entry point: `src/fine-tune.py --blr 2e-3 --epochs 120 --nb_classes N --finetune <ckpt> --data_path <dir> --model net_mamba_classifier --no_amp`
- Model: `src/models_net_mamba.py` `net_mamba_classifier` (embed 256, depth 4, stride 4, byte_length 1600).
- Hyperparameters: batch 64, epochs 120 (paper), AdamW, wd 0.05, layer_decay 0.75, warmup 20ep, min_lr 1e-6, smoothing 0.1, drop_path 0.1, **--no_amp recommended**.
- Saves the best checkpoint and runs the test evaluation itself (`checkpoint-best.pth`).

## 4. Pretrained checkpoint
- HuggingFace `wangtz/NetMamba` → `pre-train.pth` (26.2MB). Not included in the repo.
- Loading: `checkpoint['model']` → remove head → interpolate_pos_embed → strict=False → head trunc_normal_(2e-5).
- Location: `00_assets/models/06_netmamba/pre-train.pth`

## 5. Dependencies (moderately hard)
- Python 3.10.13 / torch 2.1.1+cu121 / torchvision 0.16.1 / timm 0.4.12 / triton 2.1.0 / causal-conv1d 1.1.0
- **mamba_ssm 1.1.1 bundled** (`mamba-1p1p1/`, `pip install -e .`) — source build with nvcc (CUDA≥11.6) required. Cannot run on CPU.
- scapy is for preprocessing only (not needed in our adapter).

## 6. Pseudocode for dpkt session → sample conversion
```python
def netmamba_sample(pkts: list[bytes]) -> np.ndarray:  # uint8 (40,40)
    NH, NP, M = 80, 240, 5
    out = bytearray()
    for raw in pkts[:M]:
        hdr, pay = split_l3(raw)          # IP header + L4 header / L4 payload
        # original default behavior = zero-mask IP addresses only (this framework: reproduced with --ip_mask)
        out += hdr[:NH].ljust(NH, b"\x00")
        out += pay[:NP].ljust(NP, b"\x00")
    out = bytes(out).ljust(M * 320, b"\x00")
    return np.frombuffer(out, dtype=np.uint8).reshape(40, 40)
```
- The original splits by scapy Raw, so parsing upper layers such as DNS has a quirk where payload ends up in the header.
  Splitting by IHL+L4 offset (our approach) is semantically correct and identical for TCP/UDP+TLS.
- Non-IP packets become 0-slots rather than being skipped (when reproducing the original).
