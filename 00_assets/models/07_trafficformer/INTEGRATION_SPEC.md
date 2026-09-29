# TrafficFormer (IEEE S&P 2025) pipeline integration spec

> Original repo (bundled UER-py fork) + paper analysis. Written 2026-07-26.

## 1. pcap → input conversion (`data_generation/finetuning_data_gen.py`)

### Session splitting/filtering (original)
- SplitCap bidirectional 5-tuple sessions. Label = directory name.
- Filters: drop pcaps <2KB, drop flows <3 packets, drop classes <10 flows, at most 500 flows per class, **drop IPv6 flows entirely**, re-split pcaps >10MB into 300-second chunks.

### Byte extraction/tokenization
- **First 5 packets**, per packet **64 bytes from the first byte of the IP header** (start_index=28 hex chars — skips 14B Ethernet).
  Note: the function default start_index=76 is the ET-BERT setting; the TrafficFormer paper setting is 28.
- ET-BERT-style sliding-window byte-bigram (stride 1B): 64B → 63 tokens (4-hex-digit).
- Prepend the string `"[SEP] "` to each packet and concatenate → text_a.
- **Important pitfall (must reproduce)**: the BasicTokenizer of BertTokenizer splits `[SEP]` into `[`,`sep`,`]`, encoding it as **[UNK]×3** (verified empirically). 3+63=66 tokens per packet, 331 including [CLS] → truncated to seq_length 320. This behavior must be kept to match the original.

### Anonymization (both train/test, right before tokenization)
- `random_ip_port()`: random client/server IPs (direction-consistent), random 16-bit ports.
- `random_tcp_ts_option()`: random 32-bit TCP Timestamp base value, inter-packet delta preserved.
- `random_tls_randomtime()`: randomizes TLS1.2 Hello gmt_unix_time.
- No checksum recomputation.

### RIFA augmentation (`enhance_based_tsv`, **train TSV only**)
- Directly modifies TSV bigram tokens at fixed offsets (IPID=4, srcIP=12–16, sport=20, dport=22, seq=24, ack=28).
- Randomly re-initializes IPID/IP/port/seq/ack initial values (increments preserved), replicates with factor=5 then shuffles. epoch 20→4 in this case.

## 2. Fine-tune input format
- TSV, header `label\ttext_a`. label=integer (from 0), text_a=`[SEP] 4500 0000 ...` token string.
- train_dataset.tsv / valid_dataset.tsv / test_dataset.tsv (8:1:1 stratified).

## 3. Fine-tune entry point
```
python3 fine-tuning/run_classifier.py --vocab_path models/encryptd_vocab.txt \
    --train_path train_dataset.tsv --dev_path valid_dataset.tsv --test_path test_dataset.tsv \
    --pretrained_model_path pretrain_model.bin --output_model_path models/finetuned_model.bin \
    --epochs_num 4 --earlystop 4 --batch_size 128 --embedding word_pos_seg \
    --encoder transformer --mask fully_visible --seq_length 320 --learning_rate 6e-5
```
- vocab: `models/encryptd_vocab.txt` (60,005 tokens). config: BERT-base (`models/bert/base_config.json`).
- lr 6e-5, batch 128, seq 320, epoch 4(augmented)/20(non-augmented), earlystop dev macro-F1, adamw, warmup 0.1, seed 7.
- Special tokens: [PAD]=0, [SEP]=1, [CLS]=2, [UNK]=3, [MASK]=4.

## 4. Pre-trained checkpoint
- Required. Not included in the repo. Google Drive: https://drive.google.com/file/d/1pR6ZaWE7MWFDQWiF4LDzSyjSq0Gj3kV7/view
- **Compatible with UER-format ET-BERT checkpoints** (same BERT-base, vocab 60,005, loaded with strict=False) → usable as fallback.
- Location: `00_assets/models/07_trafficformer/pretrain_model.bin`

## 5. Dependencies
- Bundled UER fork (uer/). torch==2.0.1, scapy==2.5.0, flowcontainer==7.2 (requires tshark), sklearn 1.3.1.
- Pitfalls in the original scripts: hard-coded temp path (/mnt/data/zgm/...), global `_category` undefined, README signature mismatch.

## 6. Preprocessing verification view (for the table in paper Sec. 2.2)
| Item | Present |
|---|---|
| Label error verification | None (label = directory name as is) |
| Removal of task-irrelevant traffic | None (apart from SplitCap by-products) |
| Noise session refinement | Only size/count-based structural filters (<2KB, <3pkt, <10flow/cls, IPv6 removal) |
| Shortcut prevention | Yes — IP/port/TCP TS/TLS time randomization + RIFA (core contribution) |

## 7. dpkt session → sample pseudocode
```python
def make_trafficformer_sample(pkts_l3: list[bytes]) -> str | None:
    if len(pkts_l3) < 3: return None
    if pkts_l3[0][0] >> 4 == 6: return None          # drop IPv6
    # anonymization: random IP/port (direction-consistent), random TCP TS with delta preserved, random TLS time
    parts = []
    for pkt in pkts_l3[:5]:
        h = pkt[:64].hex()
        toks = [h[i*2:i*2+4] for i in range(len(h)//2 - 1)]
        parts.append("[SEP] " + " ".join(toks))
    return " ".join(parts) + " "
```
