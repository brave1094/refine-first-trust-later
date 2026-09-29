#!/usr/bin/env bash
# 00_assets/models/setup_models.sh — Step 2: fetch the third-party model code and pre-trained weights.
#
#   bash 00_assets/models/setup_models.sh            # all four upstream repositories + weights
#   bash 00_assets/models/setup_models.sh --no-weights
#
# The upstream code is NOT redistributed in this repository. Each repository is cloned at the exact commit whose files
# are byte-identical to the copies used in the experiments (verified file by file on 2026-09-30), and its contents are
# placed in 00_assets/models/<model>/ next to our own integration files (INTEGRATION_SPEC.md), which are never overwritten.
#   ET-BERT       linwhitehat/ET-BERT       d59470624acd11405f9e88da650cf3bf65521e9a   (MIT)
#   YaTC          NSSL-SJTU/YaTC            a220b75f8365acaf63dbac7df87aaaa357698b85   (no license file; not redistributed)
#   NetMamba      wangtz19/NetMamba         bef641e161553aaf8cf524e73b11b6ca3932e5ed   (no license file; not redistributed)
#   TrafficFormer IDP-code/TrafficFormer    6d0ba64d82e74fb130c6c7301ef20885dbfbdf29   (MIT)
# Our only change to upstream code: a 4-line forward() shim in YaTC's models_YaTC.py for timm>=0.9
# (applied by 00_assets/models/05_yatc/apply_timm_compat.py). 2D-CNN is our own implementation (03_model/own_models/03_2dcnn/model.py).
# Pre-trained weights are downloaded from the links given in each upstream README (see 99_documents/MODELS.md).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS=1; [[ "${1:-}" == "--no-weights" ]] && WEIGHTS=0
FAIL=0
have() { command -v "$1" >/dev/null 2>&1; }

fetch_repo() {   # fetch_repo <owner/name> <commit> <target dir under 00_assets/models>
  local repo="$1" commit="$2" dst="$HERE/$3" tmp
  echo "── $repo @ ${commit:0:12} → 00_assets/models/$3"
  tmp="$(mktemp -d)"
  if git clone --quiet "https://github.com/$repo.git" "$tmp/src" && git -C "$tmp/src" checkout --quiet "$commit"; then
    mkdir -p "$dst"
    (cd "$tmp/src" && tar --exclude=.git -cf - .) | (cd "$dst" && tar -xkf - 2>/dev/null)   # -k: keep our own files
    echo "  [ok] $(git -C "$tmp/src" log -1 --format='%h %ad' --date=short)"
  else
    echo "  [FAIL] clone/checkout failed"; FAIL=1
  fi
  rm -rf "$tmp"
}

ensure_gdown() { have gdown || pip3 install --quiet gdown || pip3 install --user --quiet gdown || true; }
gdrive_get() {   # gdrive_get <file id> <destination>
  mkdir -p "$(dirname "$2")"
  if have gdown; then gdown "$1" -O "$2" && return 0; fi
  python3 -c "import gdown,sys; gdown.download(id=sys.argv[1], output=sys.argv[2], quiet=False)" "$1" "$2"
}
check_size() {   # check_size <file> <minimum MB>
  [[ -f "$1" ]] || return 1
  local sz; sz=$(stat -c %s "$1" 2>/dev/null || stat -f %z "$1")
  if (( sz < $2 * 1024 * 1024 )); then echo "  [FAIL] $1 is ${sz} B (< $2 MB) — deleted"; rm -f "$1"; return 1; fi
  echo "  [ok] $1 ($(( sz / 1024 / 1024 )) MB)"
}

fetch_repo linwhitehat/ET-BERT     d59470624acd11405f9e88da650cf3bf65521e9a 04_etbert
fetch_repo NSSL-SJTU/YaTC          a220b75f8365acaf63dbac7df87aaaa357698b85 05_yatc
fetch_repo wangtz19/NetMamba       bef641e161553aaf8cf524e73b11b6ca3932e5ed 06_netmamba
fetch_repo IDP-code/TrafficFormer  6d0ba64d82e74fb130c6c7301ef20885dbfbdf29 07_trafficformer
python3 "$HERE/05_yatc/apply_timm_compat.py" || FAIL=1

if (( WEIGHTS )); then
  echo "── pre-trained weights"
  ensure_gdown
  # ET-BERT (upstream README): https://drive.google.com/file/d/1r1yE34dU2W8zSqx1FkB8gCWri4DQWVtE
  f="$HERE/04_etbert/models/pre-trained_model.bin"
  check_size "$f" 500 || { gdrive_get 1r1yE34dU2W8zSqx1FkB8gCWri4DQWVtE "$f"; check_size "$f" 500 || FAIL=1; }
  # YaTC (upstream README): https://drive.google.com/file/d/1wWmZN87NgwujSd2-o5nm3HaQUIzWlv16
  f="$HERE/05_yatc/output_dir/pretrained-model.pth"
  check_size "$f" 5 || { gdrive_get 1wWmZN87NgwujSd2-o5nm3HaQUIzWlv16 "$f"; check_size "$f" 5 || FAIL=1; }
  # NetMamba (upstream README): https://huggingface.co/wangtz/NetMamba
  f="$HERE/06_netmamba/pre-train.pth"
  check_size "$f" 10 || { mkdir -p "$(dirname "$f")"
    (have wget && wget -q -c -O "$f" https://huggingface.co/wangtz/NetMamba/resolve/main/pre-train.pth) \
      || curl -sL -o "$f" https://huggingface.co/wangtz/NetMamba/resolve/main/pre-train.pth
    check_size "$f" 10 || FAIL=1; }
  # TrafficFormer (upstream README): https://drive.google.com/file/d/1pR6ZaWE7MWFDQWiF4LDzSyjSq0Gj3kV7
  f="$HERE/07_trafficformer/pretrain_model.bin"
  check_size "$f" 100 || { gdrive_get 1pR6ZaWE7MWFDQWiF4LDzSyjSq0Gj3kV7 "$f"; check_size "$f" 100 || FAIL=1; }
fi

echo
if (( FAIL )); then
  echo "[done — with failures] see [FAIL] above. If a Google Drive quota is exceeded, download the file in a browser"
  echo "  from the link in 99_documents/MODELS.md and place it at the path shown."
  exit 1
fi
echo "[done] upstream code and weights are in place"
