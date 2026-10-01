#!/usr/bin/env bash
# Fetch SplitCap (Netresec; free software under CC BY-ND 4.0, https://www.netresec.com/?page=SplitCap), used by step 3 to
# split the captures into sessions. It is not redistributed in this repository. On Linux it runs with mono (version 5 or
# later; e.g. apt install mono-complete).
#   bash 00_assets/tools/setup_splitcap.sh          -> 00_assets/tools/SplitCap/SplitCap.exe
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="$HERE/SplitCap"
mkdir -p "$DST"
if [ ! -f "$DST/SplitCap.exe" ]; then
  tmp="$(mktemp -d)"
  page="https://www.netresec.com/?page=SplitCap"
  curl -fsSL -c "$tmp/jar" -b "$tmp/jar" "$page" -o /dev/null                     # the download needs the page's cookie
  curl -fsSL -c "$tmp/jar" -b "$tmp/jar" -e "$page" "https://www.netresec.com/?download=SplitCap" -o "$tmp/dl"
  case "$(head -c 2 "$tmp/dl")" in
    MZ) cp "$tmp/dl" "$DST/SplitCap.exe" ;;                                           # a single SplitCap.exe
    PK) unzip -q -o "$tmp/dl" -d "$tmp/x" && cp "$(dirname "$(find "$tmp/x" -name SplitCap.exe | head -n 1)")"/* "$DST"/ ;;
    *) echo "unexpected download (not SplitCap.exe or a zip); get it by hand from $page into $DST/"; rm -rf "$tmp"; exit 1 ;;
  esac
  rm -rf "$tmp"
fi
command -v mono >/dev/null || echo "mono is not installed: see https://www.mono-project.com/download/stable/"
echo "SplitCap: $DST/SplitCap.exe"
