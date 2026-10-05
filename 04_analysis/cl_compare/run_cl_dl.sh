#!/usr/bin/env bash
# run_cl_dl.sh — DL rows of the rules vs. confident learning comparison (Exp3, refined test set).
#
# 1) build_cl_dl.py builds the training subsets 01_dataset/<ds>/<variant>/<model>/ from the sizectrl sample
#    (needs <ds>_cl_flags.csv from `cl_compare.py keys cl` in 99_documents/results/analysis/09_cl_compare/).
# 2) 250 training runs = 5 models x 5 seeds x 10 (dataset, variant) cells:
#      vpn16 / tor16 / iot23 x {cl_rule, cl_budget, cl_random_s<seed>}  (the random removal uses the seed's own set)
#      cic17 x cl_default                                               (control: no rule-noise sessions)
#    This is exactly the union of the job lists used for the article (four GPU scripts, 62 + 62 + 64 + 62 runs,
#    no duplicates), which equals the full product above.
# 3) cl_dl_summary.py reads 99_documents/results/result_tables/<variant>[_seed<N>]/<model>.csv.
#
# Usage: GPU=0 bash run_cl_dl.sh
# Only the accuracy tables are needed; the trained weights in 03_model/param/ can be deleted afterwards.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

python3 "$HERE/build_cl_dl.py" || exit 1

declare -A BS=([etbert]=64 [2dcnn]=256 [yatc]=256 [netmamba]=256 [trafficformer]=32)
declare -A EXTRA=([2dcnn]="--epochs 100" [etbert]="--epochs 20 --test_interval 3" [yatc]="--epochs 20 --test_interval 3" [netmamba]="--epochs 20 --test_interval 3" [trafficformer]="--epochs 20 --test_interval 3")

run() {
  local m=$1 ds=$2 v=$3 s=$4
  echo "=== $m $ds $v seed $s ==="
  (cd "$REPO/03_model" && python3 -u 01_train.py --model $m --dataset $ds --exp 3 --variant $v --seed $s \
      --gpu ${GPU:-0} --batch_size ${BS[$m]} ${EXTRA[$m]})
}

for m in 2dcnn etbert yatc netmamba trafficformer; do
  for s in 42 1 7 2024 31337; do
    for ds in vpn16 tor16 iot23; do
      for v in cl_rule cl_budget cl_random_s$s; do
        run $m $ds $v $s
      done
    done
    run $m cic17 cl_default $s
  done
done

python3 "$HERE/cl_dl_summary.py"
