#!/bin/bash
# 2026-11-16: wider panels in every plot module, position_by_length_<task> figures, L8/L64 thinking-token figures.
cd /insomnia001/home/pm3361/qwen35-bench
export MPLBACKEND=Agg
OFF="results/recognition_single_foil results/serial_order_final results/pam_final"
ON="results/recognition_single_foil_thinking results/serial_order_final_thinking results/pam_final_thinking"
echo "=== plot (baseline)"; python3 -m bench.plot $OFF --out plots/baseline 2>&1 | tail -5
echo "=== plot_thinking"; python3 -m bench.plot_thinking --off $OFF --on $ON --out plots/thinking 2>&1 | tail -5
echo "=== plot_tokens"; python3 -m bench.plot_tokens 2>&1 | grep -vE "^saved" | tail -8
echo "=== plot_thinking_tokens"; python3 -m bench.plot_thinking_tokens 2>&1 | tail -8
echo "=== plot_thinking_tokens L8 L64"; python3 -m bench.plot_thinking_tokens --lengths 8 64 2>&1 | tail -4
echo "=== plot_corruption"; python3 -m bench.plot_corruption 2>&1 | grep -vE "^saved" | tail -8
echo "=== plot_natural"; python3 -m bench.plot_natural 2>&1 | tail -5
echo "=== plot_decay"; python3 -m bench.plot_decay 2>&1 | tail -5
echo "=== plot_position"; python3 -m bench.plot_position 2>&1 | tail -8
echo "=== plot_ported"; python3 -m bench.plot_ported --out plots/ported 2>&1 | tail -5
echo "=== interp"
source /insomnia001/depts/zgroup/zgroup_burg/zgroup/users/pm3361/envs/qwen35/bin/activate
for d in phase2_brady_4b phase2_brady_9b phase2_tokens_4b phase2_tokens_9b; do
  for m in recovery flip ld; do python -m interp.phase2_3.plot_patch results/interp/$d --metric $m 2>&1 | tail -2; done
done
echo "=== done"
