#!/bin/bash
cd /insomnia001/home/pm3361/qwen35-bench
export MPLBACKEND=Agg
echo "=== plot_tokens"; python3 -m bench.plot_tokens 2>&1 | grep -vE "^saved" | tail -10
echo "=== plot_thinking_tokens"; python3 -m bench.plot_thinking_tokens 2>&1 | tail -10
echo "=== plot_corruption"; python3 -m bench.plot_corruption 2>&1 | grep -vE "^saved" | tail -10
echo "=== done"
