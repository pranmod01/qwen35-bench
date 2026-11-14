#!/bin/bash
cd /insomnia001/home/pm3361/qwen35-bench
export MPLBACKEND=Agg
echo "=== plot_tokens"; python3 -m bench.plot_tokens 2>&1 | tail -15
echo "=== plot_thinking_tokens"; python3 -m bench.plot_thinking_tokens 2>&1 | tail -15
echo "=== plot_natural"; python3 -m bench.plot_natural 2>&1 | tail -8
echo "=== done"
