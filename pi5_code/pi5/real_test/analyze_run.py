#!/usr/bin/env python3
"""
Summarize a stress_test.py run: max temp, throttle events, FPS stats,
memory growth, and save a 3-panel plot.

Usage: python3 analyze_run.py logs/20260617_143000/
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_run.py <run_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    sysdf = pd.read_csv(run_dir / "system_metrics.csv")
    pipedf = pd.read_csv(run_dir / "pipeline_metrics.csv")

    print("=== System metrics ===")
    print(f"Max CPU temp:            {sysdf['cpu_temp_c'].max():.1f} C")
    print(f"Mean CPU temp:           {sysdf['cpu_temp_c'].mean():.1f} C")
    print(f"Throttled-now events:    {int(sysdf['throttled_now'].sum())}")
    print(f"Soft temp limit events:  {int(sysdf['soft_temp_limit_now'].sum())}")
    print(f"Throttled since boot:    {bool(sysdf['throttled_since_boot'].any())}")
    print(f"Min CPU freq:            {sysdf['cpu_freq_mhz'].min():.0f} MHz")
    rss_start = sysdf['process_rss_mb'].iloc[0]
    rss_end = sysdf['process_rss_mb'].iloc[-1]
    print(f"Process RSS start->end: {rss_start:.1f} -> {rss_end:.1f} MB "
          f"(delta {rss_end - rss_start:+.1f} MB)")

    print("\n=== Pipeline metrics ===")
    print(f"Frames processed:        {len(pipedf)}")
    print(f"Mean FPS:                {pipedf['instant_fps'].mean():.2f}")
    print(f"Min FPS:                 {pipedf['instant_fps'].min():.2f}")
    print(f"Mean loop time:          {pipedf['loop_total_ms'].mean():.1f} ms")
    print(f"Max loop time:           {pipedf['loop_total_ms'].max():.1f} ms")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    axes[0].plot(sysdf["elapsed_s"], sysdf["cpu_temp_c"])
    axes[0].axhline(80, color="r", linestyle="--", label="throttle threshold")
    axes[0].set_ylabel("CPU temp (C)")
    axes[0].legend()
    axes[0].set_title("CPU temperature over run")

    axes[1].plot(sysdf["elapsed_s"], sysdf["process_rss_mb"])
    axes[1].set_ylabel("Process RSS (MB)")
    axes[1].set_title("Memory growth over run")

    axes[2].plot(pipedf["elapsed_s"], pipedf["instant_fps"])
    axes[2].set_ylabel("Instant FPS")
    axes[2].set_xlabel("Elapsed time (s)")
    axes[2].set_title("Pipeline FPS over run")

    plt.tight_layout()
    out_path = run_dir / "summary.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
