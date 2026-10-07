#!/usr/bin/env python3
"""
Lightweight, non-intrusive power and thermal monitor for Apple Silicon.
Monitors SoC power (Watts), GPU/CPU power, thermals, fans, and memory without
interfering with active benchmarks or requiring root privileges.

Uses `mactop` in headless streaming mode (native IOReport/SMC counters).
Correlates power samples with the active benchmark arm from state.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

DEFAULT_INTERVAL_SEC = 2.0
CSV_FIELDNAMES = [
    "timestamp",
    "epoch_sec",
    "arm",
    "total_power_w",
    "gpu_power_w",
    "system_power_w",
    "cpu_power_w",
    "ane_power_w",
    "gpu_active_pct",
    "gpu_freq_mhz",
    "gpu_temp_c",
    "cpu_temp_c",
    "soc_temp_c",
    "ram_used_gb",
    "swap_used_gb",
    "fan0_rpm",
    "fan1_rpm",
    "thermal_state",
    "delta_sec",
    "delta_joules",
    "cumulative_joules",
    "cumulative_wh",
]


@dataclass(frozen=True)
class PowerSample:
    timestamp: str
    epoch_sec: float
    arm: str
    total_power_w: float
    gpu_power_w: float
    system_power_w: float
    cpu_power_w: float
    ane_power_w: float
    gpu_active_pct: float
    gpu_freq_mhz: int
    gpu_temp_c: float
    cpu_temp_c: float
    soc_temp_c: float
    ram_used_gb: float
    swap_used_gb: float
    fan0_rpm: int
    fan1_rpm: int
    thermal_state: str
    delta_sec: float
    delta_joules: float
    cumulative_joules: float
    cumulative_wh: float

    def to_csv_row(self) -> dict[str, str | float | int]:
        return {
            "timestamp": self.timestamp,
            "epoch_sec": round(self.epoch_sec, 3),
            "arm": self.arm,
            "total_power_w": round(self.total_power_w, 2),
            "gpu_power_w": round(self.gpu_power_w, 2),
            "system_power_w": round(self.system_power_w, 2),
            "cpu_power_w": round(self.cpu_power_w, 2),
            "ane_power_w": round(self.ane_power_w, 2),
            "gpu_active_pct": round(self.gpu_active_pct, 1),
            "gpu_freq_mhz": self.gpu_freq_mhz,
            "gpu_temp_c": round(self.gpu_temp_c, 1),
            "cpu_temp_c": round(self.cpu_temp_c, 1),
            "soc_temp_c": round(self.soc_temp_c, 1),
            "ram_used_gb": round(self.ram_used_gb, 2),
            "swap_used_gb": round(self.swap_used_gb, 2),
            "fan0_rpm": self.fan0_rpm,
            "fan1_rpm": self.fan1_rpm,
            "thermal_state": self.thermal_state,
            "delta_sec": round(self.delta_sec, 3),
            "delta_joules": round(self.delta_joules, 2),
            "cumulative_joules": round(self.cumulative_joules, 2),
            "cumulative_wh": round(self.cumulative_wh, 4),
        }


@dataclass(frozen=True)
class ArmSummary:
    arm: str
    sample_count: int
    duration_sec: float
    total_energy_wh: float
    avg_power_w: float
    peak_power_w: float
    avg_gpu_power_w: float
    peak_gpu_power_w: float
    avg_gpu_active_pct: float
    peak_gpu_temp_c: float


def find_mactop_binary() -> Path:
    cmd = shutil.which("mactop")
    if cmd:
        return Path(cmd)
    brew_path = Path("/opt/homebrew/bin/mactop")
    if brew_path.exists():
        return brew_path
    raise FileNotFoundError(
        "mactop binary not found. Install it via 'brew install metaspartan/apple/mactop'"
    )


def resolve_active_arm(run_dir: Path | None) -> tuple[str, bool]:
    """
    Reads state.json to identify the currently executing arm.
    Returns (active_arm_name, all_complete_bool).
    """
    if run_dir is None:
        return ("unassigned", False)

    state_file = run_dir / "state.json"
    if not state_file.exists():
        return ("unknown", False)

    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
        arms = data.get("arms", {})
        running_arms = [name for name, st in arms.items() if st == "running"]
        if running_arms:
            return (running_arms[0], False)

        all_done = bool(arms) and all(st == "complete" for st in arms.values())
        return ("idle", all_done)
    except (OSError, json.JSONDecodeError):
        return ("state_read_error", False)


def parse_sample_payload(
    raw_json: dict,
    prev: PowerSample | None,
    current_arm: str,
    now_epoch: float,
) -> PowerSample:
    soc = raw_json.get("soc_metrics", {})
    mem = raw_json.get("memory", {})
    fans = raw_json.get("fans", [])

    total_w = float(soc.get("total_power", 0.0))
    gpu_w = float(soc.get("gpu_power", 0.0))
    sys_w = float(soc.get("system_power", 0.0))
    cpu_w = float(soc.get("cpu_power", 0.0))
    ane_w = float(soc.get("ane_power", 0.0))

    delta_sec = (now_epoch - prev.epoch_sec) if prev is not None else 0.0
    delta_joules = total_w * delta_sec if delta_sec > 0 else 0.0
    cum_joules = (prev.cumulative_joules + delta_joules) if prev is not None else 0.0
    cum_wh = cum_joules / 3600.0

    fan0_rpm = int(fans[0].get("rpm", 0)) if len(fans) > 0 else 0
    fan1_rpm = int(fans[1].get("rpm", 0)) if len(fans) > 1 else 0

    return PowerSample(
        timestamp=str(raw_json.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S%z"))),
        epoch_sec=now_epoch,
        arm=current_arm,
        total_power_w=total_w,
        gpu_power_w=gpu_w,
        system_power_w=sys_w,
        cpu_power_w=cpu_w,
        ane_power_w=ane_w,
        gpu_active_pct=float(soc.get("gpu_active", 0.0)),
        gpu_freq_mhz=int(soc.get("gpu_freq_mhz", 0)),
        gpu_temp_c=float(soc.get("gpu_temp", 0.0)),
        cpu_temp_c=float(soc.get("cpu_temp", 0.0)),
        soc_temp_c=float(soc.get("soc_temp", 0.0)),
        ram_used_gb=float(mem.get("used", 0)) / (1024.0**3),
        swap_used_gb=float(mem.get("swap_used", 0)) / (1024.0**3),
        fan0_rpm=fan0_rpm,
        fan1_rpm=fan1_rpm,
        thermal_state=str(raw_json.get("thermal_state", "Nominal")),
        delta_sec=delta_sec,
        delta_joules=delta_joules,
        cumulative_joules=cum_joules,
        cumulative_wh=cum_wh,
    )


def stream_mactop_lines(mactop_bin: Path, interval_ms: int) -> Iterator[dict]:
    cmd = [
        str(mactop_bin),
        "--headless",
        "--format",
        "json",
        "--count",
        "0",
        "--interval",
        str(interval_ms),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )

    try:
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            clean = raw_line.strip()
            if not clean or clean == "]":
                continue
            if clean.startswith("["):
                clean = clean[1:].strip()
            elif clean.startswith(","):
                clean = clean[1:].strip()
            if clean.endswith("]"):
                clean = clean[:-1].strip()
            if not clean:
                continue
            try:
                yield json.loads(clean)
            except json.JSONDecodeError:
                continue
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()


def format_live_line(s: PowerSample) -> str:
    time_str = s.timestamp.split("T")[-1].split("-")[0].split("+")[0]
    short_arm = s.arm.split(".")[0] if "." in s.arm else s.arm
    return (
        f"[{time_str}] [{short_arm:<14}] "
        f"Total: {s.total_power_w:5.1f}W | "
        f"GPU: {s.gpu_power_w:4.1f}W ({s.gpu_active_pct:4.1f}%@{s.gpu_freq_mhz:4d}M) | "
        f"Sys: {s.system_power_w:4.1f}W | "
        f"Temp: {s.gpu_temp_c:4.1f}°C | "
        f"Fans: {s.fan0_rpm}/{s.fan1_rpm} RPM | "
        f"Energy: {s.cumulative_wh:6.2f} Wh"
    )


def build_summary_from_csv(csv_path: Path) -> list[ArmSummary]:
    if not csv_path.exists():
        return []

    arm_records: dict[str, list[dict]] = {}
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            arm = row.get("arm", "unknown")
            if arm not in arm_records:
                arm_records[arm] = []
            arm_records[arm].append(row)

    summaries: list[ArmSummary] = []
    for arm, rows in arm_records.items():
        if not rows:
            continue
        count = len(rows)
        duration = sum(float(r["delta_sec"]) for r in rows)
        energy_wh = sum(float(r["delta_joules"]) for r in rows) / 3600.0
        tot_powers = [float(r["total_power_w"]) for r in rows]
        gpu_powers = [float(r["gpu_power_w"]) for r in rows]
        gpu_actives = [float(r["gpu_active_pct"]) for r in rows]
        gpu_temps = [float(r["gpu_temp_c"]) for r in rows]

        avg_pow = sum(tot_powers) / count
        peak_pow = max(tot_powers)
        avg_gpu = sum(gpu_powers) / count
        peak_gpu = max(gpu_powers)
        avg_act = sum(gpu_actives) / count
        peak_temp = max(gpu_temps)

        summaries.append(
            ArmSummary(
                arm=arm,
                sample_count=count,
                duration_sec=duration,
                total_energy_wh=energy_wh,
                avg_power_w=avg_pow,
                peak_power_w=peak_pow,
                avg_gpu_power_w=avg_gpu,
                peak_gpu_power_w=peak_gpu,
                avg_gpu_active_pct=avg_act,
                peak_gpu_temp_c=peak_temp,
            )
        )
    return summaries


def print_summary_table(summaries: list[ArmSummary], kwh_rate: float = 0.16) -> None:
    if not summaries:
        print("No samples recorded to summarize.")
        return

    print("\n" + "=" * 106)
    print(
        f"{'ARM / QUANT':<26} {'DURATION':<10} {'AVG (W)':<8} {'PEAK (W)':<9} "
        f"{'GPU (W)':<8} {'GPU (%)':<8} {'WH':<9} {'KWH':<9} {f'COST (@${kwh_rate:.2f})':<11}"
    )
    print("-" * 106)

    total_wh = 0.0
    total_dur = 0.0

    for s in summaries:
        dur_min = s.duration_sec / 60.0
        total_wh += s.total_energy_wh
        total_dur += s.duration_sec
        kwh = s.total_energy_wh / 1000.0
        cost = kwh * kwh_rate
        short_arm = s.arm[:25]
        print(
            f"{short_arm:<26} "
            f"{dur_min:6.1f} min  "
            f"{s.avg_power_w:5.1f} W  "
            f"{s.peak_power_w:6.1f} W   "
            f"{s.avg_gpu_power_w:5.1f} W  "
            f"{s.avg_gpu_active_pct:5.1f} %  "
            f"{s.total_energy_wh:6.2f} Wh  "
            f"{kwh:6.4f} kWh  "
            f"${cost:6.4f}"
        )

    print("-" * 106)
    tot_min = total_dur / 60.0
    tot_kwh = total_wh / 1000.0
    tot_cost = tot_kwh * kwh_rate
    print(
        f"{'TOTAL':<26} {tot_min:6.1f} min {' ':24} "
        f"{total_wh:6.2f} Wh  {tot_kwh:6.4f} kWh  ${tot_cost:6.4f}"
    )
    print("=" * 106 + "\n")



def run_monitor(
    run_dir: Path | None,
    output_csv: Path,
    interval_sec: float,
    stop_when_done: bool,
    quiet: bool,
    kwh_rate: float = 0.16,
) -> None:
    mactop_bin = find_mactop_binary()
    interval_ms = int(interval_sec * 1000)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_csv.exists() or output_csv.stat().st_size == 0

    csv_file = open(output_csv, mode="a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
    if write_header:
        writer.writeheader()
        csv_file.flush()

    prev_sample: PowerSample | None = None
    running = True

    def sig_handler(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    if not quiet:
        print(f"Monitoring power via mactop every {interval_sec}s -> {output_csv}")
        if run_dir:
            print(f"Tracking active showdown arm from: {run_dir / 'state.json'}")
        print(f"Electricity rate: ${kwh_rate:.2f}/kWh | Press Ctrl+C to stop.\n")

    try:
        for payload in stream_mactop_lines(mactop_bin, interval_ms):
            if not running:
                break

            current_arm, all_done = resolve_active_arm(run_dir)
            now_epoch = time.time()
            sample = parse_sample_payload(payload, prev_sample, current_arm, now_epoch)

            writer.writerow(sample.to_csv_row())
            csv_file.flush()
            prev_sample = sample

            if not quiet:
                print(format_live_line(sample))

            if stop_when_done and all_done:
                if not quiet:
                    print("\nAll showdown arms marked complete in state.json. Exiting.")
                break
    finally:
        csv_file.close()
        summaries = build_summary_from_csv(output_csv)
        print_summary_table(summaries, kwh_rate=kwh_rate)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Non-intrusive power & energy monitor for local LLM benchmarking."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("results/qwen38_showdown/core02"),
        help="Path to showdown run directory to track active arm (default: results/qwen38_showdown/core02)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Path to write power CSV (default: <run-dir>/power_metrics.csv)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_SEC,
        help=f"Sampling interval in seconds (default: {DEFAULT_INTERVAL_SEC})",
    )
    parser.add_argument(
        "--kwh-rate",
        type=float,
        default=0.16,
        help="Electricity cost per kilowatt-hour in USD (default: 0.16)",
    )
    parser.add_argument(
        "--stop-when-done",
        action="store_true",
        help="Automatically exit when state.json indicates all arms are complete",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print live samples to stdout",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        metavar="CSV_FILE",
        help="Display summary table for an existing power CSV file and exit",
    )

    args = parser.parse_args()

    if args.summary:
        summaries = build_summary_from_csv(args.summary)
        print_summary_table(summaries, kwh_rate=args.kwh_rate)
        return 0

    run_dir = args.run_dir if (args.run_dir and args.run_dir.exists()) else None
    output_csv = args.output_csv
    if output_csv is None:
        if run_dir:
            output_csv = run_dir / "power_metrics.csv"
        else:
            output_csv = Path("results/power_metrics.csv")

    try:
        run_monitor(
            run_dir=run_dir,
            output_csv=output_csv,
            interval_sec=args.interval,
            stop_when_done=args.stop_when_done,
            quiet=args.quiet,
            kwh_rate=args.kwh_rate,
        )

    except FileNotFoundError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
