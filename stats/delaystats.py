import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def parse_exclude_list(exclude_str):
    result = set()
    if exclude_str.strip() == "":
        return result
    for part in exclude_str.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    return result

def load_and_concat_csvs(folder_path, pattern, excluded_files):
    all_dfs = []
    for f in sorted(folder_path.glob(pattern)):
        num = int(f.name.split('_')[0])
        if num in excluded_files:
            continue
        df = pd.read_csv(f)
        df['file_num'] = num
        all_dfs.append(df)
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()

def compute_conntrack_rate(df):
    df = df.sort_values('time').reset_index(drop=True)
    df['conntrack_count'] = pd.to_numeric(df['conntrack_count'], errors='coerce')
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df['conntrack_rate'] = df['conntrack_count'].diff() / df['time'].diff()
    df['conntrack_rate'].fillna(0, inplace=True)
    return df

def split_base_onload_by_rate(df, rate_threshold):
    base_df = df[df['conntrack_rate'] <= rate_threshold]
    onload_df = df[df['conntrack_rate'] > rate_threshold]
    return base_df, onload_df

def get_onload_time_bounds(onload_df):
    if onload_df.empty:
        return None, None
    start_time = onload_df['time'].min()
    end_time = onload_df['time'].max()
    return start_time, end_time

def compute_stats(df, cols, extra_cols=None):
    stats = {}
    for col in cols:
        data = pd.to_numeric(df[col], errors='coerce').dropna()
        if data.empty:
            stats[col] = "No valid data"
        else:
            mean = data.mean()
            median = data.median()
            std = data.std()
            if extra_cols and col in extra_cols:
                min_val = data.min()
                max_val = data.max()
                stats[col] = (mean, median, std, min_val, max_val)
            else:
                stats[col] = (mean, median, std)
    return stats

def format_stats(stats, desc_map):
    lines = []
    for col, val in stats.items():
        desc = desc_map.get(col, col)
        if isinstance(val, str):
            lines.append(f"  {desc}: {val}")
        else:
            if len(val) == 5:  # includes min/max
                mean, median, std, min_val, max_val = val
                lines.append(f"  {desc}: Mean = {mean:.2f}, Median = {median:.2f}, Std = {std:.2f}, Min = {min_val:.2f}, Max = {max_val:.2f}")
            else:
                mean, median, std = val
                lines.append(f"  {desc}: Mean = {mean:.2f}, Median = {median:.2f}, Std = {std:.2f}")
    return lines

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rawdir', required=True, help="Base raw directory")
    parser.add_argument('--foldername', required=True, help="Folder name inside connt1 and connt2")
    parser.add_argument('--exclude', default="", help="Exclude files by number, e.g. 2-3,5")
    parser.add_argument('--outdir', required=True, help="Output directory")
    args = parser.parse_args()

    rawdir = Path(args.rawdir)
    foldername = args.foldername
    exclude_set = parse_exclude_list(args.exclude)
    outdir = Path(args.outdir)
    outdir_path = outdir / foldername
    outdir_path.mkdir(parents=True, exist_ok=True)
    summary_file = outdir_path / "combined_base_onload_summary.txt"

    all_summary = []

    # Columns of interest for cm_monitor stats
    cm_cols = ['proc_cpu_percent', 'proc_cpu_cycles_ghz', 'proc_mem_mb', 'conntrack_rate', 'clock_delta_ms']
    cm_desc = {
        'proc_cpu_percent': 'Process CPU (%)',
        'proc_cpu_cycles_ghz': 'CPU Cycles (GHz)',
        'proc_mem_mb': 'Process Memory (MB)',
        'conntrack_rate': 'Conntrack Rate (Δcount/sec)',
        'clock_delta_ms': 'Clock Delta (ms)'
    }
    cm_extra_cols = ['clock_delta_ms']

    # Columns for n_monitor stats
    n_cols = ['iface_rx_bytes_per_sec', 'iface_tx_bytes_per_sec']
    n_desc = {'iface_rx_bytes_per_sec': 'RX KB/s', 'iface_tx_bytes_per_sec': 'TX KB/s'}

    for conn_folder in ['connt1', 'connt2']:
        folder_path = rawdir / conn_folder / foldername
        if not folder_path.exists():
            all_summary.append(f"Warning: Folder {folder_path} does not exist. Skipping.\n")
            continue

        all_summary.append(f"--- Summary for {conn_folder}/{foldername} ---")

        # Load and concat cm_monitor files
        cm_df = load_and_concat_csvs(folder_path, "*_conntrackd_cm_monitor.csv", exclude_set)
        if cm_df.empty:
            all_summary.append("No cm_monitor data found.\n")
            continue

        # Compute conntrack rate
        cm_df = compute_conntrack_rate(cm_df)

        # Detect base and onload segments by conntrack_rate threshold (~c1000 in folder name)
        expected_rate = int(''.join(filter(str.isdigit, foldername)))  # extract number like 1000 from c1000
        base_cm, onload_cm = split_base_onload_by_rate(cm_df, expected_rate)

        # Compute stats for combined base and onload cm_monitor data
        all_summary.append("cm_monitor Base (Idle) Segment:")
        base_stats = compute_stats(base_cm, cm_cols, extra_cols=cm_extra_cols)
        all_summary.extend(format_stats(base_stats, cm_desc))

        all_summary.append("\ncm_monitor Onload (Experiment) Segment:")
        onload_stats = compute_stats(onload_cm, cm_cols, extra_cols=cm_extra_cols)
        all_summary.extend(format_stats(onload_stats, cm_desc))
        all_summary.append("")

        # Determine onload time bounds for splitting n_monitor data
        onload_start, onload_end = None, None
        if not onload_cm.empty:
            onload_start = onload_cm['time'].min()
            onload_end = onload_cm['time'].max()

        # Load and concat n_monitor files
        n_df = load_and_concat_csvs(folder_path, "*_conntrackd_n_monitor.csv", exclude_set)
        if n_df.empty:
            all_summary.append("No n_monitor data found.\n")
            continue

        # Convert bytes/sec to KB/s
        n_df['iface_rx_bytes_per_sec'] = pd.to_numeric(n_df['iface_rx_bytes_per_sec'], errors='coerce') / 1024.0
        n_df['iface_tx_bytes_per_sec'] = pd.to_numeric(n_df['iface_tx_bytes_per_sec'], errors='coerce') / 1024.0
        n_df = n_df.dropna(subset=['iface_rx_bytes_per_sec', 'iface_tx_bytes_per_sec'])

        # Split base and onload for n_monitor using onload time bounds from cm_monitor
        if onload_start is not None and onload_end is not None:
            base_n = n_df[(n_df['time'] < onload_start) | (n_df['time'] > onload_end)]
            onload_n = n_df[(n_df['time'] >= onload_start) & (n_df['time'] <= onload_end)]
        else:
            base_n = n_df
            onload_n = pd.DataFrame()

        # Compute stats for n_monitor base and onload segments
        all_summary.append("n_monitor Base (Idle) Segment:")
        base_n_stats = {}
        for col in n_cols:
            data = base_n[col].dropna()
            if data.empty:
                base_n_stats[col] = "No valid data"
            else:
                mean = data.mean()
                median = data.median()
                std = data.std()
                base_n_stats[col] = (mean, median, std)
        all_summary.extend(format_stats(base_n_stats, n_desc))

        all_summary.append("\nn_monitor Onload (Experiment) Segment:")
        onload_n_stats = {}
        for col in n_cols:
            data = onload_n[col].dropna()
            if data.empty:
                onload_n_stats[col] = "No valid data"
            else:
                mean = data.mean()
                median = data.median()
                std = data.std()
                onload_n_stats[col] = (mean, median, std)
        all_summary.extend(format_stats(onload_n_stats, n_desc))

        all_summary.append("\n\n")

    # Write all summary to output file
    with open(summary_file, 'w') as f:
        f.write('\n'.join(all_summary))

    print(f"Summary written to {summary_file}")

if __name__ == "__main__":
    main()
