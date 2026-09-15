"""
Results Table Generator
========================
Produces:
  1. Formatted terminal table (colour-coded best values)
  2. CSV file (results/tables/metrics_summary.csv)
  3. Per-image CSV (results/tables/per_image_metrics.csv)
  4. LaTeX table snippet (results/tables/metrics_table.tex)
  5. HTML report (results/tables/report.html)
"""

import os
import csv
import html as _html
import sys
from typing import Dict

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Metric display config:  (display_name, higher_is_better, format_str)
METRIC_CONFIG = [
    # Reference-based
    ('psnr',         'PSNR (dB)',              True,  '{:.3f}'),
    ('ssim',         'SSIM',                   True,  '{:.4f}'),
    ('mse',          'MSE (px²)',               False, '{:.3f}'),
    ('mae',          'MAE (px)',                False, '{:.3f}'),
    ('rmse',         'RMSE (px)',               False, '{:.3f}'),
    ('psnr_gain',    'PSNR Gain (dB) ↑',        True,  '{:+.3f}'),
    ('ssim_gain',    'SSIM Gain ↑',             True,  '{:+.4f}'),
    ('mse_ratio',    'MSE Ratio (fog/out) ↑',   True,  '{:.2f}'),
    # No-reference
    ('sharpness',    'Sharpness ↑',             True,  '{:.2f}'),
    ('sharpness_gain','Sharpness Gain ↑',       True,  '{:+.2f}'),
    ('fog_density',  'Fog Density ↓',           False, '{:.4f}'),
    ('fog_reduction','Fog Reduction ↑',         True,  '{:+.4f}'),
    ('colorfulness', 'Colorfulness ↑',          True,  '{:.2f}'),
    ('color_gain',   'Color Gain ↑',            True,  '{:+.2f}'),
    ('contrast',     'Contrast ↑',              True,  '{:.2f}'),
    ('contrast_gain','Contrast Gain ↑',         True,  '{:+.2f}'),
    ('dark_channel', 'Dark Channel ↓',          False, '{:.4f}'),
    ('entropy',      'Entropy ↑',               True,  '{:.4f}'),
    ('lane_contrast','Lane Contrast ↑',         True,  '{:.2f}'),
    ('visibility',   'Visibility Depth ↑',      True,  '{:.3f}'),
    # Speed
    ('infer_ms',     'Inference (ms) ↓',        False, '{:.1f}'),
]

ANSI_GREEN  = '\033[92m'
ANSI_RED    = '\033[91m'
ANSI_BOLD   = '\033[1m'
ANSI_RESET  = '\033[0m'
ANSI_CYAN   = '\033[96m'
ANSI_YELLOW = '\033[93m'


def _fmt(v, fmt):
    if v is None: return 'N/A'
    try:    return fmt.format(float(v))
    except: return str(v)


def _winner(v1, v2, higher_better):
    """Return (is_v1_better, is_v2_better) ignoring None."""
    if v1 is None or v2 is None: return False, False
    if higher_better: return v1 > v2, v2 > v1
    else:             return v1 < v2, v2 < v1


def print_terminal_table(results: dict) -> None:
    """Print a colour-coded comparison table to stdout."""
    model_names = list(results.keys())
    means = {n: results[n]['mean'] for n in model_names}
    speeds= {n: results[n]['speed'] for n in model_names}
    stats = {n: results[n]['model_stats'] for n in model_names}

    print()
    print(ANSI_BOLD + '═'*78 + ANSI_RESET)
    print(ANSI_BOLD + '  MODEL COMPARISON RESULTS' + ANSI_RESET)
    print(ANSI_BOLD + '═'*78 + ANSI_RESET)

    # Header
    col_w = 22
    print(f"  {'Metric':<28}", end='')
    for n in model_names:
        print(f'{n:>{col_w}}', end='')
    print(f"  {'Winner':>10}")
    print('─'*78)

    # Group: Reference-based
    print(ANSI_CYAN + '  ── Reference-Based Metrics ──' + ANSI_RESET)
    _print_group(model_names, means, col_w, [
        ('psnr','ssim','mse','mae','rmse','psnr_gain','ssim_gain','mse_ratio')])

    print(ANSI_CYAN + '  ── No-Reference Metrics (Autonomous Driving) ──' + ANSI_RESET)
    _print_group(model_names, means, col_w, [
        ('sharpness','sharpness_gain','fog_density','fog_reduction',
         'colorfulness','color_gain','contrast','contrast_gain',
         'dark_channel','entropy','lane_contrast','visibility')])

    print(ANSI_CYAN + '  ── Speed & Efficiency ──' + ANSI_RESET)
    # Speed row
    vals = [speeds[n]['mean_ms'] for n in model_names]
    w1,w2 = _winner(vals[0], vals[1], False) if len(vals)>=2 else (False,False)
    print(f"  {'Inference time (ms)':<28}", end='')
    for v, is_win in zip(vals, [w1,w2]):
        s = f'{v:.1f}'
        col = ANSI_GREEN if is_win else ''
        print(f'{col}{s:>{col_w}}{ANSI_RESET if col else ""}', end='')
    print(f"  {model_names[0 if w1 else 1] if (w1 or w2) else "—":>10}")

    vals = [speeds[n]['fps'] for n in model_names]
    w1,w2 = _winner(vals[0], vals[1], True) if len(vals)>=2 else (False,False)
    print(f"  {'FPS':<28}", end='')
    for v, is_win in zip(vals, [w1,w2]):
        s = f'{v:.1f}'
        col = ANSI_GREEN if is_win else ''
        print(f'{col}{s:>{col_w}}{ANSI_RESET if col else ""}', end='')
    print(f"  {model_names[0 if w1 else 1] if (w1 or w2) else "—":>10}")

    vals = [stats[n]['params_M'] for n in model_names]
    w1,w2 = _winner(vals[0], vals[1], False) if len(vals)>=2 else (False,False)
    print(f"  {'Parameters (M)':<28}", end='')
    for v, is_win in zip(vals, [w1,w2]):
        s = f'{v:.3f} M'
        col = ANSI_GREEN if is_win else ''
        print(f'{col}{s:>{col_w}}{ANSI_RESET if col else ""}', end='')
    print(f"  {model_names[0 if w1 else 1] if (w1 or w2) else "—":>10}")

    print('═'*78)

    # Win summary
    wins = {n: 0 for n in model_names}
    all_metric_keys = [cfg[0] for cfg in METRIC_CONFIG]
    for key, _, higher_better, _ in METRIC_CONFIG:
        row_vals = [means[n].get(key) for n in model_names]
        if len(row_vals) >= 2 and None not in row_vals:
            w1,w2 = _winner(row_vals[0], row_vals[1], higher_better)
            if w1: wins[model_names[0]] += 1
            if w2: wins[model_names[1]] += 1

    print(ANSI_BOLD + '  Win count:' + ANSI_RESET)
    for n, w in wins.items():
        bar = '█' * w
        print(f'  {n:<20} {bar}  ({w} metrics)')
    print('═'*78 + '\n')


def _print_group(model_names, means, col_w, key_lists):
    for key_group in key_lists:
        for key in key_group:
            cfg = next((c for c in METRIC_CONFIG if c[0]==key), None)
            if cfg is None: continue
            _, display, higher_better, fmt = cfg
            row_vals = [means[n].get(key) for n in model_names]
            if len(model_names) >= 2:
                w1,w2 = _winner(row_vals[0], row_vals[1], higher_better)
                winners = [w1, w2]
            else:
                winners = [False]*len(model_names)

            print(f"  {display:<28}", end='')
            for v, is_win in zip(row_vals, winners):
                s = _fmt(v, fmt)
                col = ANSI_GREEN if is_win else (ANSI_RED if not is_win and v is not None else '')
                print(f'{col}{s:>{col_w}}{ANSI_RESET if col else ""}', end='')
            winner_name = (model_names[winners.index(True)]
                           if any(winners) else '—')
            print(f"  {winner_name:>10}")


def save_csv(results: dict, out_dir: str) -> None:
    """Save summary CSV and per-image CSV."""
    os.makedirs(out_dir, exist_ok=True)
    model_names = list(results.keys())

    # ── Summary CSV ───────────────────────────────────────────────────────────
    summary_path = os.path.join(out_dir, 'metrics_summary.csv')
    all_keys     = [c[0] for c in METRIC_CONFIG]

    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Metric', 'Display Name', 'Higher=Better'] +
                   [f'{n}_mean' for n in model_names] +
                   [f'{n}_std'  for n in model_names] +
                   ['Winner'])
        for key, display, higher_better, fmt in METRIC_CONFIG:
            row_means = [results[n]['mean'].get(key) for n in model_names]
            row_stds  = [results[n]['std'].get(key)  for n in model_names]
            if len(model_names) >= 2 and None not in row_means:
                w1,w2 = _winner(row_means[0], row_means[1], higher_better)
                winner = model_names[0] if w1 else (model_names[1] if w2 else '—')
            else:
                winner = '—'
            w.writerow([key, display, higher_better] + row_means + row_stds + [winner])

    # Speed
    with open(summary_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        fps_vals   = [results[n]['speed']['fps']      for n in model_names]
        ms_vals    = [results[n]['speed']['mean_ms']  for n in model_names]
        param_vals = [results[n]['model_stats']['params_M'] for n in model_names]
        w.writerow(['fps','FPS',True]+fps_vals+['']*len(model_names)+['—'])
        w.writerow(['infer_ms','Inference ms',False]+ms_vals+['']*len(model_names)+['—'])
        w.writerow(['params_M','Parameters (M)',False]+param_vals+['']*len(model_names)+['—'])

    print(f'[Table] Summary CSV → {summary_path}')

    # ── Per-image CSV ─────────────────────────────────────────────────────────
    per_path = os.path.join(out_dir, 'per_image_metrics.csv')
    metric_keys = [c[0] for c in METRIC_CONFIG if c[0] not in ('infer_ms',)]
    with open(per_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        header = ['model', 'image'] + metric_keys + ['infer_ms']
        w.writerow(header)
        for n in model_names:
            for row in results[n]['per_image']:
                w.writerow([n, row['name']] +
                           [row.get(k) for k in metric_keys] +
                           [row.get('infer_ms')])
    print(f'[Table] Per-image CSV → {per_path}')


def save_latex(results: dict, out_dir: str) -> None:
    """Save a LaTeX table snippet."""
    os.makedirs(out_dir, exist_ok=True)
    model_names = list(results.keys())
    path = os.path.join(out_dir, 'metrics_table.tex')

    key_subset = [('psnr','PSNR (dB)',True,'{:.3f}'),
                  ('ssim','SSIM',True,'{:.4f}'),
                  ('mse','MSE',False,'{:.3f}'),
                  ('sharpness','Sharpness',True,'{:.2f}'),
                  ('fog_reduction','Fog Reduction',True,'{:+.4f}'),
                  ('colorfulness','Colorfulness',True,'{:.2f}'),
                  ('lane_contrast','Lane Contrast',True,'{:.2f}'),
                  ('visibility','Visibility',True,'{:.3f}'),]

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\\begin{table}[h]\n\\centering\n')
        f.write('\\caption{AOD-Net vs GridDehazeNet — Dehazing Performance}\n')
        f.write('\\label{tab:comparison}\n')
        f.write('\\begin{tabular}{l' + 'r'*len(model_names) + '}\n')
        f.write('\\hline\n')
        f.write('\\textbf{Metric} & ' +
                ' & '.join(f'\\textbf{{{n}}}' for n in model_names) +
                ' \\\\\n\\hline\n')
        for key,display,higher_better,fmt in key_subset:
            vals = [results[n]['mean'].get(key) for n in model_names]
            fmts = [_fmt(v,fmt) for v in vals]
            if len(model_names)>=2 and None not in vals:
                w1,w2 = _winner(vals[0], vals[1], higher_better)
                if w1: fmts[0] = '\\textbf{'+fmts[0]+'}'
                if w2: fmts[1] = '\\textbf{'+fmts[1]+'}'
            f.write(f'{display} & ' + ' & '.join(fmts) + ' \\\\\n')
        f.write('\\hline\n')
        # Speed
        for n in model_names:
            pass
        fps_s  = [f"{results[n]['speed']['fps']:.1f}" for n in model_names]
        ms_s   = [f"{results[n]['speed']['mean_ms']:.1f}" for n in model_names]
        par_s  = [f"{results[n]['model_stats']['params_M']:.3f} M" for n in model_names]
        f.write('FPS & ' + ' & '.join(fps_s) + ' \\\\\n')
        f.write('Inference (ms) & ' + ' & '.join(ms_s) + ' \\\\\n')
        f.write('Parameters & ' + ' & '.join(par_s) + ' \\\\\n')
        f.write('\\hline\n\\end{tabular}\n\\end{table}\n')
    print(f'[Table] LaTeX → {path}')


def save_html(results: dict, out_dir: str) -> None:
    """Save an HTML report with styled table."""
    os.makedirs(out_dir, exist_ok=True)
    model_names = list(results.keys())
    path = os.path.join(out_dir, 'report.html')

    rows_html = ''
    for key,display,higher_better,fmt in METRIC_CONFIG:
        vals  = [results[n]['mean'].get(key) for n in model_names]
        fmts  = [_fmt(v,fmt) for v in vals]
        styles= [''] * len(model_names)
        if len(model_names)>=2 and None not in vals:
            w1,w2 = _winner(vals[0],vals[1],higher_better)
            if w1: styles[0]='background:#1a4a1a;color:#7fff7f;font-weight:bold;'
            if w2: styles[1]='background:#1a4a1a;color:#7fff7f;font-weight:bold;'
        cells = ''.join(f'<td style="{s}">{v}</td>'
                        for s,v in zip(styles,fmts))
        arrow = '↑' if higher_better else '↓'
        rows_html += f'<tr><td>{_html.escape(display)} {arrow}</td>{cells}</tr>\n'

    model_headers = ''.join(f'<th>{n}</th>' for n in model_names)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Dehazing Model Comparison</title>
<style>
 body{{font-family:monospace;background:#111;color:#ddd;padding:20px}}
 h1{{color:#6af}}
 table{{border-collapse:collapse;width:100%;margin-top:16px}}
 th{{background:#223;padding:8px 14px;color:#adf;border:1px solid #334}}
 td{{padding:6px 14px;border:1px solid #334;text-align:right}}
 td:first-child{{text-align:left;color:#bbb}}
 tr:nth-child(even){{background:#181818}}
 .note{{color:#888;font-size:0.85em;margin-top:12px}}
</style>
</head><body>
<h1>🚗 Dehazing Model Comparison Report</h1>
<h2>AOD-Net vs GridDehazeNet</h2>
<p class="note">Green = best value for that metric.
↑ = higher is better, ↓ = lower is better.</p>
<table>
<thead><tr><th>Metric</th>{model_headers}</tr></thead>
<tbody>{rows_html}</tbody>
</table>
<p class="note">Generated by model_comparison/compare.py</p>
</body></html>"""

    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'[Table] HTML report → {path}')
