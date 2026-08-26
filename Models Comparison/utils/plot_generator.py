"""
Plot Generator — all graphical comparisons
==========================================
Produces the following plots in results/plots/:
  1.  bar_reference_metrics.png   — PSNR/SSIM/MSE/MAE grouped bars
  2.  bar_noreference_metrics.png — Sharpness/Fog/Color/Contrast/etc
  3.  radar_chart.png             — Spider/radar across all metrics
  4.  bar_speed.png               — Inference time + FPS
  5.  scatter_psnr_ssim.png       — Per-image PSNR vs SSIM scatter
  6.  boxplot_metrics.png         — Distribution of per-image scores
  7.  per_image_psnr.png          — PSNR per image (line chart)
  8.  per_image_ssim.png          — SSIM per image
  9.  gain_comparison.png         — Improvement over foggy baseline
  10. fog_density_plot.png        — Fog density remaining per model
  11. visual_comparison.png       — Side-by-side image samples
  12. heatmap_correlation.png     — Metric correlation heatmap
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# Colour palette (consistent across all plots)
PALETTE  = {'AOD-Net': '#4C9BE8', 'GridDehazeNet': '#E8724C'}
DARK_BG  = '#0F0F14'
PANEL_BG = '#1A1A24'
GRID_COL = '#2A2A38'
TEXT_COL = '#DDDDEE'

plt.rcParams.update({
    'figure.facecolor': DARK_BG, 'axes.facecolor':  PANEL_BG,
    'axes.edgecolor':   GRID_COL,'axes.labelcolor':  TEXT_COL,
    'xtick.color':      TEXT_COL,'ytick.color':      TEXT_COL,
    'text.color':       TEXT_COL,'grid.color':       GRID_COL,
    'grid.alpha':       0.4,     'legend.facecolor': PANEL_BG,
    'legend.edgecolor': GRID_COL,'font.size':        11,
    'axes.titlesize':   13,      'axes.labelsize':   11,
})

MODEL_NAMES = ['AOD-Net', 'GridDehazeNet']


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _color(name): return PALETTE.get(name, '#888888')
def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches='tight',
                facecolor=DARK_BG, edgecolor='none')
    plt.close(fig)
    print(f'[Plot] → {path}')

def _get_means(results, keys):
    """Return {name: [mean_val, …]} for given metric keys."""
    out = {}
    for n in MODEL_NAMES:
        if n not in results: continue
        means = results[n]['mean']
        out[n] = [means.get(k) for k in keys]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 1. Bar — Reference-based metrics
# ─────────────────────────────────────────────────────────────────────────────

def plot_reference_metrics(results, out_dir):
    keys    = ['psnr','ssim','rmse','mae']
    labels  = ['PSNR (dB)↑','SSIM↑','RMSE↓','MAE↓']
    hi_good = [True,True,False,False]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle('Reference-Based Quality Metrics', fontsize=15, y=1.02)

    for ax, key, label, higher in zip(axes, keys, labels, hi_good):
        vals = [results[n]['mean'].get(key) for n in MODEL_NAMES if n in results]
        stds = [results[n]['std'].get(key,0) for n in MODEL_NAMES if n in results]
        names_present = [n for n in MODEL_NAMES if n in results]

        if all(v is None for v in vals):
            ax.text(0.5,0.5,'N/A\n(no reference)',ha='center',va='center',
                    transform=ax.transAxes, color='#888')
            ax.set_title(label); continue

        vals = [v if v is not None else 0 for v in vals]
        colors = [_color(n) for n in names_present]
        bars = ax.bar(names_present, vals, color=colors,
                      yerr=stds, capsize=5, error_kw={'color':'#aaa','lw':1.5},
                      edgecolor='none', width=0.55)

        # Highlight winner
        best_i = (vals.index(max(vals)) if higher
                  else vals.index(min(vals)))
        bars[best_i].set_edgecolor('#FFDD44')
        bars[best_i].set_linewidth(2.5)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(stds)*0.1,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9,
                    color=TEXT_COL)

        ax.set_title(label)
        ax.set_ylim(0, max(vals)*1.25 if max(vals)>0 else 1)
        ax.grid(axis='y', alpha=0.4)
        ax.set_axisbelow(True)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'bar_reference_metrics.png'))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Bar — No-reference metrics
# ─────────────────────────────────────────────────────────────────────────────

def plot_noreference_metrics(results, out_dir):
    groups = [
        (['sharpness','sharpness_gain'],  ['Sharpness↑','Sharpness Gain↑'],  [True,True]),
        (['fog_density','fog_reduction'],  ['Fog Density↓','Fog Reduction↑'], [False,True]),
        (['colorfulness','color_gain'],    ['Colorfulness↑','Color Gain↑'],   [True,True]),
        (['contrast','contrast_gain'],     ['Contrast↑','Contrast Gain↑'],    [True,True]),
        (['lane_contrast','entropy'],      ['Lane Contrast↑','Entropy↑'],     [True,True]),
        (['visibility','dark_channel'],    ['Visibility↑','Dark Channel↓'],   [True,False]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('No-Reference & Autonomous Driving Metrics', fontsize=15)

    for ax, (keys, labels, hi_good) in zip(axes.flat, groups):
        x = np.arange(len(keys))
        w = 0.35
        names_p = [n for n in MODEL_NAMES if n in results]
        for i, n in enumerate(names_p):
            means = results[n]['mean']
            stds  = results[n]['std']
            vals  = [means.get(k, 0) or 0 for k in keys]
            errs  = [stds.get(k, 0) or 0 for k in keys]
            ax.bar(x + i*w - w/2*(len(names_p)-1)/len(names_p),
                   vals, w, label=n, color=_color(n),
                   yerr=errs, capsize=4,
                   error_kw={'color':'#aaa','lw':1},
                   edgecolor='none')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.4)
        ax.set_axisbelow(True)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'bar_noreference_metrics.png'))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Radar / Spider chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_radar(results, out_dir):
    """Normalised radar chart across all key metrics."""
    keys = [
        ('psnr',True),('ssim',True),('sharpness',True),
        ('fog_reduction',True),('colorfulness',True),
        ('lane_contrast',True),('contrast',True),
        ('visibility',True),('entropy',True),
    ]
    labels = ['PSNR','SSIM','Sharpness','Fog\nReduction',
              'Colorful','Lane\nContrast','Contrast','Visibility','Entropy']
    N = len(keys)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9,9),
                           subplot_kw=dict(polar=True))
    ax.set_facecolor(PANEL_BG)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)

    # Collect & normalise values
    all_vals = {}
    for n in MODEL_NAMES:
        if n not in results: continue
        means = results[n]['mean']
        all_vals[n] = [means.get(k, 0) or 0 for k,_ in keys]

    # Normalise to 0-1 per metric
    for i in range(N):
        col = [all_vals[n][i] for n in all_vals]
        col_min, col_max = min(col), max(col)
        if col_max > col_min:
            for n in all_vals:
                all_vals[n][i] = (all_vals[n][i] - col_min) / (col_max - col_min)
        else:
            for n in all_vals:
                all_vals[n][i] = 0.5

    for n, vals in all_vals.items():
        v = vals + vals[:1]
        ax.plot(angles, v, 'o-', lw=2.5, color=_color(n), label=n, markersize=5)
        ax.fill(angles, v, alpha=0.12, color=_color(n))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, color=TEXT_COL)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25,0.50,0.75,1.0])
    ax.set_yticklabels(['0.25','0.50','0.75','1.0'], fontsize=8, color='#888')
    ax.yaxis.grid(True, color=GRID_COL, alpha=0.5)
    ax.xaxis.grid(True, color=GRID_COL, alpha=0.5)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35,1.1), fontsize=11)
    ax.set_title('Normalised Metric Radar\n(larger = better for all axes)',
                 pad=20, fontsize=13)

    _save(fig, os.path.join(out_dir, 'radar_chart.png'))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Speed comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_speed(results, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Speed & Efficiency Comparison', fontsize=14)

    names_p = [n for n in MODEL_NAMES if n in results]
    colors  = [_color(n) for n in names_p]

    def _bar(ax, vals, ylabel, title, higher_better=True):
        bars = ax.bar(names_p, vals, color=colors, edgecolor='none', width=0.5)
        best_i = (vals.index(max(vals)) if higher_better
                  else vals.index(min(vals)))
        bars[best_i].set_edgecolor('#FFDD44')
        bars[best_i].set_linewidth(2.5)
        for bar,v in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=10)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.grid(axis='y', alpha=0.4); ax.set_axisbelow(True)
        ax.set_ylim(0, max(vals)*1.3)

    _bar(axes[0], [results[n]['speed']['mean_ms'] for n in names_p],
         'ms / frame', 'Inference Time ↓', False)
    _bar(axes[1], [results[n]['speed']['fps'] for n in names_p],
         'FPS', 'Frames Per Second ↑', True)
    _bar(axes[2], [results[n]['model_stats']['params_M'] for n in names_p],
         'Million parameters', 'Model Size (params) ↓', False)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'bar_speed.png'))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Scatter PSNR vs SSIM
# ─────────────────────────────────────────────────────────────────────────────

def plot_scatter_psnr_ssim(results, out_dir):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title('Per-Image PSNR vs SSIM')

    for n in MODEL_NAMES:
        if n not in results: continue
        rows  = results[n]['per_image']
        psnrs = [r.get('psnr') for r in rows if r.get('psnr') is not None]
        ssims = [r.get('ssim') for r in rows if r.get('ssim') is not None]
        if not psnrs: continue
        ax.scatter(psnrs, ssims, c=_color(n), label=n,
                   s=70, alpha=0.8, edgecolors='white', linewidths=0.5)
        # Mean cross
        ax.axvline(np.mean(psnrs), color=_color(n), lw=1.5, ls='--', alpha=0.5)
        ax.axhline(np.mean(ssims), color=_color(n), lw=1.5, ls='--', alpha=0.5)

    ax.set_xlabel('PSNR (dB) ↑')
    ax.set_ylabel('SSIM ↑')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if all(all(r.get('psnr') is None for r in results[n]['per_image'])
           for n in MODEL_NAMES if n in results):
        ax.text(0.5,0.5,'N/A — no reference images provided',
                ha='center',va='center',transform=ax.transAxes,color='#888')

    _save(fig, os.path.join(out_dir, 'scatter_psnr_ssim.png'))


# ─────────────────────────────────────────────────────────────────────────────
# 6. Box plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_boxplots(results, out_dir):
    metrics = [
        ('sharpness',    'Sharpness'),
        ('fog_density',  'Fog Density'),
        ('colorfulness', 'Colorfulness'),
        ('lane_contrast','Lane Contrast'),
        ('entropy',      'Entropy'),
        ('contrast',     'Contrast'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('Distribution of Metrics Across Test Images', fontsize=14)

    for ax, (key, label) in zip(axes.flat, metrics):
        data, ticks, cols = [], [], []
        for n in MODEL_NAMES:
            if n not in results: continue
            vals = [r.get(key) for r in results[n]['per_image']
                    if r.get(key) is not None]
            if vals:
                data.append(vals); ticks.append(n); cols.append(_color(n))

        if data:
            bp = ax.boxplot(data, patch_artist=True, notch=False,
                            medianprops={'color':'#FFDD44','lw':2},
                            whiskerprops={'color':'#888'},
                            capprops={'color':'#888'},
                            flierprops={'marker':'o','color':'#888',
                                        'markerfacecolor':'#888','ms':4})
            for patch, col in zip(bp['boxes'], cols):
                patch.set_facecolor(col)
                patch.set_alpha(0.7)
            ax.set_xticks(range(1, len(ticks)+1))
            ax.set_xticklabels(ticks, fontsize=9)
        ax.set_title(label)
        ax.grid(axis='y', alpha=0.4)
        ax.set_axisbelow(True)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'boxplot_metrics.png'))


# ─────────────────────────────────────────────────────────────────────────────
# 7 & 8. Per-image line charts
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_image(results, out_dir, key, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title(title)

    for n in MODEL_NAMES:
        if n not in results: continue
        rows = results[n]['per_image']
        vals = [r.get(key) for r in rows]
        if all(v is None for v in vals): continue
        xs   = list(range(len(vals)))
        ax.plot(xs, vals, 'o-', color=_color(n), label=n,
                lw=2, markersize=5, alpha=0.9)
        ax.axhline(np.nanmean([v for v in vals if v]),
                   color=_color(n), ls='--', lw=1, alpha=0.5)

    names = [r['name'] for r in next(iter(results.values()))['per_image']]
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, filename))


# ─────────────────────────────────────────────────────────────────────────────
# 9. Gain comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_gain_comparison(results, out_dir):
    gains = [
        ('psnr_gain',    'PSNR Gain (dB)'),
        ('ssim_gain',    'SSIM Gain'),
        ('sharpness_gain','Sharpness Gain'),
        ('fog_reduction','Fog Reduction'),
        ('color_gain',   'Colorfulness Gain'),
        ('contrast_gain','Contrast Gain'),
    ]
    names_p = [n for n in MODEL_NAMES if n in results]
    x = np.arange(len(gains))
    w = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title('Improvement Over Foggy Input (Gain Metrics)')

    for i, n in enumerate(names_p):
        vals = [results[n]['mean'].get(k, 0) or 0 for k,_ in gains]
        bars = ax.bar(x + i*w - w*(len(names_p)-1)/2, vals, w,
                      label=n, color=_color(n), edgecolor='none', alpha=0.85)
        for bar,v in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+(0.001 if v>=0 else -0.001),
                    f'{v:+.3f}', ha='center',
                    va='bottom' if v>=0 else 'top',
                    fontsize=7.5, color=TEXT_COL)

    ax.axhline(0, color='#888', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([l for _,l in gains], rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylabel('Gain over foggy input')

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'gain_comparison.png'))


# ─────────────────────────────────────────────────────────────────────────────
# 10. Fog density per image
# ─────────────────────────────────────────────────────────────────────────────

def plot_fog_density(results, out_dir):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title('Residual Fog Density Per Image (Lower = Better Dehazing)')

    names_p = [n for n in MODEL_NAMES if n in results]
    # Also show foggy baseline if stored
    for n in names_p:
        rows = results[n]['per_image']
        vals = [r.get('fog_density', 0) for r in rows]
        ax.plot(vals, 'o-', color=_color(n), label=n, lw=2, markersize=5)

    img_names = [r['name'] for r in next(iter(results.values()))['per_image']]
    ax.set_xticks(range(len(img_names)))
    ax.set_xticklabels(img_names, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Fog Density Score (0=clear, 1=dense fog)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'fog_density_plot.png'))


# ─────────────────────────────────────────────────────────────────────────────
# 11. Visual comparison grid
# ─────────────────────────────────────────────────────────────────────────────

def plot_visual_comparison(results, out_dir,
                            foggy_dir: str,
                            enhanced_cache: dict,
                            n_samples: int = 4):
    """Side-by-side visual comparison of foggy/AOD-Net/GridDehazeNet."""
    import glob
    paths = sorted(glob.glob(os.path.join(foggy_dir,'*.png')))[:n_samples]
    if not paths:
        return

    model_names_p = [n for n in MODEL_NAMES if n in results]
    cols  = 1 + len(model_names_p)
    rows  = len(paths)
    pw,ph = 280, 180

    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.2, rows*2.2))
    if rows == 1: axes = axes[np.newaxis,:]
    fig.suptitle('Visual Comparison: Foggy → Enhanced', fontsize=13)

    labels = ['Foggy Input'] + [n for n in model_names_p]
    for r, fp in enumerate(paths):
        img_name = Path(fp).stem
        foggy    = cv2.imread(fp)
        foggy    = cv2.resize(foggy, (pw,ph)) if foggy is not None else np.zeros((ph,pw,3),np.uint8)
        frames   = [foggy] + [cv2.resize(enhanced_cache.get((n,img_name),
                               np.zeros((ph,pw,3),np.uint8)), (pw,ph))
                               for n in model_names_p]
        for c, (frame, label) in enumerate(zip(frames, labels)):
            ax = axes[r,c]
            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ax.set_title(f'{label}\n{img_name}' if r==0 else img_name,
                         fontsize=8, pad=3)
            ax.axis('off')

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'visual_comparison.png'))


# ─────────────────────────────────────────────────────────────────────────────
# 12. Correlation heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(results, out_dir):
    import pandas as pd

    metric_keys = ['sharpness','fog_density','colorfulness','contrast',
                   'lane_contrast','visibility','entropy','dark_channel']
    if results[MODEL_NAMES[0]]['mean'].get('psnr') is not None:
        metric_keys = ['psnr','ssim','mse'] + metric_keys

    fig, axes = plt.subplots(1, len([n for n in MODEL_NAMES if n in results]),
                              figsize=(10*len([n for n in MODEL_NAMES if n in results])//2 + 2, 9))
    if not hasattr(axes, '__len__'): axes = [axes]

    names_p = [n for n in MODEL_NAMES if n in results]
    for ax, n in zip(axes, names_p):
        rows = results[n]['per_image']
        data = {k: [r.get(k) for r in rows] for k in metric_keys}
        df   = pd.DataFrame(data).dropna(axis=1, how='all').dropna()
        if df.empty or df.shape[1] < 2:
            ax.text(0.5,0.5,'N/A',ha='center',va='center',transform=ax.transAxes)
            continue

        corr  = df.corr()
        im    = ax.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1,
                          aspect='auto')
        ticks = list(corr.columns)
        ax.set_xticks(range(len(ticks))); ax.set_xticklabels(ticks, rotation=40,
                                                              ha='right', fontsize=9)
        ax.set_yticks(range(len(ticks))); ax.set_yticklabels(ticks, fontsize=9)
        ax.set_title(f'{n}\nMetric Correlation', fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for i in range(len(ticks)):
            for j in range(len(ticks)):
                ax.text(j, i, f'{corr.values[i,j]:.2f}',
                        ha='center', va='center', fontsize=7,
                        color='black' if abs(corr.values[i,j])<0.6 else 'white')

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'heatmap_correlation.png'))


# ─────────────────────────────────────────────────────────────────────────────
# Master caller
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_plots(results: dict,
                       out_dir: str,
                       foggy_dir: str = None,
                       enhanced_cache: dict = None) -> None:
    """Generate all plots. enhanced_cache = {(model_name, img_name): bgr_array}."""
    print('\n[Plots] Generating all visualisations …')
    plot_reference_metrics(results, out_dir)
    plot_noreference_metrics(results, out_dir)
    plot_radar(results, out_dir)
    plot_speed(results, out_dir)
    plot_scatter_psnr_ssim(results, out_dir)
    plot_boxplots(results, out_dir)
    plot_per_image(results, out_dir, 'sharpness',
                   'Sharpness (Laplacian var)', 'Sharpness Per Image',
                   'per_image_sharpness.png')
    plot_per_image(results, out_dir, 'psnr', 'PSNR (dB)',
                   'PSNR Per Image', 'per_image_psnr.png')
    plot_per_image(results, out_dir, 'ssim', 'SSIM',
                   'SSIM Per Image', 'per_image_ssim.png')
    plot_gain_comparison(results, out_dir)
    plot_fog_density(results, out_dir)
    if foggy_dir and enhanced_cache:
        plot_visual_comparison(results, out_dir, foggy_dir, enhanced_cache)
    plot_correlation_heatmap(results, out_dir)
    print(f'[Plots] All plots saved to {out_dir}/')
