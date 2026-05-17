import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
import json
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from datetime import datetime
import sys


def lighten_color(color, amount=0.5):
    """Mix `color` with white. amount=0 returns the original; amount=1 returns white."""
    rgb = mcolors.to_rgb(color)
    return tuple(c + (1 - c) * amount for c in rgb)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['axes.facecolor'] = 'white'

TARGET_MULTI_TRIAL_MODELS = {"GPT-5-mini", "o4-mini"}
RATE_METRICS = ['sensitivity', 'specificity', 'precision', 'accuracy', 'f1_score']
COUNT_METRICS = ['true_positive', 'false_positive', 'true_negative', 'false_negative']
ALL_METRICS = RATE_METRICS + COUNT_METRICS
METRIC_DECIMALS = {
    'sensitivity': 3,
    'specificity': 3,
    'precision': 3,
    'accuracy': 3,
    'f1_score': 3,
    'true_positive': 1,
    'false_positive': 1,
    'true_negative': 1,
    'false_negative': 1,
}

# 3-position X-axis: v1-dev → v2-dev → v2-test
XAXIS_ORDER = [
    ("v1", "prompt-set"),   # position 0
    ("v2", "prompt-set"),   # position 1
    ("v2", "test-set"),     # position 2
]
XAXIS_LABELS = [
    "Train\n(v1, n = 82)",
    "Train\n(v2, n = 82)",
    "Test\n(v2, n = 176)",
]


def load_results(filename="model_performance_results.json"):
    """Load results from JSON file"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f"📊 Loaded results from {filename}")
        print(f"Last updated: {data['metadata']['last_updated']}")
        print(f"Total results: {len(data['results'])}")
        return data
    except FileNotFoundError:
        print(f"❌ Results file {filename} not found. Please run the evaluation script first.")
        return None

def process_data(results_data):
    """Convert JSON results to structured format keyed by model/version/split."""
    if not results_data:
        return None

    data = {}
    for result in results_data['results']:
        split = result.get('split', 'prompt-set')
        if split == 'combined-prompt-test':
            continue

        model = result.get('model', 'Unknown')
        version = result.get('version', 'v1')
        metrics = result.get('metrics', {})

        model_data = data.setdefault(model, {})
        version_data = model_data.setdefault(version, {})
        split_entry = version_data.setdefault(split, {"trials": []})

        existing_trials = [trial['trial'] for trial in split_entry['trials']]
        trial_label = result.get('trial')
        if not trial_label:
            trial_label = f"trial-{len(existing_trials) + 1}"
        elif trial_label in existing_trials:
            trial_label = f"{trial_label}-{len(existing_trials) + 1}"

        split_entry['trials'].append({
            "trial": trial_label,
            "metrics": metrics,
            "file_path": result.get('file_path')
        })

    for model_data in data.values():
        for version_splits in model_data.values():
            for split, payload in version_splits.items():
                trials = payload.get("trials", [])
                metrics_records = [trial["metrics"] for trial in trials if trial.get("metrics")]

                if not metrics_records:
                    payload["summary"] = {"median": {}, "range": {}, "mean": {}, "ci95": {}}
                    payload["trial_count"] = 0
                    payload["metric_columns"] = []
                    payload["trial_labels"] = []
                    continue

                metrics_df = pd.DataFrame(metrics_records, dtype=float)
                metric_columns = metrics_df.columns.tolist()

                summary = {"median": {}, "range": {}, "mean": {}, "ci95": {}}
                for metric in metric_columns:
                    values = metrics_df[metric].dropna().astype(float)
                    if values.empty:
                        continue

                    median = float(values.median())
                    min_val = float(values.min())
                    max_val = float(values.max())
                    summary["median"][metric] = median
                    summary["range"][metric] = (min_val, max_val)
                    summary["mean"][metric] = float(values.mean())

                    if len(values) > 1:
                        std = float(values.std(ddof=1))
                        margin = 1.96 * std / (len(values) ** 0.5) if len(values) > 0 else 0.0
                        summary["ci95"][metric] = (
                            summary["mean"][metric] - margin,
                            summary["mean"][metric] + margin,
                        )
                    else:
                        summary["ci95"][metric] = (median, median)

                payload["summary"] = summary
                payload["trial_count"] = len(trials)
                payload["metric_columns"] = metric_columns
                payload["trial_labels"] = [trial["trial"] for trial in trials]

    return data


def should_include_range(model: str, version_data: dict, split: str) -> bool:
    """Return True when the visualization should display min-max bands."""
    return (model in TARGET_MULTI_TRIAL_MODELS
            and split == "test-set"
            and version_data.get("trial_count", 0) > 1)


def get_metric_summary(version_data: dict, metric: str):
    """Fetch the median value and range bounds for a metric."""
    summary = version_data.get("summary", {})
    median = summary.get("median", {}).get(metric)
    lower, upper = summary.get("range", {}).get(metric, (median, median))
    return median, (lower, upper)


def format_metric_value(summary: dict, metric: str, include_range: bool, decimals: int = 3) -> str:
    """Format a metric using its median and range (or provide alternatives)."""
    median = summary.get("median", {}).get(metric)
    if median is None:
        return "–"

    lower, upper = summary.get("range", {}).get(metric, (median, median))

    def _fmt(value: float, precision: int) -> str:
        if value is None:
            return "–"
        if abs(value - round(value)) < 10 ** (-(precision + 1)):
            return f"{int(round(value))}"
        return f"{value:.{precision}f}"

    median_fmt = _fmt(median, decimals)
    lower_fmt = _fmt(lower, decimals)
    upper_fmt = _fmt(upper, decimals)

    value_label = median_fmt
    if include_range:
        value_label = f"{median_fmt}\n({lower_fmt}-{upper_fmt})"

    return value_label

def create_publication_plot(data, metadata):
    """Create main figure with 6 subplots (A-F) showing dev and held-out test performance."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle(
        'Large Language Model Performance for ProteomeXchange Dataset Screening\n',
        fontsize=20,
        fontweight='bold',
        y=0.98,
    )

    # Use Okabe–Ito colorblind-safe palette
    model_colors = {
        'GPT-4o-mini':   "#CC79A7", 
        'o4-mini':       "#E69F00",  
        'GPT-4.1-mini':  "#009E73",  
        'GPT-5-mini':    "#0072B2",  
    }
    # Dev bars use a lightened shade; v2 test bars use the full color.
    DEV_LIGHTEN = 0.55
    model_color_pairs = {
        m: {'light': lighten_color(c, DEV_LIGHTEN), 'dark': c}
        for m, c in model_colors.items()
    }

    # Max trial count per model (used for legend R= label)
    replication_counts = {}
    for model, model_data in data.items():
        counts = []
        for version_splits in model_data.values():
            for split_data in version_splits.values():
                counts.append(split_data.get("trial_count", 0))
        replication_counts[model] = max(counts) if counts else 0

    models = list(data.keys())
    n_models = len(models)
    # Bars within a position group: total width ~0.8, split evenly across models.
    group_width = 0.8
    bar_width = group_width / max(n_models, 1)

    def plot_metric(ax, metric, title, ylabel, ylim=(0, 1.04)):
        x_positions = np.arange(len(XAXIS_ORDER))

        for pos_idx, (version, split) in enumerate(XAXIS_ORDER):
            shade_key = 'dark' if pos_idx == 2 else 'light'
            for i, model in enumerate(models):
                model_data = data[model]
                if version not in model_data or split not in model_data[version]:
                    continue
                split_data = model_data[version][split]
                median, (lower, upper) = get_metric_summary(split_data, metric)
                if median is None:
                    continue

                pair = model_color_pairs.get(model)
                color = pair[shade_key] if pair else 'gray'
                # Center the cluster of model bars around the position tick.
                offset = (i - (n_models - 1) / 2) * bar_width
                bar_x = pos_idx + offset

                ax.bar(bar_x, median, width=bar_width, color=color,
                       edgecolor='black', linewidth=0.6, zorder=2)

                # Error bars for multi-trial test-set
                if pos_idx == 2 and should_include_range(model, split_data, "test-set"):
                    if lower is not None and upper is not None and lower != upper:
                        ax.errorbar(bar_x, median,
                                    yerr=[[median - lower], [upper - median]],
                                    fmt='none', color='black',
                                    capsize=3, linewidth=1.2, zorder=3)

        # Extra title pad reserves space for the horizontal legend below the title.
        ax.set_title(title, fontsize=20, fontweight='bold', pad=32)
        ax.set_xlabel('Prompt Version and Evaluation Set', fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(XAXIS_LABELS, fontsize=13)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Legend: one entry per model (its darker/test shade), placed just below the title.
        legend_handles = [
            Patch(facecolor=model_colors.get(model, 'gray'),
                  edgecolor='black', linewidth=0.6, label=model)
            for model in models
        ]
        ax.legend(handles=legend_handles, loc='lower center',
                  bbox_to_anchor=(0.5, 1.0), ncol=len(legend_handles),
                  frameon=False, fontsize=11, handlelength=1.4,
                  columnspacing=1.2, borderaxespad=0.2)

    plot_metric(axes[0, 0], 'sensitivity', 'A. Sensitivity', 'Sensitivity')
    plot_metric(axes[0, 1], 'specificity', 'B. Specificity', 'Specificity')
    plot_metric(axes[1, 0], 'precision', 'C. Precision', 'Precision')
    plot_metric(axes[1, 1], 'accuracy', 'D. Accuracy', 'Accuracy')
    plot_metric(axes[2, 0], 'f1_score', 'E. F1 Score', 'F1 Score')

    # Plot F: ROC space — 3-point trajectory per model
    ax6 = axes[2, 1]
    for model, model_data in data.items():
        color = model_colors.get(model, 'gray')
        legend_label = f"{model}"

        roc_points = []
        for pos_idx, (version, split) in enumerate(XAXIS_ORDER):
            if version not in model_data or split not in model_data[version]:
                continue
            split_data = model_data[version][split]
            sens_median, (sens_low, sens_high) = get_metric_summary(split_data, 'sensitivity')
            spec_median, (spec_low, spec_high) = get_metric_summary(split_data, 'specificity')
            if sens_median is None or spec_median is None:
                continue
            fpr = 1 - spec_median
            fpr_low = (1 - spec_high) if spec_high is not None else fpr
            fpr_high = (1 - spec_low) if spec_low is not None else fpr
            roc_points.append({
                'pos_idx': pos_idx,
                'version': version,
                'split': split,
                'fpr': fpr,
                'sens': sens_median,
                'fpr_low': fpr_low,
                'fpr_high': fpr_high,
                'sens_low': sens_low,
                'sens_high': sens_high,
                'split_data': split_data,
            })

        if not roc_points:
            continue

        dev_pts = [p for p in roc_points if p['split'] == 'prompt-set']
        test_pts = [p for p in roc_points if p['split'] == 'test-set']

        # Solid line between dev points
        if len(dev_pts) >= 2:
            ax6.plot([p['fpr'] for p in dev_pts], [p['sens'] for p in dev_pts],
                     '-', alpha=0.5, color=color, linewidth=1.5)

        # Dashed connector from last dev to test
        if dev_pts and test_pts:
            ax6.plot([dev_pts[-1]['fpr'], test_pts[0]['fpr']],
                     [dev_pts[-1]['sens'], test_pts[0]['sens']],
                     '--', alpha=0.5, color=color, linewidth=1.5)

        added_legend = False
        for p in roc_points:
            if p['split'] == 'prompt-set':
                # Both v1 and v2 development points: hollow circles
                ax6.scatter([p['fpr']], [p['sens']], s=100, marker='o',
                            facecolors='none', edgecolors=color, linewidths=2.0,
                            label=legend_label if not added_legend else None)
                added_legend = True
            else:
                # v2 test: solid square, with error cross for multi-trial models
                ax6.scatter([p['fpr']], [p['sens']], s=120, marker='s',
                            facecolors=color, edgecolors=color, linewidths=2.0)
                if should_include_range(model, p['split_data'], 'test-set'):
                    if p['fpr_low'] is not None and p['fpr_high'] is not None and p['fpr_low'] != p['fpr_high']:
                        ax6.hlines(p['sens'], p['fpr_low'], p['fpr_high'],
                                   colors=color, alpha=0.35, linewidth=4)
                    if p['sens_low'] is not None and p['sens_high'] is not None and p['sens_low'] != p['sens_high']:
                        ax6.vlines(p['fpr'], p['sens_low'], p['sens_high'],
                                   colors=color, alpha=0.35, linewidth=4)

    ax6.set_title('F. ROC Space',
                  fontsize=20, fontweight='bold', pad=32)
    ax6.set_xlabel('1 - Specificity', fontsize=16)
    ax6.set_ylabel('Sensitivity', fontsize=16)
    ax6.set_xlim(0, 1.04)
    ax6.set_ylim(0, 1.04)
    ax6.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax6.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    handles, labels = ax6.get_legend_handles_labels()
    if handles:
        ax6.legend(handles=handles, loc='lower center',
                   bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
                   frameon=False, fontsize=11, handlelength=1.4,
                   columnspacing=1.2, borderaxespad=0.2)
    ax6.grid(True, alpha=0.3)
    ax6.tick_params(axis='both', which='major', labelsize=14)
    ax6.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)

    plt.tight_layout()
    return fig

def create_confusion_matrix_plots_by_version(data):
    """Create confusion matrix visualizations for each (version, split) position."""
    figures = {}

    for version, split in XAXIS_ORDER:
        key = f"{version}-{'dev' if split == 'prompt-set' else 'test'}"
        models_here = [model for model, model_data in data.items()
                       if version in model_data and split in model_data[version]]
        if not models_here:
            continue

        n_models = len(models_here)
        if n_models <= 2:
            rows, cols = 1, n_models
            figsize = (16, 5)
        else:
            rows, cols = 2, (n_models + 1) // 2
            figsize = (16, 10)

        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        split_label = 'Development' if split == 'prompt-set' else 'Held-Out Test'
        fig.suptitle(f'Confusion Matrices — {version} ({split_label})',
                     fontsize=24, fontweight='bold')

        for idx, model in enumerate(models_here):
            split_data = data[model][version][split]
            trial_count = split_data.get("trial_count", 1)
            include_range = should_include_range(model, split_data, split)

            median_summary = split_data.get("summary", {}).get("median", {})
            tn_med = median_summary.get('true_negative')
            fp_med = median_summary.get('false_positive')
            fn_med = median_summary.get('false_negative')
            tp_med = median_summary.get('true_positive')

            if None in {tn_med, fp_med, fn_med, tp_med}:
                axes[idx].set_title(f'{model} — Data Missing', fontsize=16)
                axes[idx].set_visible(False)
                continue

            cm_display = np.round([[tn_med, fp_med], [fn_med, tp_med]]).astype(int)

            annot_labels = np.full_like(cm_display, "", dtype=object)
            range_summary = split_data.get("summary", {}).get("range", {})
            metrics_for_cm = [
                ['true_negative', 'false_positive'],
                ['false_negative', 'true_positive']
            ]

            for i in range(2):
                for j in range(2):
                    metric = metrics_for_cm[i][j]
                    val = cm_display[i, j]
                    if include_range:
                        min_val, max_val = range_summary.get(metric, (val, val))
                        if min_val != max_val:
                            annot_labels[i, j] = f"{val}\n({int(min_val)}-{int(max_val)})"
                        else:
                            annot_labels[i, j] = str(val)
                    else:
                        annot_labels[i, j] = str(val)

            sns.heatmap(cm_display, annot=annot_labels, fmt='', cmap='Blues',
                        xticklabels=['Predicted Negative', 'Predicted Positive'],
                        yticklabels=['Actual Negative', 'Actual Positive'],
                        ax=axes[idx], annot_kws={'size': 14, 'va': 'center'})

            axes[idx].tick_params(axis='both', which='major', labelsize=18)
            axes[idx].set_title(f'{model} (n={trial_count})', fontsize=16, fontweight='bold')

        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        figures[key] = fig

    return figures

def create_summary_table(data):
    """Create a comprehensive summary table and return as a pandas DataFrame."""
    rows = []
    for model in sorted(data.keys()):
        for version in sorted(data[model].keys()):
            for split in sorted(data[model][version].keys()):
                version_data = data[model][version][split]
                summary = version_data.get("summary", {})
                trials = version_data.get("trial_count", 1)
                include_range = should_include_range(model, version_data, split)

                row = [
                    model,
                    version,
                    split,
                    trials,
                    format_metric_value(summary, 'sensitivity', include_range, METRIC_DECIMALS['sensitivity']),
                    format_metric_value(summary, 'specificity', include_range, METRIC_DECIMALS['specificity']),
                    format_metric_value(summary, 'precision', include_range, METRIC_DECIMALS['precision']),
                    format_metric_value(summary, 'accuracy', include_range, METRIC_DECIMALS['accuracy']),
                    format_metric_value(summary, 'f1_score', include_range, METRIC_DECIMALS['f1_score']),
                    format_metric_value(summary, 'true_positive', include_range, METRIC_DECIMALS['true_positive']),
                    format_metric_value(summary, 'false_positive', include_range, METRIC_DECIMALS['false_positive']),
                    format_metric_value(summary, 'true_negative', include_range, METRIC_DECIMALS['true_negative']),
                    format_metric_value(summary, 'false_negative', include_range, METRIC_DECIMALS['false_negative'])
                ]
                rows.append([str(item) for item in row])

    col_labels = ['Model', 'Version', 'Split', 'Trials', 'Sensitivity', 'Specificity',
                  'Precision', 'Accuracy', 'F1 Score', 'TP', 'FP', 'TN', 'FN']

    if not rows:
        return pd.DataFrame(columns=col_labels)

    df = pd.DataFrame(rows, columns=col_labels)
    return df

def create_heatmap(data):
    """Create heatmap with columns [v1-dev, v2-dev, v2-test] — split into 3 files."""
    models = list(data.keys())
    col_labels_display = [
        f"v1\n(dev)",
        f"v2\n(dev)",
        f"v2\n(test)",
    ]

    metrics = ['sensitivity', 'specificity', 'precision', 'accuracy', 'f1_score']
    titles = ['Sensitivity', 'Specificity', 'Precision', 'Accuracy', 'F1 Score']

    figures = []

    for fig_idx in range(3):
        start_idx = fig_idx * 2
        end_idx = min(start_idx + 2, len(metrics))

        if start_idx >= len(metrics):
            break

        n_plots = end_idx - start_idx
        if n_plots == 2:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        else:
            fig, axes = plt.subplots(1, 1, figsize=(8, 6))
            axes = [axes]

        for plot_idx, metric_idx in enumerate(range(start_idx, end_idx)):
            metric = metrics[metric_idx]
            title = titles[metric_idx]

            matrix = np.full((len(models), len(XAXIS_ORDER)), np.nan)
            for i, model in enumerate(models):
                for j, (version, split) in enumerate(XAXIS_ORDER):
                    if version in data[model] and split in data[model][version]:
                        summary = data[model][version][split].get("summary", {})
                        value = summary.get("median", {}).get(metric)
                        if value is not None:
                            matrix[i, j] = value

            mask = np.isnan(matrix)
            sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                       xticklabels=col_labels_display, yticklabels=models,
                       ax=axes[plot_idx], cbar_kws={'label': title},
                       mask=mask, annot_kws={'size': 12})

            axes[plot_idx].set_title(f'{title}', fontsize=20, fontweight='bold')
            axes[plot_idx].set_xlabel('Prompt Version / Evaluation Set', fontsize=14)
            axes[plot_idx].set_ylabel('Model', fontsize=14)
            axes[plot_idx].tick_params(axis='both', which='major', labelsize=12)

            cbar = axes[plot_idx].collections[0].colorbar
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label(title, fontsize=14)

        if fig_idx == 0:
            plt.suptitle('LLM Performance Heatmap for ProteomeXchange Dataset Screening',
                         fontsize=20, fontweight='bold')

        plt.tight_layout()
        figures.append(fig)

    return figures

def print_insights(data, metadata):
    """Print key insights from the analysis."""
    print("\n" + "="*60)
    print("KEY INSIGHTS:")
    print("="*60)

    all_combinations = []
    for model, versions in data.items():
        for version, splits in versions.items():
            for split, split_data in splits.items():
                medians = split_data.get("summary", {}).get("median", {})
                if not medians:
                    continue
                all_combinations.append((model, version, split, medians))

    for target_split in ['prompt-set', 'test-set']:
        subset = [(m, v, s, metrics) for m, v, s, metrics in all_combinations if s == target_split]
        if not subset:
            continue
        split_label = 'Prompt-set (dev)' if target_split == 'prompt-set' else 'Test-set (held-out)'
        print(f"\n{split_label}:")

        best_sensitivity = max(subset, key=lambda x: x[3].get('sensitivity', float('-inf')))
        best_specificity = max(subset, key=lambda x: x[3].get('specificity', float('-inf')))
        best_f1 = max(subset, key=lambda x: x[3].get('f1_score', float('-inf')))
        best_accuracy = max(subset, key=lambda x: x[3].get('accuracy', float('-inf')))

        print(f"  Best Sensitivity: {best_sensitivity[0]} {best_sensitivity[1]} ({best_sensitivity[3].get('sensitivity', 0):.3f})")
        print(f"  Best Specificity: {best_specificity[0]} {best_specificity[1]} ({best_specificity[3].get('specificity', 0):.3f})")
        print(f"  Best F1 Score:    {best_f1[0]} {best_f1[1]} ({best_f1[3].get('f1_score', 0):.3f})")
        print(f"  Best Accuracy:    {best_accuracy[0]} {best_accuracy[1]} ({best_accuracy[3].get('accuracy', 0):.3f})")

    print("\nVERSION PROGRESSION (prompt-set, v1 → v2):")
    for model, versions_data in data.items():
        versions = sorted(versions_data.keys())
        if len(versions) > 1:
            print(f"\n{model}:")
            for i in range(1, len(versions)):
                prev_v = versions[i-1]
                curr_v = versions[i]
                prev_summary = versions_data[prev_v].get('prompt-set', {}).get("summary", {}).get("median", {})
                curr_summary = versions_data[curr_v].get('prompt-set', {}).get("summary", {}).get("median", {})
                f1_change = curr_summary.get('f1_score', 0) - prev_summary.get('f1_score', 0)
                sens_change = curr_summary.get('sensitivity', 0) - prev_summary.get('sensitivity', 0)
                spec_change = curr_summary.get('specificity', 0) - prev_summary.get('specificity', 0)
                print(f"  {prev_v} → {curr_v}: F1 {f1_change:+.3f}, Sens {sens_change:+.3f}, Spec {spec_change:+.3f}")

    print("\nGENERALIZATION GAP (v2 dev → v2 test):")
    for model, versions_data in data.items():
        v2_dev = versions_data.get('v2', {}).get('prompt-set', {})
        v2_test = versions_data.get('v2', {}).get('test-set', {})
        if not v2_dev or not v2_test:
            continue
        dev_medians = v2_dev.get('summary', {}).get('median', {})
        test_medians = v2_test.get('summary', {}).get('median', {})
        f1_gap = test_medians.get('f1_score', 0) - dev_medians.get('f1_score', 0)
        sens_gap = test_medians.get('sensitivity', 0) - dev_medians.get('sensitivity', 0)
        spec_gap = test_medians.get('specificity', 0) - dev_medians.get('specificity', 0)
        trial_count = v2_test.get('trial_count', 1)
        trial_note = f" (n={trial_count} seeds)" if trial_count > 1 else ""
        print(f"  {model}{trial_note}: F1 {f1_gap:+.3f}, Sens {sens_gap:+.3f}, Spec {spec_gap:+.3f}")

def main():
    """Main function to run all visualizations"""
    results_data = load_results()
    if not results_data:
        return

    data = process_data(results_data)
    if not data:
        print("❌ No data to plot")
        return

    print(f"📊 Found data for models: {list(data.keys())}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("🎨 Creating visualizations...")

    # 1. Main performance plot
    fig1 = create_publication_plot(data, results_data['metadata'])
    fig1.savefig(f'ProteomeXchange_LLM_model_performance_{timestamp}.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig1.savefig(f'ProteomeXchange_LLM_model_performance_{timestamp}.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')

    # 2. Confusion matrices per (version, split)
    cm_figs = create_confusion_matrix_plots_by_version(data)
    for key, fig in cm_figs.items():
        fig.savefig(f'ProteomeXchange_LLM_confusion_matrices_{key}_{timestamp}.png', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

    # 3. Summary table
    df_summary = create_summary_table(data)
    csv_filename = f'ProteomeXchange_LLM_performance_summary_table_{timestamp}.csv'
    df_summary.to_csv(csv_filename, index=False)
    print('\n--- Summary table (CSV) ---\n')
    df_summary.to_csv(sys.stdout, index=False)

    # 4. Heatmaps
    heatmap_figs = create_heatmap(data)
    for i, fig in enumerate(heatmap_figs, 1):
        fig.savefig(f'ProteomeXchange_LLM_performance_heatmap_part{i}_{timestamp}.png', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

    plt.show()

    print(f"\n📈 Plots saved with timestamp {timestamp}:")
    print(f"1. ProteomeXchange_LLM_model_performance_{timestamp}.png/pdf")
    for key in cm_figs.keys():
        print(f"2. ProteomeXchange_LLM_confusion_matrices_{key}_{timestamp}.png")
    print(f"3. {csv_filename}")
    for i in range(1, len(heatmap_figs) + 1):
        print(f"4. ProteomeXchange_LLM_performance_heatmap_part{i}_{timestamp}.png")

    print_insights(data, results_data['metadata'])

if __name__ == "__main__":
    main()
