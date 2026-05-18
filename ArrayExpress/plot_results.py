import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
import json
from matplotlib.patches import Patch
from datetime import datetime
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent


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

# X-axis on subplots A-E: per-disease panels in this order.
PER_DISEASE_ORDER = ["AD", "LBD", "ALS-FTD"]
# Subplot F (ROC) uses the combined all-diseases values.
ROC_DISEASE = "all-diseases"

MODEL_COLORS = {
    'GPT-4o-mini':   "#CC79A7",
    'o4-mini':       "#E69F00",
    'GPT-4.1-mini':  "#009E73",
    'GPT-5-mini':    "#0072B2",
    'Ensemble':   "#D55E00",
}

MODEL_ORDER = ['GPT-4o-mini', 'GPT-4.1-mini', 'GPT-5-mini', 'o4-mini', 'Ensemble']


def canonical_model_name(model: str) -> str:
    """Normalize model names written by older evaluation scripts."""
    if str(model).startswith("ensemble"):
        return "Ensemble"
    return model


def trial_label_for_result(result: dict, existing_trials: list[str]) -> str:
    """
    Label repeated GPT-5-mini/o4-mini runs consistently.

    The unseeded base file is created with seed 42 by json-to-openai-arrayexpress.py,
    while repetition files use explicit seed suffixes 43-51.
    """
    model = canonical_model_name(result.get('model', 'Unknown'))
    trial_label = result.get('trial')
    if not trial_label and model in TARGET_MULTI_TRIAL_MODELS:
        trial_label = "seed42"
    if not trial_label:
        trial_label = f"trial-{len(existing_trials) + 1}"
    elif trial_label in existing_trials:
        trial_label = f"{trial_label}-{len(existing_trials) + 1}"
    return trial_label

def load_results(filename=SCRIPT_DIR / "model_performance_results.json"):
    """Load results from JSON file"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f"Loaded results from {filename}")
        print(f"Last updated: {data['metadata']['last_updated']}")
        print(f"Total results: {len(data['results'])}")
        return data
    except FileNotFoundError:
        print(f"Results file {filename} not found. Please run the evaluation script first.")
        return None


def process_data(results_data):
    """Convert JSON results to structured format keyed by model/version/target_disease."""
    if not results_data:
        return None

    data = {}
    for result in results_data['results']:
        target_disease = result.get('target_disease', '')
        if not target_disease:
            continue

        model = canonical_model_name(result.get('model', 'Unknown'))
        version = result.get('version', 'v1')
        metrics = result.get('metrics', {})

        model_data = data.setdefault(model, {})
        version_data = model_data.setdefault(version, {})
        disease_entry = version_data.setdefault(target_disease, {"trials": []})

        existing_trials = [trial['trial'] for trial in disease_entry['trials']]
        trial_label = trial_label_for_result(result, existing_trials)

        disease_entry['trials'].append({
            "trial": trial_label,
            "metrics": metrics,
            "file_path": result.get('file_path'),
        })

    for model_data in data.values():
        for version_diseases in model_data.values():
            for disease, payload in version_diseases.items():
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


def should_include_range(model: str, disease_data: dict) -> bool:
    """Return True when the visualization should display min-max bands."""
    return (model in TARGET_MULTI_TRIAL_MODELS
            and disease_data.get("trial_count", 0) > 1)


def get_metric_summary(disease_data: dict, metric: str):
    """Fetch the median value and range bounds for a metric."""
    summary = disease_data.get("summary", {})
    median = summary.get("median", {}).get(metric)
    lower, upper = summary.get("range", {}).get(metric, (median, median))
    return median, (lower, upper)


def format_metric_value(summary: dict, metric: str, include_range: bool, decimals: int = 3) -> str:
    """Format a metric using its median and range (or provide alternatives)."""
    median = summary.get("median", {}).get(metric)
    if median is None:
        return "-"

    lower, upper = summary.get("range", {}).get(metric, (median, median))

    def _fmt(value: float, precision: int) -> str:
        if value is None:
            return "-"
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


def model_keys_in_data(data):
    """Stable model order: known models first, then any extras alphabetically."""
    keys = [m for m in MODEL_ORDER if m in data]
    extras = sorted(m for m in data if m not in MODEL_ORDER)
    return keys + extras


def latest_version_for_model(model_data):
    """Pick the latest version available for a model (e.g., v2 over v1)."""
    versions = sorted(model_data.keys())
    return versions[-1] if versions else None


def get_disease_counts(data):
    """Pull the per-disease dataset size (total_files_evaluated) from the data."""
    counts = {}
    for model_data in data.values():
        for version_diseases in model_data.values():
            for disease, payload in version_diseases.items():
                if disease in counts:
                    continue
                n = payload.get("summary", {}).get("median", {}).get("total_files_evaluated")
                if n:
                    counts[disease] = int(n)
    return counts


def make_xaxis_labels(data):
    counts = get_disease_counts(data)
    return [f"{disease}\n(n = {counts.get(disease, '?')})" for disease in PER_DISEASE_ORDER]


def create_publication_plot(data, metadata):
    """Create main figure with 6 subplots (A-F): per-disease metrics + all-diseases ROC."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle(
        'Large Language Model Performance for ArrayExpress Dataset Screening\n',
        fontsize=20,
        fontweight='bold',
        y=0.98,
    )

    models = model_keys_in_data(data)
    n_models = len(models)
    group_width = 0.8
    bar_width = group_width / max(n_models, 1)
    xaxis_labels = make_xaxis_labels(data)

    def plot_metric(ax, metric, title, ylabel, ylim=(0, 1.04)):
        x_positions = np.arange(len(PER_DISEASE_ORDER))

        for pos_idx, disease in enumerate(PER_DISEASE_ORDER):
            for i, model in enumerate(models):
                model_data = data[model]
                version = latest_version_for_model(model_data)
                if version is None or disease not in model_data[version]:
                    continue
                disease_data = model_data[version][disease]
                median, (lower, upper) = get_metric_summary(disease_data, metric)
                if median is None:
                    continue

                color = MODEL_COLORS.get(model, 'gray')
                offset = (i - (n_models - 1) / 2) * bar_width
                bar_x = pos_idx + offset

                ax.bar(bar_x, median, width=bar_width, color=color,
                       edgecolor='black', linewidth=0.6, zorder=2)

                if should_include_range(model, disease_data):
                    if lower is not None and upper is not None and lower != upper:
                        ax.errorbar(bar_x, median,
                                    yerr=[[median - lower], [upper - median]],
                                    fmt='none', color='black',
                                    capsize=3, linewidth=1.2, zorder=3)

        ax.set_title(title, fontsize=20, fontweight='bold', pad=32)
        ax.set_xlabel('Target Disease', fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(xaxis_labels, fontsize=13)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', which='major', labelsize=12)

        legend_handles = [
            Patch(facecolor=MODEL_COLORS.get(model, 'gray'),
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

    # Plot F: ROC space — one point per model using all-diseases sensitivity/specificity.
    ax6 = axes[2, 1]
    for model in models:
        model_data = data[model]
        version = latest_version_for_model(model_data)
        if version is None or ROC_DISEASE not in model_data[version]:
            continue
        disease_data = model_data[version][ROC_DISEASE]
        sens_median, (sens_low, sens_high) = get_metric_summary(disease_data, 'sensitivity')
        spec_median, (spec_low, spec_high) = get_metric_summary(disease_data, 'specificity')
        if sens_median is None or spec_median is None:
            continue

        color = MODEL_COLORS.get(model, 'gray')
        fpr = 1 - spec_median
        ax6.scatter([fpr], [sens_median], s=140, marker='s',
                    facecolors=color, edgecolors=color, linewidths=2.0,
                    label=model, zorder=3)

        if should_include_range(model, disease_data):
            fpr_low = (1 - spec_high) if spec_high is not None else fpr
            fpr_high = (1 - spec_low) if spec_low is not None else fpr
            if fpr_low != fpr_high:
                ax6.hlines(sens_median, fpr_low, fpr_high,
                           colors=color, alpha=0.35, linewidth=4)
            if sens_low is not None and sens_high is not None and sens_low != sens_high:
                ax6.vlines(fpr, sens_low, sens_high,
                           colors=color, alpha=0.35, linewidth=4)

    ax6.set_title('F. ROC Space (all-diseases)',
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


def create_confusion_matrix_plots_by_disease(data):
    """Create confusion matrix visualizations for each target_disease."""
    figures = {}
    diseases = PER_DISEASE_ORDER + [ROC_DISEASE]

    for disease in diseases:
        models_here = []
        for model in model_keys_in_data(data):
            model_data = data[model]
            version = latest_version_for_model(model_data)
            if version is not None and disease in model_data[version]:
                models_here.append(model)
        if not models_here:
            continue

        n_models = len(models_here)
        if n_models <= 3:
            rows, cols = 1, n_models
            figsize = (5 * n_models, 5)
        else:
            rows, cols = 2, (n_models + 1) // 2
            figsize = (16, 10)

        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        fig.suptitle(f'Confusion Matrices — {disease}',
                     fontsize=24, fontweight='bold')

        for idx, model in enumerate(models_here):
            model_data = data[model]
            version = latest_version_for_model(model_data)
            disease_data = model_data[version][disease]
            trial_count = disease_data.get("trial_count", 1)
            include_range = should_include_range(model, disease_data)

            median_summary = disease_data.get("summary", {}).get("median", {})
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
            range_summary = disease_data.get("summary", {}).get("range", {})
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

            axes[idx].tick_params(axis='both', which='major', labelsize=14)
            axes[idx].set_title(f'{model} (n={trial_count})', fontsize=16, fontweight='bold')

        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        figures[disease] = fig

    return figures


def create_summary_table(data):
    """Create a comprehensive summary table and return as a pandas DataFrame."""
    rows = []
    for model in sorted(data.keys()):
        for version in sorted(data[model].keys()):
            for disease in sorted(data[model][version].keys()):
                disease_data = data[model][version][disease]
                summary = disease_data.get("summary", {})
                trials = disease_data.get("trial_count", 1)
                include_range = should_include_range(model, disease_data)

                row = [
                    model,
                    version,
                    disease,
                    trials,
                    format_metric_value(summary, 'sensitivity', include_range, METRIC_DECIMALS['sensitivity']),
                    format_metric_value(summary, 'specificity', include_range, METRIC_DECIMALS['specificity']),
                    format_metric_value(summary, 'precision', include_range, METRIC_DECIMALS['precision']),
                    format_metric_value(summary, 'accuracy', include_range, METRIC_DECIMALS['accuracy']),
                    format_metric_value(summary, 'f1_score', include_range, METRIC_DECIMALS['f1_score']),
                    format_metric_value(summary, 'true_positive', include_range, METRIC_DECIMALS['true_positive']),
                    format_metric_value(summary, 'false_positive', include_range, METRIC_DECIMALS['false_positive']),
                    format_metric_value(summary, 'true_negative', include_range, METRIC_DECIMALS['true_negative']),
                    format_metric_value(summary, 'false_negative', include_range, METRIC_DECIMALS['false_negative']),
                ]
                rows.append([str(item) for item in row])

    col_labels = ['Model', 'Version', 'Target_Disease', 'Trials', 'Sensitivity', 'Specificity',
                  'Precision', 'Accuracy', 'F1 Score', 'TP', 'FP', 'TN', 'FN']

    if not rows:
        return pd.DataFrame(columns=col_labels)
    return pd.DataFrame(rows, columns=col_labels)


def create_heatmap(data):
    """Create heatmap with columns [AD, LBD, ALS-FTD, all-diseases] — split into files."""
    models = model_keys_in_data(data)
    diseases = PER_DISEASE_ORDER + [ROC_DISEASE]

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

            matrix = np.full((len(models), len(diseases)), np.nan)
            for i, model in enumerate(models):
                model_data = data[model]
                version = latest_version_for_model(model_data)
                if version is None:
                    continue
                for j, disease in enumerate(diseases):
                    if disease in model_data[version]:
                        summary = model_data[version][disease].get("summary", {})
                        value = summary.get("median", {}).get(metric)
                        if value is not None:
                            matrix[i, j] = value

            mask = np.isnan(matrix)
            sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                       xticklabels=diseases, yticklabels=models,
                       ax=axes[plot_idx], cbar_kws={'label': title},
                       mask=mask, annot_kws={'size': 12})

            axes[plot_idx].set_title(f'{title}', fontsize=20, fontweight='bold')
            axes[plot_idx].set_xlabel('Target Disease', fontsize=14)
            axes[plot_idx].set_ylabel('Model', fontsize=14)
            axes[plot_idx].tick_params(axis='both', which='major', labelsize=12)

            cbar = axes[plot_idx].collections[0].colorbar
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label(title, fontsize=14)

        if fig_idx == 0:
            plt.suptitle('LLM Performance Heatmap for ArrayExpress Dataset Screening',
                         fontsize=20, fontweight='bold')

        plt.tight_layout()
        figures.append(fig)

    return figures


def print_insights(data, metadata):
    """Print key insights from the analysis."""
    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("=" * 60)

    all_combinations = []
    for model, versions in data.items():
        for version, diseases in versions.items():
            for disease, disease_data in diseases.items():
                medians = disease_data.get("summary", {}).get("median", {})
                if not medians:
                    continue
                all_combinations.append((model, version, disease, medians))

    for target_disease in PER_DISEASE_ORDER + [ROC_DISEASE]:
        subset = [(m, v, d, metrics) for m, v, d, metrics in all_combinations if d == target_disease]
        if not subset:
            continue
        print(f"\n{target_disease}:")

        best_sensitivity = max(subset, key=lambda x: x[3].get('sensitivity', float('-inf')))
        best_specificity = max(subset, key=lambda x: x[3].get('specificity', float('-inf')))
        best_f1 = max(subset, key=lambda x: x[3].get('f1_score', float('-inf')))
        best_accuracy = max(subset, key=lambda x: x[3].get('accuracy', float('-inf')))

        print(f"  Best Sensitivity: {best_sensitivity[0]} {best_sensitivity[1]} ({best_sensitivity[3].get('sensitivity', 0):.3f})")
        print(f"  Best Specificity: {best_specificity[0]} {best_specificity[1]} ({best_specificity[3].get('specificity', 0):.3f})")
        print(f"  Best F1 Score:    {best_f1[0]} {best_f1[1]} ({best_f1[3].get('f1_score', 0):.3f})")
        print(f"  Best Accuracy:    {best_accuracy[0]} {best_accuracy[1]} ({best_accuracy[3].get('accuracy', 0):.3f})")


def main():
    """Main function to run all visualizations"""
    results_data = load_results()
    if not results_data:
        return

    data = process_data(results_data)
    if not data:
        print("No data to plot")
        return

    print(f"Found data for models: {list(data.keys())}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Creating visualizations...")

    fig1 = create_publication_plot(data, results_data['metadata'])
    fig1.savefig(SCRIPT_DIR / f'ArrayExpress_LLM_model_performance_{timestamp}.png', dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig1.savefig(SCRIPT_DIR / f'ArrayExpress_LLM_model_performance_{timestamp}.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')

    cm_figs = create_confusion_matrix_plots_by_disease(data)
    for key, fig in cm_figs.items():
        fig.savefig(SCRIPT_DIR / f'ArrayExpress_LLM_confusion_matrices_{key}_{timestamp}.png', dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

    df_summary = create_summary_table(data)
    csv_filename = f'ArrayExpress_LLM_performance_summary_table_{timestamp}.csv'
    df_summary.to_csv(SCRIPT_DIR / csv_filename, index=False)
    print('\n--- Summary table (CSV) ---\n')
    df_summary.to_csv(sys.stdout, index=False)

    heatmap_figs = create_heatmap(data)
    for i, fig in enumerate(heatmap_figs, 1):
        fig.savefig(SCRIPT_DIR / f'ArrayExpress_LLM_performance_heatmap_part{i}_{timestamp}.png', dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

    plt.show()

    print(f"\nPlots saved with timestamp {timestamp}:")
    print(f"1. ArrayExpress_LLM_model_performance_{timestamp}.png/pdf")
    for key in cm_figs.keys():
        print(f"2. ArrayExpress_LLM_confusion_matrices_{key}_{timestamp}.png")
    print(f"3. {csv_filename}")
    for i in range(1, len(heatmap_figs) + 1):
        print(f"4. ArrayExpress_LLM_performance_heatmap_part{i}_{timestamp}.png")

    print_insights(data, results_data['metadata'])


if __name__ == "__main__":
    main()
