import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
from matplotlib.patches import Rectangle
from datetime import datetime
import sys

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

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
    """Convert JSON results to structured format with trial-level metrics."""
    if not results_data:
        return None

    data = {}
    for result in results_data['results']:
        model = result.get('model', 'Unknown')
        version = result.get('version', 'v1')
        metrics = result.get('metrics', {})

        model_data = data.setdefault(model, {})
        version_entry = model_data.setdefault(version, {"trials": []})

        existing_trials = [trial['trial'] for trial in version_entry['trials']]
        trial_label = result.get('trial')
        if not trial_label:
            trial_label = f"trial-{len(existing_trials) + 1}"
        elif trial_label in existing_trials:
            # Ensure uniqueness if duplicated entries slip through.
            trial_label = f"{trial_label}-{len(existing_trials) + 1}"

        version_entry['trials'].append({
            "trial": trial_label,
            "metrics": metrics,
            "file_path": result.get('file_path')
        })

    for model_data in data.values():
        for version, payload in model_data.items():
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


def should_include_range(model: str, version_data: dict) -> bool:
    """Return True when the visualization should display min-max bands."""
    return model in TARGET_MULTI_TRIAL_MODELS and version_data.get("trial_count", 0) > 1


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

    # Uncomment the block below to switch to mean ± 95% CI reporting instead of median/range.
    # mean_val = summary.get("mean", {}).get(metric)
    # if mean_val is not None:
    #     ci_low, ci_high = summary.get("ci95", {}).get(metric, (mean_val, mean_val))
    #     mean_fmt = _fmt(mean_val, decimals)
    #     ci_low_fmt = _fmt(ci_low, decimals)
    #     ci_high_fmt = _fmt(ci_high, decimals)
    #     value_label = f"{mean_fmt} ({ci_low_fmt}-{ci_high_fmt})"

    return value_label

def create_publication_plot(data, metadata):
    """Create main figure with 6 subplots (A-F)"""
    # Create figure with subplots (3 rows x 2 cols)
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle(f'Performance of LLMs for ProteomeXchange Dataset Screening Across Prompt Versions\n',
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Colors for each model
    model_colors = {
        'GPT-4o-mini': '#1f77b4', 
        'o4-mini': '#ff7f0e', 
        'GPT-4.1-mini': '#2ca02c', 
        'GPT-4.1-nano': '#d62728',
        'GPT-5-mini': '#9467bd'
    }
    
    # Get available versions
    all_versions = set()
    for model_data in data.values():
        all_versions.update(model_data.keys())
    versions = sorted(list(all_versions))
    version_positions = {version: idx for idx, version in enumerate(versions)}
    xtick_positions = list(version_positions.values())

    def plot_metric(ax, metric, marker, title, ylabel, ylim=(0, 1.0)):
        for model, model_data in data.items():
            plotted_versions = []
            x_vals = []
            y_vals = []
            range_bounds = []

            for version in versions:
                if version not in model_data:
                    continue
                median, (lower, upper) = get_metric_summary(model_data[version], metric)
                if median is None:
                    continue
                plotted_versions.append(version)
                x_vals.append(version_positions[version])
                y_vals.append(median)
                range_bounds.append((lower, upper))

            if not x_vals:
                continue

            color = model_colors.get(model, 'gray')
            ax.plot(x_vals, y_vals, marker=marker, linewidth=2.5, markersize=8,
                    label=model, color=color)

            for xv, (lower, upper), version in zip(x_vals, range_bounds, plotted_versions):
                if lower is None or upper is None:
                    continue
                if should_include_range(model, model_data[version]):
                    ax.vlines(xv, lower, upper, color=color, alpha=0.35, linewidth=6)
                    ax.scatter([xv, xv], [lower, upper], color=color, alpha=1, s=50, marker='_')

        ax.set_title(title, fontsize=20, fontweight='bold')
        ax.set_xlabel('Prompt Version', fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(versions, fontsize=14)
        if ylim:
            ax.set_ylim(*ylim)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='best', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Plot A: Sensitivity across prompts
    ax1 = axes[0, 0]
    plot_metric(ax1, 'sensitivity', '.', 'A. Sensitivity (Recall)', 'Sensitivity', ylim=(0, 1.0))
    
    # Plot B: Specificity across prompts
    ax2 = axes[0, 1]
    plot_metric(ax2, 'specificity', '.', 'B. Specificity', 'Specificity', ylim=(0, 1.0))
    
    # Plot C: Precision across prompts
    ax3 = axes[1, 0]
    plot_metric(ax3, 'precision', '.', 'C. Precision', 'Precision', ylim=(0, 1.0))
    
    # Plot D: Accuracy across prompts
    ax4 = axes[1, 1]
    plot_metric(ax4, 'accuracy', '.', 'D. Accuracy', 'Accuracy', ylim=(0, 1.0))
    
    # Plot E: F1 Score across prompts
    ax5 = axes[2, 0]
    plot_metric(ax5, 'f1_score', '.', 'E. F1 Score', 'F1 Score', ylim=(0, 1.0))

    # Plot F: ROC-like plot (Sensitivity vs Specificity)
    ax6 = axes[2, 1]
    for model, model_data in data.items():
        sens_vals = []
        spec_vals = []
        version_labels = []

        for v in versions:
            if v not in model_data:
                continue
            sens_median, (sens_low, sens_high) = get_metric_summary(model_data[v], 'sensitivity')
            spec_median, (spec_low, spec_high) = get_metric_summary(model_data[v], 'specificity')
            if sens_median is None or spec_median is None:
                continue
            sens_vals.append((sens_median, sens_low, sens_high))
            spec_vals.append((spec_median, spec_low, spec_high))
            version_labels.append(v)

        if sens_vals:  # Only plot if we have data
            fpr_vals = []
            sens_med_vals = []
            fpr_range = []
            sens_range = []

            for (spec_median, spec_low, spec_high), (sens_median, sens_low, sens_high) in zip(spec_vals, sens_vals):
                fpr = 1 - spec_median
                fpr_vals.append(fpr)
                sens_med_vals.append(sens_median)

                if spec_low is None or spec_high is None:
                    fpr_range.append((fpr, fpr))
                else:
                    fpr_range.append((1 - spec_high, 1 - spec_low))
                if sens_low is None or sens_high is None:
                    sens_range.append((sens_median, sens_median))
                else:
                    sens_range.append((sens_low, sens_high))

            color = model_colors.get(model, 'gray')

            # Separate points: v3 filled, others hollow
            mask_v3 = [v == 'v3' for v in version_labels]
            x_v3 = [x for x, m in zip(fpr_vals, mask_v3) if m]
            y_v3 = [y for y, m in zip(sens_med_vals, mask_v3) if m]
            x_other = [x for x, m in zip(fpr_vals, mask_v3) if not m]
            y_other = [y for y, m in zip(sens_med_vals, mask_v3) if not m]

            if x_other:
                ax6.scatter(x_other, y_other, s=100, label=model,
                            facecolors='none', edgecolors=color, linewidths=2.0)
            if x_v3:
                ax6.scatter(x_v3, y_v3, s=100,
                            facecolors=color, edgecolors=color, linewidths=1.5,
                            label=model if not x_other else None)

            ax6.plot(fpr_vals, sens_med_vals, '--', alpha=0.5, color=color)

            for fpr, sens, version, (fpr_low, fpr_high), (sens_low, sens_high) in zip(
                fpr_vals, sens_med_vals, version_labels, fpr_range, sens_range
            ):
                #ax6.annotate(version, (fpr, sens), xytext=(5, 5), textcoords='offset points',
                             #fontsize=12, alpha=0.8)
                if should_include_range(model, model_data[version]):
                    if fpr_low is not None and fpr_high is not None:
                        ax6.hlines(sens, fpr_low, fpr_high, colors=color, alpha=0.35, linewidth=4)
                    if sens_low is not None and sens_high is not None:
                        ax6.vlines(fpr, sens_low, sens_high, colors=color, alpha=0.35, linewidth=4)

    ax6.set_title('F. ROC Space (Sensitivity vs 1-Specificity)', fontsize=20, fontweight='bold')
    ax6.set_xlabel('1 - Specificity (False Positive Rate)', fontsize=16)
    ax6.set_ylabel('Sensitivity (True Positive Rate)', fontsize=16)
    ax6.set_xlim(0, 1.0)
    ax6.set_ylim(0, 1.0)
    handles, labels = ax6.get_legend_handles_labels()
    if handles:
        ax6.legend(loc='lower right', fontsize=14)
    ax6.grid(True, alpha=0.3)
    ax6.tick_params(axis='both', which='major', labelsize=14)

    # Add diagonal line for reference
    ax6.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)

    plt.tight_layout()
    return fig

def create_confusion_matrix_plots_by_version(data):
    """Create confusion matrix visualizations for each version"""
    
    # Get available versions
    all_versions = set()
    for model_data in data.values():
        all_versions.update(model_data.keys())
    versions = sorted(list(all_versions))
    
    figures = {}
    
    for version in versions:
        models_in_version = [model for model, model_data in data.items() if version in model_data]
        if not models_in_version:
            continue

        n_models = len(models_in_version)
        
        # Determine subplot layout
        if n_models <= 2:
            rows, cols = 1, n_models
            figsize = (16, 5)
        else:
            rows, cols = 2, (n_models + 1) // 2
            figsize = (16, 10)
        
        if n_models == 0: continue

        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        fig.suptitle(f'Confusion Matrices by Model for ProteomeXchange Dataset Screening', 
                     fontsize=24, fontweight='bold')
        
        for idx, model in enumerate(models_in_version):
            version_data = data[model][version]
            trial_count = version_data.get("trial_count", 1)
            include_range = trial_count > 1

            # Get median values for display
            median_summary = version_data.get("summary", {}).get("median", {})
            tn_med = median_summary.get('true_negative')
            fp_med = median_summary.get('false_positive')
            fn_med = median_summary.get('false_negative')
            tp_med = median_summary.get('true_positive')

            if None in {tn_med, fp_med, fn_med, tp_med}:
                axes[idx].set_title(f'{model} ({version})\nData Missing', fontsize=16)
                axes[idx].set_visible(False)
                continue

            cm_display = np.round([[tn_med, fp_med], [fn_med, tp_med]]).astype(int)
            
            # Prepare annotations with ranges if applicable
            annot_labels = np.full_like(cm_display, "", dtype=object)
            range_summary = version_data.get("summary", {}).get("range", {})

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

            axes[idx].set_title(f'{model} ({version}, n={trial_count})',
                                fontsize=16, fontweight='bold')
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        figures[version] = fig
        
    return figures

def create_summary_table(data):
    """Create a comprehensive summary table and return as a pandas DataFrame.

    The table is returned so callers can save or print it. This function intentionally
    avoids any interactive plotting libraries and focuses on tabular output.
    """
    # Prepare data for table
    rows = []
    for model in sorted(data.keys()):
        for version in sorted(data[model].keys()):
            version_data = data[model][version]
            summary = version_data.get("summary", {})
            trials = version_data.get("trial_count", 1)
            include_range = should_include_range(model, version_data)

            row = [
                model,
                version,
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

    col_labels = ['Model', 'Version', 'Trials', 'Sensitivity', 'Specificity',
                  'Precision', 'Accuracy', 'F1 Score', 'TP', 'FP', 'TN', 'FN']

    if not rows:
        # Return an empty DataFrame with the expected columns
        return pd.DataFrame(columns=col_labels)

    df = pd.DataFrame(rows, columns=col_labels)
    return df

def create_heatmap(data):
    """Create heatmap showing performance across models and versions - split into 3 files"""
    # Prepare data for heatmaps
    models = list(data.keys())
    all_versions = set()
    for model_data in data.values():
        all_versions.update(model_data.keys())
    versions = sorted(list(all_versions))
    
    metrics = ['sensitivity', 'specificity', 'precision', 'accuracy', 'f1_score']
    titles = ['Sensitivity', 'Specificity', 'Precision', 'Accuracy', 'F1 Score']
    
    figures = []
    
    # Create 3 figures with 2 heatmaps each (except last one might have 1)
    for fig_idx in range(3):
        start_idx = fig_idx * 2
        end_idx = min(start_idx + 2, len(metrics))
        
        if start_idx >= len(metrics):
            break
            
        # Determine subplot layout
        n_plots = end_idx - start_idx
        if n_plots == 2:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        else:
            fig, axes = plt.subplots(1, 1, figsize=(8, 6))
            axes = [axes]
        
        for plot_idx, metric_idx in enumerate(range(start_idx, end_idx)):
            metric = metrics[metric_idx]
            title = titles[metric_idx]
            
            # Create matrix for heatmap
            matrix = np.full((len(models), len(versions)), np.nan)
            for i, model in enumerate(models):
                for j, version in enumerate(versions):
                    if version in data[model]:
                        summary = data[model][version].get("summary", {})
                        value = summary.get("median", {}).get(metric)
                        if value is not None:
                            matrix[i, j] = value
            
            # Create heatmap with larger fonts
            mask = np.isnan(matrix)
            sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
                       xticklabels=versions, yticklabels=models,
                       ax=axes[plot_idx], cbar_kws={'label': title},
                       mask=mask, annot_kws={'size': 12})
            
            axes[plot_idx].set_title(f'{title}', fontsize=20, fontweight='bold')
            axes[plot_idx].set_xlabel('Prompt Version', fontsize=14)
            axes[plot_idx].set_ylabel('Model', fontsize=14)
            
            # Increase tick label sizes
            axes[plot_idx].tick_params(axis='both', which='major', labelsize=12)
            
            # Increase colorbar label size
            cbar = axes[plot_idx].collections[0].colorbar
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label(title, fontsize=14)
        
        # Only add title to the first figure
        if fig_idx == 0:
            plt.suptitle('LLM Performance Heatmap Across Prompt Versions for ProteomeXchange Dataset Screening', 
                         fontsize=20, fontweight='bold')
        
        plt.tight_layout()
        figures.append(fig)
    
    return figures

def print_insights(data, metadata):
    """Print key insights from the analysis"""
    print("\n" + "="*60)
    print("KEY INSIGHTS:")
    print("="*60)
    
    # Best performing model-version combinations
    all_combinations = []
    for model, versions in data.items():
        for version, version_data in versions.items():
            medians = version_data.get("summary", {}).get("median", {})
            if not medians:
                continue
            all_combinations.append((model, version, medians))
    
    if all_combinations:
        best_sensitivity = max(
            all_combinations, key=lambda x: x[2].get('sensitivity', float('-inf'))
        )
        best_specificity = max(
            all_combinations, key=lambda x: x[2].get('specificity', float('-inf'))
        )
        best_f1 = max(
            all_combinations, key=lambda x: x[2].get('f1_score', float('-inf'))
        )
        best_accuracy = max(
            all_combinations, key=lambda x: x[2].get('accuracy', float('-inf'))
        )

        print(f"Best Sensitivity: {best_sensitivity[0]} {best_sensitivity[1]} ({best_sensitivity[2].get('sensitivity', 0):.3f})")
        print(f"Best Specificity: {best_specificity[0]} {best_specificity[1]} ({best_specificity[2].get('specificity', 0):.3f})")
        print(f"Best F1 Score: {best_f1[0]} {best_f1[1]} ({best_f1[2].get('f1_score', 0):.3f})")
        print(f"Best Accuracy: {best_accuracy[0]} {best_accuracy[1]} ({best_accuracy[2].get('accuracy', 0):.3f})")
    else:
        print("No metrics available to compute insights.")
    
    # Version progression analysis
    print("\nVERSION PROGRESSION ANALYSIS:")
    for model, versions_data in data.items():
        versions = sorted(versions_data.keys())
        if len(versions) > 1:
            print(f"\n{model}:")
            for i in range(1, len(versions)):
                prev_v = versions[i-1]
                curr_v = versions[i]
                
                curr_summary = versions_data[curr_v].get("summary", {}).get("median", {})
                prev_summary = versions_data[prev_v].get("summary", {}).get("median", {})

                f1_change = curr_summary.get('f1_score', 0) - prev_summary.get('f1_score', 0)
                sens_change = curr_summary.get('sensitivity', 0) - prev_summary.get('sensitivity', 0)
                spec_change = curr_summary.get('specificity', 0) - prev_summary.get('specificity', 0)
                
                print(f"  {prev_v} → {curr_v}: F1 {f1_change:+.3f}, Sens {sens_change:+.3f}, Spec {spec_change:+.3f}")

def main():
    """Main function to run all visualizations"""
    # Load results
    results_data = load_results()
    if not results_data:
        return
    
    # Process data
    data = process_data(results_data)
    if not data:
        print("❌ No data to plot")
        return
    
    print(f"📊 Found data for models: {list(data.keys())}")
    
    # Create timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create and save all plots
    print("🎨 Creating visualizations...")
    
    # 1. Main performance plot
    fig1 = create_publication_plot(data, results_data['metadata'])
    fig1.savefig(f'ProteomeXchange_LLM_model_performance_{timestamp}.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    fig1.savefig(f'ProteomeXchange_LLM_model_performance_{timestamp}.pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    # 2. Confusion matrices by version
    cm_figs = create_confusion_matrix_plots_by_version(data)
    for version, fig in cm_figs.items():
        fig.savefig(f'ProteomeXchange_LLM_confusion_matrices_{version}_{timestamp}.png', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    
    # 3. Summary table (CSV printed to terminal)
    df_summary = create_summary_table(data)
    csv_filename = f'ProteomeXchange_LLM_performance_summary_table_{timestamp}.csv'
    df_summary.to_csv(csv_filename, index=False)
    # Print CSV to terminal
    print('\n--- Summary table (CSV) ---\n')
    df_summary.to_csv(sys.stdout, index=False)
    
    # 4. Heatmaps (multiple figures)
    heatmap_figs = create_heatmap(data)
    for i, fig in enumerate(heatmap_figs, 1):
        fig.savefig(f'ProteomeXchange_LLM_performance_heatmap_part{i}_{timestamp}.png', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    
    plt.show()
    
    print(f"\n📈 Plots saved with timestamp {timestamp}:")
    print(f"1. ProteomeXchange_LLM_model_performance_{timestamp}.png/pdf - Main figure with 4 subplots")
    for version in cm_figs.keys():
        print(f"2. ProteomeXchange_LLM_confusion_matrices_{version}_{timestamp}.png - Confusion matrices for {version}")
    print(f"3. ProteomeXchange_LLM_performance_summary_table_{timestamp}.csv - Comprehensive summary table (CSV)")
    print(f"4. ProteomeXchange_LLM_performance_heatmap_part1_{timestamp}.png - Performance heatmap (Sensitivity & Specificity)")
    print(f"5. ProteomeXchange_LLM_performance_heatmap_part2_{timestamp}.png - Performance heatmap (Precision & Accuracy)")
    print(f"6. ProteomeXchange_LLM_performance_heatmap_part3_{timestamp}.png - Performance heatmap (F1 Score)")
    
    # Print insights
    print_insights(data, results_data['metadata'])

if __name__ == "__main__":
    main()
