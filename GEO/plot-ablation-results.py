import argparse
import json
from datetime import datetime
from pathlib import Path
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["axes.facecolor"] = "white"

MODEL_ORDER = ["GPT-4o-mini", "GPT-4.1-mini", "GPT-5-mini", "o4-mini"]
INPUT_CONFIG_ORDER = [
    "title_only",
    "title_description",
    "title_description_clinical",
]
INPUT_CONFIG_LABELS = {
    "title_only": "Title",
    "title_description": "Title & Description",
    "title_description_clinical": "Title, Description,\nClinical Table",
}
INPUT_CONFIG_MARKERS = {
    "title_only": "o",
    "title_description": "^",
    "title_description_clinical": "s",
}
MODEL_COLORS = {
    "GPT-4o-mini": "#CC79A7",
    "GPT-4.1-mini": "#009E73",
    "GPT-5-mini": "#0072B2",
    "o4-mini": "#E69F00",
}
RATE_METRICS = ["sensitivity", "specificity", "precision", "accuracy", "f1_score"]
COUNT_METRICS = ["true_positive", "false_positive", "true_negative", "false_negative"]
ALL_METRICS = RATE_METRICS + COUNT_METRICS
METRIC_DECIMALS = {
    "sensitivity": 3,
    "specificity": 3,
    "precision": 3,
    "accuracy": 3,
    "f1_score": 3,
    "true_positive": 1,
    "false_positive": 1,
    "true_negative": 1,
    "false_negative": 1,
}
METRIC_TITLES = {
    "sensitivity": "A. Sensitivity",
    "specificity": "B. Specificity",
    "precision": "C. Precision",
    "accuracy": "D. Accuracy",
    "f1_score": "E. F1 Score",
}
CONFUSION_METRIC_LAYOUT = [
    ["true_negative", "false_positive"],
    ["false_negative", "true_positive"],
]


def lighten_color(color, amount=0.45):
    """Mix a color with white. amount=0 returns the original; amount=1 returns white."""
    rgb = mcolors.to_rgb(color)
    return tuple(component + (1 - component) * amount for component in rgb)


def load_results(path):
    results_path = Path(path)
    if not results_path.exists() and not results_path.is_absolute():
        results_path = Path(__file__).resolve().parent / results_path

    with results_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if "results" not in data:
        raise ValueError(f"{results_path} does not contain a top-level 'results' list.")

    return data


def ordered_values(values, preferred_order):
    ordered = [value for value in preferred_order if value in values]
    extras = sorted(value for value in values if value not in preferred_order)
    return ordered + extras


def summarize_metric(values):
    series = pd.Series(values, dtype=float).dropna()
    if series.empty:
        return {"median": np.nan, "min": np.nan, "max": np.nan, "mean": np.nan}
    return {
        "median": float(series.median()),
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": float(series.mean()),
    }


def process_results(results_data):
    grouped = {}

    for result in results_data.get("results", []):
        model = result.get("model", "Unknown")
        input_config = result.get("input_config", "unknown")
        metrics = result.get("metrics", {})
        if not metrics:
            continue

        entry = grouped.setdefault((model, input_config), {"trials": []})
        entry["trials"].append(
            {
                "trial": result.get("trial", f"trial-{len(entry['trials']) + 1}"),
                "metrics": metrics,
                "file_path": result.get("file_path"),
            }
        )

    for entry in grouped.values():
        metric_rows = [trial["metrics"] for trial in entry["trials"]]
        metric_names = sorted({metric for row in metric_rows for metric in row})
        summary = {}
        for metric in metric_names:
            summary[metric] = summarize_metric(
                row.get(metric) for row in metric_rows if metric in row
            )

        entry["summary"] = summary
        entry["trial_count"] = len(entry["trials"])

    return grouped


def get_summary(grouped, model, input_config, metric):
    entry = grouped.get((model, input_config))
    if not entry:
        return None
    return entry.get("summary", {}).get(metric)


def yerr_from_summary(summary):
    median = summary["median"]
    lower = summary["min"]
    upper = summary["max"]
    if np.isnan(median) or np.isnan(lower) or np.isnan(upper) or lower == upper:
        return None
    return [[median - lower], [upper - median]]


def format_metric_value(summary, include_range, decimals=3):
    """Format a metric using its median and optional min-max range."""
    if not summary:
        return "-"

    median = summary.get("median")
    if median is None or pd.isna(median):
        return "-"

    lower = summary.get("min", median)
    upper = summary.get("max", median)

    def _fmt(value, precision):
        if value is None or pd.isna(value):
            return "-"
        if abs(value - round(value)) < 10 ** (-(precision + 1)):
            return f"{int(round(value))}"
        return f"{value:.{precision}f}"

    median_fmt = _fmt(median, decimals)
    lower_fmt = _fmt(lower, decimals)
    upper_fmt = _fmt(upper, decimals)

    if include_range:
        return f"{median_fmt}\n({lower_fmt}-{upper_fmt})"
    return median_fmt


def create_metric_panel(grouped, metadata):
    models = ordered_values({model for model, _ in grouped}, MODEL_ORDER)
    input_configs = ordered_values(
        {input_config for _, input_config in grouped}, INPUT_CONFIG_ORDER
    )

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle(
        "Input Configuration Ablation on LLM Performance for GEO Screening\n",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )
    axes = axes.flatten()

    group_width = 0.82
    bar_width = group_width / max(len(models), 1)
    x_positions = np.arange(len(input_configs))
    model_handles = [
        Patch(
            facecolor=MODEL_COLORS.get(model, "gray"),
            edgecolor="black",
            linewidth=0.6,
            label=model,
        )
        for model in models
    ]

    for ax, metric in zip(axes, RATE_METRICS):
        for config_index, input_config in enumerate(input_configs):
            for model_index, model in enumerate(models):
                summary = get_summary(grouped, model, input_config, metric)
                if not summary or np.isnan(summary["median"]):
                    continue

                bar_x = config_index + (model_index - (len(models) - 1) / 2) * bar_width
                base_color = MODEL_COLORS.get(model, "gray")
                color = (
                    base_color
                    if input_config == "title_description_clinical"
                    else lighten_color(base_color, 0.35)
                )
                ax.bar(
                    bar_x,
                    summary["median"],
                    width=bar_width,
                    color=color,
                    edgecolor="black",
                    linewidth=0.6,
                    zorder=2,
                )

                entry = grouped.get((model, input_config), {})
                if entry.get("trial_count", 0) > 1:
                    yerr = yerr_from_summary(summary)
                    if yerr:
                        ax.errorbar(
                            bar_x,
                            summary["median"],
                            yerr=yerr,
                            fmt="none",
                            color="black",
                            capsize=3,
                            linewidth=1.2,
                            zorder=3,
                        )

        ax.set_title(METRIC_TITLES[metric], fontsize=18, fontweight="bold", pad=32)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([INPUT_CONFIG_LABELS.get(config, config) for config in input_configs])
        ax.set_xlabel("Input Configuration", fontsize=13)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=13)
        ax.set_ylim(0, 1.04)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", which="major", labelsize=11)
        ax.legend(
            handles=model_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=max(len(model_handles), 1),
            frameon=False,
            fontsize=11,
            handlelength=1.4,
            columnspacing=1.2,
            borderaxespad=0.2,
        )

    roc_ax = axes[-1]
    for model in models:
        color = MODEL_COLORS.get(model, "gray")
        points = []
        for input_config in input_configs:
            sensitivity = get_summary(grouped, model, input_config, "sensitivity")
            specificity = get_summary(grouped, model, input_config, "specificity")
            if not sensitivity or not specificity:
                continue
            if np.isnan(sensitivity["median"]) or np.isnan(specificity["median"]):
                continue
            points.append(
                {
                    "config": input_config,
                    "fpr": 1 - specificity["median"],
                    "sensitivity": sensitivity["median"],
                    "fpr_low": 1 - specificity["max"],
                    "fpr_high": 1 - specificity["min"],
                    "sensitivity_low": sensitivity["min"],
                    "sensitivity_high": sensitivity["max"],
                    "trial_count": grouped[(model, input_config)]["trial_count"],
                }
            )

        if not points:
            continue

        roc_ax.plot(
            [point["fpr"] for point in points],
            [point["sensitivity"] for point in points],
            "-",
            color=color,
            alpha=0.45,
            linewidth=1.5,
        )

        added_legend = False
        for point in points:
            marker = INPUT_CONFIG_MARKERS.get(point["config"], "o")
            facecolor = color if marker == "s" else "white"
            roc_ax.scatter(
                [point["fpr"]],
                [point["sensitivity"]],
                s=110,
                marker=marker,
                facecolors=facecolor,
                edgecolors=color,
                linewidths=2.0,
                label=model if not added_legend else None,
                zorder=3,
            )
            added_legend = True
            if point["trial_count"] > 1:
                if point["fpr_low"] != point["fpr_high"]:
                    roc_ax.hlines(
                        point["sensitivity"],
                        point["fpr_low"],
                        point["fpr_high"],
                        colors=color,
                        alpha=0.35,
                        linewidth=4,
                    )
                if point["sensitivity_low"] != point["sensitivity_high"]:
                    roc_ax.vlines(
                        point["fpr"],
                        point["sensitivity_low"],
                        point["sensitivity_high"],
                        colors=color,
                        alpha=0.35,
                        linewidth=4,
                    )

    roc_ax.set_title("F. ROC Space", fontsize=18, fontweight="bold", pad=32)
    roc_ax.set_xlabel("1 - Specificity", fontsize=13)
    roc_ax.set_ylabel("Sensitivity", fontsize=13)
    roc_ax.set_xlim(0, 1.04)
    roc_ax.set_ylim(0, 1.04)
    roc_ax.set_xticks(np.linspace(0, 1, 6))
    roc_ax.set_yticks(np.linspace(0, 1, 6))
    roc_ax.grid(True, alpha=0.3)
    roc_ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    handles, labels = roc_ax.get_legend_handles_labels()
    if handles:
        model_legend = roc_ax.legend(
            handles=handles,
            labels=labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(handles),
            frameon=False,
            fontsize=11,
            handlelength=1.4,
            columnspacing=1.2,
            borderaxespad=0.2,
        )
        roc_ax.add_artist(model_legend)

    marker_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=INPUT_CONFIG_MARKERS.get(input_config, "o"),
            color="black",
            markerfacecolor="black"
            if input_config == "title_description_clinical"
            else "white",
            markeredgecolor="black",
            linewidth=0,
            markersize=8,
            label=INPUT_CONFIG_LABELS.get(input_config, input_config).replace("\n", " "),
        )
        for input_config in input_configs
    ]
    roc_ax.legend(
        handles=marker_handles,
        loc="lower right",
        ncol=1,
        frameon=False,
        fontsize=10,
        handlelength=1.2,
        labelspacing=0.5,
        borderaxespad=0.6,
    )

    plt.tight_layout()
    return fig


def create_delta_heatmap(grouped):
    models = ordered_values({model for model, _ in grouped}, MODEL_ORDER)
    ablation_configs = [
        config for config in INPUT_CONFIG_ORDER if config != "title_description_clinical"
    ]
    metrics = ["sensitivity", "specificity", "precision", "accuracy", "f1_score"]

    fig, axes = plt.subplots(1, len(ablation_configs), figsize=(8 * len(ablation_configs), 6))
    if len(ablation_configs) == 1:
        axes = [axes]

    for ax, input_config in zip(axes, ablation_configs):
        matrix = np.full((len(models), len(metrics)), np.nan)
        annotations = np.full((len(models), len(metrics)), "", dtype=object)

        for model_index, model in enumerate(models):
            for metric_index, metric in enumerate(metrics):
                current = get_summary(grouped, model, input_config, metric)
                baseline = get_summary(
                    grouped, model, "title_description_clinical", metric
                )
                if not current or not baseline:
                    continue
                delta = current["median"] - baseline["median"]
                matrix[model_index, metric_index] = delta
                annotations[model_index, metric_index] = f"{delta:+.3f}"

        sns.heatmap(
            matrix,
            annot=annotations,
            fmt="",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            xticklabels=[metric.replace("_", "\n") for metric in metrics],
            yticklabels=models,
            mask=np.isnan(matrix),
            ax=ax,
            cbar_kws={"label": "Delta vs full clinical input"},
            annot_kws={"size": 11},
        )
        ax.set_title(
            f"{INPUT_CONFIG_LABELS.get(input_config, input_config)} vs full input",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_xlabel("Metric", fontsize=12)
        ax.set_ylabel("Model", fontsize=12)

    fig.suptitle(
        "Ablation Performance Delta from Full Clinical Input",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    return fig


def create_confusion_matrix_panel(grouped):
    models = ordered_values({model for model, _ in grouped}, MODEL_ORDER)
    input_configs = ordered_values(
        {input_config for _, input_config in grouped}, INPUT_CONFIG_ORDER
    )

    rows = len(models)
    cols = len(input_configs)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(5.2 * cols, 4.4 * rows),
        squeeze=False,
    )
    fig.suptitle(
        "GEO Ablation Confusion Matrices",
        fontsize=20,
        fontweight="bold",
        y=0.995,
    )

    for row_index, model in enumerate(models):
        for col_index, input_config in enumerate(input_configs):
            ax = axes[row_index][col_index]
            entry = grouped.get((model, input_config))
            if not entry:
                ax.set_visible(False)
                continue

            median_values = []
            annotation = np.full((2, 2), "", dtype=object)
            for i in range(2):
                row_values = []
                for j in range(2):
                    metric = CONFUSION_METRIC_LAYOUT[i][j]
                    summary = entry["summary"].get(metric)
                    if not summary:
                        row_values.append(np.nan)
                        continue
                    value = int(round(summary["median"]))
                    row_values.append(value)
                    if entry["trial_count"] > 1 and summary["min"] != summary["max"]:
                        annotation[i, j] = (
                            f"{value}\n({int(summary['min'])}-{int(summary['max'])})"
                        )
                    else:
                        annotation[i, j] = str(value)
                median_values.append(row_values)

            matrix = np.array(median_values, dtype=float)
            sns.heatmap(
                matrix,
                annot=annotation,
                fmt="",
                cmap="Blues",
                cbar=False,
                xticklabels=["Predicted Negative", "Predicted Positive"],
                yticklabels=["Actual Negative", "Actual Positive"],
                ax=ax,
                annot_kws={"size": 24, "va": "center"},
            )
            title = INPUT_CONFIG_LABELS.get(input_config, input_config).replace("\n", " ")
            ax.set_title(f"{title}\n(trials={entry['trial_count']})", fontsize=12)
            ax.tick_params(axis="both", which="major", labelsize=10)
            if col_index == 0:
                ax.set_ylabel(model, fontsize=13, fontweight="bold")
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def create_summary_table(grouped):
    rows = []
    for model, input_config in sorted(
        grouped,
        key=lambda key: (
            MODEL_ORDER.index(key[0]) if key[0] in MODEL_ORDER else len(MODEL_ORDER),
            INPUT_CONFIG_ORDER.index(key[1])
            if key[1] in INPUT_CONFIG_ORDER
            else len(INPUT_CONFIG_ORDER),
            key[0],
            key[1],
        ),
    ):
        entry = grouped[(model, input_config)]
        include_range = entry.get("trial_count", 1) > 1
        row = [
            model,
            input_config,
            entry["trial_count"],
            format_metric_value(
                entry["summary"].get("sensitivity"),
                include_range,
                METRIC_DECIMALS["sensitivity"],
            ),
            format_metric_value(
                entry["summary"].get("specificity"),
                include_range,
                METRIC_DECIMALS["specificity"],
            ),
            format_metric_value(
                entry["summary"].get("precision"),
                include_range,
                METRIC_DECIMALS["precision"],
            ),
            format_metric_value(
                entry["summary"].get("accuracy"),
                include_range,
                METRIC_DECIMALS["accuracy"],
            ),
            format_metric_value(
                entry["summary"].get("f1_score"),
                include_range,
                METRIC_DECIMALS["f1_score"],
            ),
            format_metric_value(
                entry["summary"].get("true_positive"),
                include_range,
                METRIC_DECIMALS["true_positive"],
            ),
            format_metric_value(
                entry["summary"].get("false_positive"),
                include_range,
                METRIC_DECIMALS["false_positive"],
            ),
            format_metric_value(
                entry["summary"].get("true_negative"),
                include_range,
                METRIC_DECIMALS["true_negative"],
            ),
            format_metric_value(
                entry["summary"].get("false_negative"),
                include_range,
                METRIC_DECIMALS["false_negative"],
            ),
        ]
        rows.append([str(item) for item in row])

    col_labels = [
        "Model",
        "Input Configuration",
        "Trials",
        "Sensitivity",
        "Specificity",
        "Precision",
        "Accuracy",
        "F1 Score",
        "TP",
        "FP",
        "TN",
        "FN",
    ]

    if not rows:
        return pd.DataFrame(columns=col_labels)

    return pd.DataFrame(rows, columns=col_labels)


def print_insights(grouped):
    print("\nAblation summary")
    print("=" * 60)

    for input_config in INPUT_CONFIG_ORDER:
        candidates = []
        for (model, config), entry in grouped.items():
            if config != input_config:
                continue
            f1_summary = entry["summary"].get("f1_score")
            sensitivity_summary = entry["summary"].get("sensitivity")
            if f1_summary and sensitivity_summary:
                candidates.append((model, entry, f1_summary, sensitivity_summary))

        if not candidates:
            continue

        best_f1 = max(candidates, key=lambda item: item[2]["median"])
        best_sensitivity = max(candidates, key=lambda item: item[3]["median"])
        label = INPUT_CONFIG_LABELS.get(input_config, input_config).replace("\n", " ")
        print(f"\n{label}:")
        print(
            f"  Best F1: {best_f1[0]} "
            f"({best_f1[2]['median']:.3f}, trials={best_f1[1]['trial_count']})"
        )
        print(
            f"  Best sensitivity: {best_sensitivity[0]} "
            f"({best_sensitivity[3]['median']:.3f}, "
            f"trials={best_sensitivity[1]['trial_count']})"
        )

    print("\nDelta vs full clinical input:")
    for model in ordered_values({model for model, _ in grouped}, MODEL_ORDER):
        baseline = grouped.get((model, "title_description_clinical"))
        if not baseline:
            continue
        baseline_f1 = baseline["summary"].get("f1_score", {}).get("median")
        if baseline_f1 is None:
            continue
        print(f"  {model}:")
        for input_config in INPUT_CONFIG_ORDER:
            if input_config == "title_description_clinical":
                continue
            entry = grouped.get((model, input_config))
            if not entry:
                continue
            f1 = entry["summary"].get("f1_score", {}).get("median")
            sensitivity = entry["summary"].get("sensitivity", {}).get("median")
            baseline_sensitivity = baseline["summary"].get("sensitivity", {}).get(
                "median"
            )
            print(
                f"    {input_config}: F1 {f1 - baseline_f1:+.3f}, "
                f"sensitivity {sensitivity - baseline_sensitivity:+.3f}"
            )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot GEO NSCLC ablation-study results."
    )
    parser.add_argument(
        "--results-json",
        default="ablation_performance_results.json",
        help="Path to ablation_performance_results.json.",
    )
    parser.add_argument(
        "--output-prefix",
        default="GEO_ablation",
        help="Prefix for generated plot and table files.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save files without opening an interactive matplotlib window.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    results_data = load_results(args.results_json)
    grouped = process_results(results_data)

    if not grouped:
        raise SystemExit("No ablation results found to plot.")

    metadata = results_data.get("metadata", {})
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_prefix = f"{args.output_prefix}_{timestamp}"

    metric_fig = create_metric_panel(grouped, metadata)
    metric_fig.savefig(
        f"{output_prefix}_performance.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    metric_fig.savefig(
        f"{output_prefix}_performance.pdf",
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )

    delta_fig = create_delta_heatmap(grouped)
    delta_fig.savefig(
        f"{output_prefix}_delta_heatmap.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )

    confusion_fig = create_confusion_matrix_panel(grouped)
    confusion_fig.savefig(
        f"{output_prefix}_confusion_matrices.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )

    summary_df = create_summary_table(grouped)
    summary_csv = f"{output_prefix}_summary_table.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"Loaded {len(results_data.get('results', []))} result rows.")
    print(f"Saved {output_prefix}_performance.png/pdf")
    print(f"Saved {output_prefix}_delta_heatmap.png")
    print(f"Saved {output_prefix}_confusion_matrices.png")
    print(f"Saved {summary_csv}")
    print_insights(grouped)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
