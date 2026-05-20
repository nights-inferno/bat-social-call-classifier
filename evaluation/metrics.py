import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    auc,
    log_loss,
    brier_score_loss,
    balanced_accuracy_score,
    hamming_loss,
    accuracy_score,
    f1_score,
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

def summarize(values):
    values = np.asarray(values, dtype=float)

    return {
        "mean": np.mean(values),
        "std": np.std(values, ddof=1),
        "min": np.min(values),
        "max": np.max(values),
    }

def compute_metrics(y_true, y_pred_proba, 
                    label_names=None, threshold=0.5):
    """Compute comprehensive multi-label metrics for a single evaluation."""
    y_pred_binary = (y_pred_proba > threshold).astype(int)
    
    n_labels = y_true.shape[1]
    if label_names is None:
        label_names = [f"Label {i}" for i in range(n_labels)]
    
    metrics = {
        'Macro-AUC': roc_auc_score(y_true, y_pred_proba, average='macro'),
        'cmAP': average_precision_score(y_true, y_pred_proba, average='macro'),
        'Macro-Balanced Accuracy' : np.mean([balanced_accuracy_score(y_true[:, i], y_pred_binary[:, i]) for i in range(n_labels)]),
        'AP per label': {},
        'Brier per label': {},
        'Log-Loss per label': {}
    }
    
    #Per-label metrics
    for i, label_name in enumerate(label_names):
        # Average Precision (Recommended over trapezoidal PR-AUC)
        ap = average_precision_score(y_true[:, i], y_pred_proba[:, i])
        metrics['AP per label'][label_name] = ap
        
        # Error Scores
        metrics['Brier per label'][label_name] = brier_score_loss(y_true[:, i], y_pred_proba[:, i])
        metrics['Log-Loss per label'][label_name] = log_loss(y_true[:, i], y_pred_proba[:, i], labels=[0,1])
    
    # Aggregate macro scores
    metrics['Brier (macro)'] = np.mean(list(metrics['Brier per label'].values()))
    metrics['Log-Loss (macro)'] = np.mean(list(metrics['Log-Loss per label'].values()))
    
    return metrics

def compute_fold_metrics(y_true, y_pred_proba, label_names=None, threshold=0.5):
    fold_metrics = []
    for fold in range(len(y_true)):
        fold_metrics.append(compute_metrics(
            y_true[fold], 
            y_pred_proba[fold], 
            label_names, 
            threshold
        ))
    return fold_metrics


def compute_cv_stats(fold_metrics):
    """Compute metrics across cross-validation folds."""
    result = {}
    metrics = fold_metrics[0].keys()
    for metric in metrics:

        first_value = fold_metrics[0][metric]

        # scalar metric
        if np.isscalar(first_value):

            vals = [m[metric] for m in fold_metrics]
            result[metric] = summarize(vals)

        # nested dict metric
        elif isinstance(first_value, dict):

            result[metric] = {}

            labels = first_value.keys()

            for label in labels:
                vals = [m[metric][label] for m in fold_metrics]
                result[metric][label] = summarize(vals)

        else:
            raise TypeError(f"Unsupported metric type for key={metric}")
    """
    aggregated_metrics = {
        'Macro-AUC': [
            np.mean([m['Macro-AUC'] for m in fold_metrics]),
            np.max([m['Macro-AUC'] for m in fold_metrics]), 
            np.min([m['Macro-AUC'] for m in fold_metrics])
        ],
        'cmAP': [
            np.mean([m['cmAP'] for m in fold_metrics]), 
            np.max([m['cmAP'] for m in fold_metrics]), 
            np.min([m['cmAP'] for m in fold_metrics])
        ],

        'AP per label': [
            {label: np.mean([m['AP per label'][label] for m in fold_metrics]) for label in label_names},
            {label: np.max([m['AP per label'][label] for m in fold_metrics]) for label in label_names},
            {label: np.min([m['AP per label'][label] for m in fold_metrics]) for label in label_names}
        ],    
        'Brier per label mean': [
            {label: np.mean([m['Brier per label'][label] for m in fold_metrics]) for label in label_names}, 
            {label: np.max([m['Brier per label'][label] for m in fold_metrics]) for label in label_names}, 
            {label: np.min([m['Brier per label'][label] for m in fold_metrics]) for label in label_names}
        ],
        'Log-Loss per label mean': [
            {label: np.mean([m['Log-Loss per label'][label] for m in fold_metrics]) for label in label_names},
            {label: np.max([m['Log-Loss per label'][label] for m in fold_metrics]) for label in label_names}, 
            {label: np.min([m['Log-Loss per label'][label] for m in fold_metrics]) for label in label_names}
        ],

        'Brier (macro) mean': [
            np.mean([m['Brier (macro)'] for m in fold_metrics]), 
            np.max([m['Brier (macro)'] for m in fold_metrics]), 
            np.min([m['Brier (macro)'] for m in fold_metrics])
        ],
        'Log-Loss (macro) mean': [
            np.mean([m['Log-Loss (macro)'] for m in fold_metrics]), 
            np.max([m['Log-Loss (macro)'] for m in fold_metrics]), 
            np.min([m['Log-Loss (macro)'] for m in fold_metrics])
        ]
    }
    """
    return result

def result_summary(y_true, y_pred_proba, label_names=None, threshold=0.5) :
    oof_true = np.concatenate(y_true, axis=0)
    oof_pred_proba = np.concatenate(y_pred_proba, axis=0)
    fold_metrics = compute_fold_metrics(y_true, y_pred_proba, label_names, threshold)

    results = {
        "oof" :{
            "metrics" : compute_metrics(oof_true, oof_pred_proba, label_names, threshold),
            "true": oof_true,
            "pred_proba": oof_pred_proba
        },
        "cv" : {
            "stats" : compute_cv_stats(fold_metrics),
            "folds": fold_metrics
        }
    }
    return results


def generate_metrics_table(all_results,label_names=None):
    global_rows = []
    class_rows = []
    for model, stats in all_results.items():
        global_row = {
            "Model": model,
            "Macro-AUC": f"{stats['Macro-AUC'][0]:.3f} ± {max(stats['Macro-AUC'][1]-stats['Macro-AUC'][0], stats['Macro-AUC'][0]-stats['Macro-AUC'][2]) :.3f}",
            "cmAP": f"{stats['cmAP'][0]:.3f} ± {max(stats['cmAP'][1]-stats['cmAP'][0], stats['cmAP'][0]-stats['cmAP'][2]):.3f}",
            "Brier Score": f"{stats['Brier (macro) mean'][0]:.4f} ± {max(stats['Brier (macro) mean'][1]-stats['Brier (macro) mean'][0], stats['Brier (macro) mean'][0]-stats['Brier (macro) mean'][2]):.3f}",
            "Log-Loss": f"{stats['Log-Loss (macro) mean'][0]:.3f} ± {max(stats['Log-Loss (macro) mean'][1]-stats['Log-Loss (macro) mean'][0], stats['Log-Loss (macro) mean'][0]-stats['Log-Loss (macro) mean'][2]):.3f}"
        }
        global_rows.append(global_row)

        class_row = {
            "Model": model,
            "AP type A": f"{stats['AP per label'][0][label_names[0]]:.3f} ± {max(stats['AP per label'][1][label_names[0]]-stats['AP per label'][0][label_names[0]], stats['AP per label'][0][label_names[0]]-stats['AP per label'][2][label_names[0]]):.3f}",
            "AP type B": f"{stats['AP per label'][0][label_names[1]]:.3f} ± {max(stats['AP per label'][1][label_names[1]]-stats['AP per label'][0][label_names[1]], stats['AP per label'][0][label_names[1]]-stats['AP per label'][2][label_names[1]]):.3f}",
            "AP type C": f"{stats['AP per label'][0][label_names[2]]:.3f} ± {max(stats['AP per label'][1][label_names[2]]-stats['AP per label'][0][label_names[2]], stats['AP per label'][0][label_names[2]]-stats['AP per label'][2][label_names[2]]):.3f}",
            "AP type D": f"{stats['AP per label'][0][label_names[3]]:.3f} ± {max(stats['AP per label'][1][label_names[3]]-stats['AP per label'][0][label_names[3]], stats['AP per label'][0][label_names[3]]-stats['AP per label'][2][label_names[3]]):.3f}",
            "AP Echo": f"{stats['AP per label'][0][label_names[4]]:.3f} ± {max(stats['AP per label'][1][label_names[4]]-stats['AP per label'][0][label_names[4]], stats['AP per label'][0][label_names[4]]-stats['AP per label'][2][label_names[4]]):.3f}"
        }
        class_rows.append(class_row)

    global_df = pd.DataFrame(global_rows)
    class_df = pd.DataFrame(class_rows)
    return global_df, class_df

"""Example Usage in Notebook
df_results = generate_metrics_table(results_vault)
display(df_results)
"""

def plot_model_comparison(all_results, metrics_to_plot=None, title="Model Performance Comparison"):
    """
    Plots mean performance with [Min, Max] error bars for multiple models.
    
    Args:
        all_results (dict): Dictionary where keys are model names and values are 
                            the output of your compute_cv_stats function.
        metrics_to_plot (list): List of metric keys to include (e.g., ['Macro-AUC', 'cmAP'])
    """
    if metrics_to_plot is None:
        metrics_to_plot = ['Macro-AUC', 'cmAP', 'Brier (macro) mean', 'Log-Loss (macro) mean']
    
    models = list(all_results.keys())
    n_metrics = len(metrics_to_plot)
    n_models = len(models)
    
    # Set up the plot dimensions
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Grouped bar settings
    width = 0.8 / n_models  # Total group width is 0.8
    x = np.arange(n_metrics)
    
    # Standard colors for bats/nature themes
    colors = plt.cm.viridis(np.linspace(0, 0.8, n_models))

    for i, model_name in enumerate(models):
        means = []
        lower_err = []
        upper_err = []
        
        for m_key in metrics_to_plot:
            stats = all_results[model_name][m_key] # [mean, max, min]
            
            mean_val = stats[0]
            max_val = stats[1]
            min_val = stats[2]
            
            means.append(mean_val)
            # Matplotlib yerr format: [ [lower_offsets], [upper_offsets] ]
            lower_err.append(mean_val - min_val)
            upper_err.append(max_val - mean_val)
        
        # Calculate x-offset for this specific model's bars
        offset = i * width - (width * n_models) / 2 + width / 2
        
        ax.bar(x + offset, means, width, 
               yerr=[lower_err, upper_err], 
               label=model_name, 
               color=colors[i],
               capsize=5, 
               alpha=0.85,
               edgecolor='white')

    # Formatting
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, fontsize=11)
    ax.set_ylabel("Score (Mean with Min/Max Range)")
    ax.legend(title="Algorithms", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_comprehensive_results(all_results, labels, title="Model Evaluation"):
    sns.set_context("paper")
    sns.set_style("whitegrid")
    
    # We now have 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(all_results)))
    width = 0.8 / len(all_results)

    # --- Subplot 1: Global Metrics (Macro-AUC & cmAP) ---
    ax1 = axes[0]
    global_keys = ['Macro-AUC', 'cmAP']
    x_global = np.arange(len(global_keys))

    for i, (model_name, stats) in enumerate(all_results.items()):
        means = [stats[k][0] for k in global_keys]
        yerr = [[stats[k][0]-stats[k][2] for k in global_keys], 
                [stats[k][1]-stats[k][0] for k in global_keys]]
        
        offset = i * width - (width * len(all_results)) / 2 + width / 2
        ax1.bar(x_global + offset, means, width, yerr=yerr, label=model_name, 
                color=colors[i], capsize=5, alpha=0.8, edgecolor='white')

    ax1.set_title("Global Performance: Macro-AUC vs cmAP", fontsize=14, fontweight='bold')
    ax1.set_xticks(x_global)
    ax1.set_xticklabels(['Macro-AUC', 'cmAP (mAP)'], fontsize=12)
    ax1.set_ylabel("Score (0.0 - 1.0)")
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc='upper right')

    # --- Subplot 2: Per-Class Average Precision ---
    ax2 = axes[1]
    x_labels = np.arange(len(labels))
    
    for i, (model_name, stats) in enumerate(all_results.items()):
        ap_data = stats['AP per label'] # [mean_dict, max_dict, min_dict]
        means = [ap_data[0][l] for l in labels]
        yerr = [[ap_data[0][l] - ap_data[2][l] for l in labels], 
                [ap_data[1][l] - ap_data[0][l] for l in labels]]
        
        offset = i * width - (width * len(all_results)) / 2 + width / 2
        ax2.bar(x_labels + offset, means, width, yerr=yerr, 
                color=colors[i], capsize=3, alpha=0.8, edgecolor='white')

    ax2.set_title("Per-Class Average Precision", fontsize=14, fontweight='bold')
    ax2.set_xticks(x_labels)
    ax2.set_xticklabels(labels, rotation=35, ha='right')
    ax2.set_ylabel("AP Score")
    ax2.set_ylim(0, 1.1)

    # --- Subplot 3: Error Metrics (Brier & Log-Loss) ---
    ax3 = axes[2]
    error_metrics = ['Brier (macro) mean', 'Log-Loss (macro) mean']
    x_err = np.arange(len(error_metrics))
    
    for i, (model_name, stats) in enumerate(all_results.items()):
        means = [stats[m][0] for m in error_metrics]
        yerr = [[stats[m][0]-stats[m][2] for m in error_metrics], 
                [stats[m][1]-stats[m][0] for m in error_metrics]]
        
        offset = i * width - (width * len(all_results)) / 2 + width / 2
        ax3.bar(x_err + offset, means, width, yerr=yerr, 
                color=colors[i], capsize=5, alpha=0.8, edgecolor='white')

    ax3.set_title("Calibration Error", fontsize=14, fontweight='bold')
    ax3.set_xticks(x_err)
    ax3.set_xticklabels(['Brier Score', 'Log-Loss'], fontsize=12)
    ax3.set_ylabel("Error Value")

    plt.suptitle(title, fontsize=20, y=1.02)
    plt.tight_layout()
    plt.show()



def plot_comprehensive_results2(all_results, labels, title="Model Evaluation"):
    sns.set_context("paper")
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 20)) # Increased size slightly
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(all_results)))
    width = 0.8 / len(all_results)
    
    handles, legend_labels = [], []

    # --- HELPER: Manual Zoom Function ---
    def apply_manual_zoom(ax, data_points, padding=0.1):
        """Forces the Y-axis to zoom in on the data range."""
        if not data_points: return
        d_min, d_max = min(data_points), max(data_points)
        diff = d_max - d_min
        # If there's no difference (e.g. all 1.0), use a default range
        if diff == 0:
            ax.set_ylim(d_min - 0.05, d_min + 0.05)
        else:
            ax.set_ylim(d_min - (diff * padding), d_max + (diff * padding))

    # --- Subplot 1: Global Metrics ---
    ax1 = axes[0]
    global_keys = ['Macro-AUC', 'cmAP']
    x_global = np.arange(len(global_keys))
    points_for_zoom1 = []

    for i, (model_name, stats) in enumerate(all_results.items()):
        means = [stats[k][0] for k in global_keys]
        low_err = [stats[k][0] - stats[k][2] for k in global_keys]
        high_err = [stats[k][1] - stats[k][0] for k in global_keys]
        
        # Track every low/high point for scaling
        points_for_zoom1.extend([m - l for m, l in zip(means, low_err)])
        points_for_zoom1.extend([m + h for m, h in zip(means, high_err)])
        
        offset = i * width - (width * len(all_results)) / 2 + width / 2
        bar = ax1.bar(x_global + offset, means, width, yerr=[low_err, high_err], 
                      color=colors[i], capsize=4, alpha=0.8, edgecolor='white')
        
        if model_name not in legend_labels:
            handles.append(bar)
            legend_labels.append(model_name)

    ax1.set_title("Global Performance: Macro-AUC vs cmAP", fontsize=15, fontweight='bold')
    ax1.set_xticks(x_global)
    ax1.set_xticklabels(['Macro-AUC', 'cmAP (mAP)'], fontsize=12)
    ax1.set_ylabel("Score")
    apply_manual_zoom(ax1, points_for_zoom1)

    # --- Subplot 2: Per-Class Average Precision ---
    ax2 = axes[1]
    x_labels = np.arange(len(labels))
    points_for_zoom2 = []
    
    for i, (model_name, stats) in enumerate(all_results.items()):
        ap_data = stats['AP per label']
        means = [ap_data[0][l] for l in labels]
        low_err = [ap_data[0][l] - ap_data[2][l] for l in labels]
        high_err = [ap_data[1][l] - ap_data[0][l] for l in labels]
        
        points_for_zoom2.extend([m - l for m, l in zip(means, low_err)])
        points_for_zoom2.extend([m + h for m, h in zip(means, high_err)])
        
        offset = i * width - (width * len(all_results)) / 2 + width / 2
        ax2.bar(x_labels + offset, means, width, yerr=[low_err, high_err], 
                color=colors[i], capsize=3, alpha=0.8, edgecolor='white')

    ax2.set_title("Per-Class Average Precision", fontsize=15, fontweight='bold')
    ax2.set_xticks(x_labels)
    ax2.set_xticklabels(labels, rotation=35, ha='right')
    ax2.set_ylabel("AP Score")
    apply_manual_zoom(ax2, points_for_zoom2)

    # --- Subplot 3: Error Metrics ---
    ax3 = axes[2]
    error_metrics = ['Brier (macro) mean', 'Log-Loss (macro) mean']
    x_err = np.arange(len(error_metrics))
    points_for_zoom3 = []
    
    for i, (model_name, stats) in enumerate(all_results.items()):
        means = [stats[m][0] for m in error_metrics]
        low_err = [stats[m][0] - stats[m][2] for m in error_metrics]
        high_err = [stats[m][1] - stats[m][0] for m in error_metrics]
        
        points_for_zoom3.extend([m - l for m, l in zip(means, low_err)])
        points_for_zoom3.extend([m + h for m, h in zip(means, high_err)])
        
        offset = i * width - (width * len(all_results)) / 2 + width / 2
        ax3.bar(x_err + offset, means, width, yerr=[low_err, high_err], 
                color=colors[i], capsize=5, alpha=0.8, edgecolor='white')

    ax3.set_title("Calibration Error (Lower is Better)", fontsize=15, fontweight='bold')
    ax3.set_xticks(x_err)
    ax3.set_xticklabels(['Brier Score', 'Log-Loss'], fontsize=12)
    ax3.set_ylabel("Error Value")
    apply_manual_zoom(ax3, points_for_zoom3)

    # --- Global Legend and Layout ---
    plt.suptitle(title, fontsize=22, y=1.02)
    
    # Legend at bottom with multiple rows if needed
    fig.legend(handles, legend_labels, loc='lower center', ncol=3, 
               bbox_to_anchor=(0.5, -0.05), fontsize=11, frameon=True)

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.show()


"""Implementation
# 1. Collect your stats into a dictionary
labels = ['Type A', 'Type B', 'Type C', 'Type D', 'Echo']
results_vault = {
    "Perch 2.0 SVM": compute_cv_stats(y_true_perch_svm, y_prob_perch_svm, label_names=labels),
    "Perch 2.0 RF": compute_cv_stats(y_true_perch_rf, y_prob_perch_rf, label_names=labels),
    "Perch 2.0 MLP": compute_cv_stats(y_true_perch_mlp, y_prob_perch_mlp, label_names=labels),
    "NLM BEATs": compute_cv_stats(y_true_beats, y_prob_beats, label_names=labels),
    "EffNet B0": compute_cv_stats(y_true_eff, y_prob_eff, label_names=labels)
    }

# 2. Call the plot
plot_model_comparison(
    all_results=results_vault, 
    metrics_to_plot=['Macro-AUC', 'cmAP'], # Choose which metrics to show
    title="Pipistrelle Classification: Encoder Comparison"
)
"""

def plot_calibration_curves(y_true,y_pred_proba,label_names=None,n_bins=10,strategy="quantile"):
    """
    Plot calibration curves + probability histograms
    for multilabel classification.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples, n_labels)
        Binary ground-truth matrix.

    y_pred_proba : ndarray of shape (n_samples, n_labels)
        Predicted probabilities.

    label_names : list[str], optional
        Names of labels.

    n_bins : int
        Number of calibration bins.

    strategy : {"uniform", "quantile"}
        Binning strategy for calibration_curve.
    """
    n_labels = y_true.shape[1]

    if label_names is None:
        label_names = [f"Label {i}" for i in range(n_labels)]

    # 2 rows: top calibration curves, bottom histograms
    fig, axes = plt.subplots(2,n_labels,figsize=(5 * n_labels, 8))

    # handle case n_labels == 1
    if n_labels == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for i in range(n_labels):
        # Skip degenerate labels
        if len(np.unique(y_true[:, i])) < 2:
            axes[0, i].set_visible(False)
            axes[1, i].set_visible(False)
            continue
        
        brier = brier_score_loss(y_true[:, i],y_pred_proba[:, i])
        # ---Calibration curve---------------
        prob_true, prob_pred = calibration_curve(y_true[:, i],y_pred_proba[:, i],n_bins=n_bins,strategy=strategy)

        ax_curve = axes[0, i]
        ax_curve.plot(prob_pred,prob_true,marker='o',linewidth=2)
        # perfect calibration
        ax_curve.plot([0, 1],[0, 1],linestyle='--',color='gray')

        ax_curve.set_title(f"{label_names[i]}\nBrier={brier:.3f}")
        ax_curve.set_xlabel("Mean predicted probability")
        ax_curve.set_ylabel("Fraction of positives")
        ax_curve.set_xlim(0, 1)
        ax_curve.set_ylim(0, 1)
        ax_curve.grid(True)

        # ---Histogram-----------------------
        ax_hist = axes[1, i]
        ax_hist.hist(y_pred_proba[:, i],bins=n_bins,alpha=0.7)

        ax_hist.set_title(f"{label_names[i]} Probability Distribution")
        ax_hist.set_xlabel("Predicted probability")
        ax_hist.set_ylabel("Count")
        ax_hist.set_xlim(0, 1)
        ax_hist.grid(True)

    plt.tight_layout()
    plt.show()




def label_confusion(y_true,y_pred_proba,y_pred_binary=None, label_names=None, threshold=0.5) :
    """
    Analyzes which labels are predicted 'instead' of the true labels.
    """
    # 1. Convert proba to binary if binary isn't provided
    if y_pred_binary is None:
        y_pred_binary = (y_pred_proba >= threshold).astype(int)
    
    y_true = np.array(y_true)
    y_pred_binary = np.array(y_pred_binary)
    num_labels = y_true.shape[1]
    
    if label_names is None:
        label_names = [f"Label_{i}" for i in range(num_labels)]

    # 2. Initialize Confusion Matrix
    # Rows: The label that was SHOULD have been there (False Negative)
    # Cols: The label that was predicted WRONGLY (False Positive)
    confusion_mtx = np.zeros((num_labels, num_labels))

    # 3. Iterate through samples
    for i in range(len(y_true)):
        actual = y_true[i]
        pred = y_pred_binary[i]

        # Indices of missed labels (FN)
        missed = np.where((actual == 1) & (pred == 0))[0]
        # Indices of extra labels (FP)
        extra = np.where((actual == 0) & (pred == 1))[0]

        # If we missed something AND predicted something else wrongly
        for m_idx in missed:
            for e_idx in extra:
                confusion_mtx[m_idx, e_idx] += 1

    # 4. Wrap in DataFrame for easy viewing
    df_cm = pd.DataFrame(confusion_mtx, index=label_names, columns=label_names)
    
    return df_cm

