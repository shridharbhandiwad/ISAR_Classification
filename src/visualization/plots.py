"""Plotting utilities for metrics and results visualization."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    normalize: bool = True,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix with annotations.
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        normalize: Whether to normalize
        title: Plot title
        cmap: Colormap
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if normalize:
        cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)
        fmt = '.2f'
    else:
        cm = confusion_matrix
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap=cmap,
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, square=True, cbar_kws={'shrink': 0.8}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix_plotly(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    normalize: bool = True,
    title: str = "Confusion Matrix"
) -> go.Figure:
    """
    Create interactive confusion matrix with Plotly.
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        normalize: Whether to normalize
        title: Plot title
        
    Returns:
        Plotly figure
    """
    if normalize:
        cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)
        text = [[f'{val:.2%}' for val in row] for row in cm]
    else:
        cm = confusion_matrix
        text = [[str(val) for val in row] for row in cm]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        text=text,
        texttemplate='%{text}',
        colorscale='Blues',
        showscale=True
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        height=500,
        width=600
    )
    
    return fig


def plot_roc_curves(
    roc_curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "ROC Curves",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curves for all classes.
    
    Args:
        roc_curves: Dictionary mapping class names to (fpr, tpr) tuples
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(roc_curves)))
    
    for (class_name, (fpr, tpr)), color in zip(roc_curves.items(), colors):
        # Calculate AUC
        auc = np.trapz(tpr, fpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{class_name} (AUC = {auc:.3f})')
    
    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_roc_curves_plotly(
    roc_curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "ROC Curves"
) -> go.Figure:
    """
    Create interactive ROC curves with Plotly.
    
    Args:
        roc_curves: Dictionary mapping class names to (fpr, tpr) tuples
        title: Plot title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    for class_name, (fpr, tpr) in roc_curves.items():
        auc = np.trapz(tpr, fpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{class_name} (AUC = {auc:.3f})'
        ))
    
    # Diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500,
        width=700
    )
    
    return fig


def plot_precision_recall_curves(
    pr_curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "Precision-Recall Curves",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot Precision-Recall curves for all classes.
    
    Args:
        pr_curves: Dictionary mapping class names to (precision, recall) tuples
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(pr_curves)))
    
    for (class_name, (precision, recall)), color in zip(pr_curves.items(), colors):
        # Calculate AP
        ap = np.trapz(precision, recall)
        ax.plot(recall, precision, color=color, lw=2, label=f'{class_name} (AP = {ap:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_history(
    history: List[Dict[str, Any]],
    metrics: List[str] = ['loss', 'accuracy'],
    title: str = "Training History",
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training history curves.
    
    Args:
        history: List of epoch dictionaries with metrics
        metrics: Metrics to plot
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    epochs = [h['epoch'] for h in history]
    
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        train_values = [h['train'].get(metric, h['train'].get('loss')) for h in history]
        val_values = [h['val'].get(metric, h['val'].get('loss')) for h in history]
        
        ax.plot(epochs, train_values, 'b-', label=f'Train {metric}', linewidth=2)
        ax.plot(epochs, val_values, 'r-', label=f'Val {metric}', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'{metric.capitalize()} over Epochs', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_history_plotly(
    history: List[Dict[str, Any]],
    title: str = "Training History"
) -> go.Figure:
    """
    Create interactive training history plot with Plotly.
    
    Args:
        history: List of epoch dictionaries
        title: Plot title
        
    Returns:
        Plotly figure
    """
    epochs = [h['epoch'] for h in history]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Loss', 'Accuracy')
    )
    
    # Loss
    fig.add_trace(
        go.Scatter(x=epochs, y=[h['train']['loss'] for h in history],
                   name='Train Loss', mode='lines'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=[h['val']['loss'] for h in history],
                   name='Val Loss', mode='lines'),
        row=1, col=1
    )
    
    # Accuracy
    fig.add_trace(
        go.Scatter(x=epochs, y=[h['train']['accuracy'] for h in history],
                   name='Train Acc', mode='lines'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=[h['val']['accuracy'] for h in history],
                   name='Val Acc', mode='lines'),
        row=1, col=2
    )
    
    fig.update_layout(title=title, height=400, width=1000)
    
    return fig


def plot_class_distribution(
    class_counts: Dict[str, int],
    title: str = "Class Distribution",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot class distribution bar chart.
    
    Args:
        class_counts: Dictionary of class name -> count
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    bars = ax.bar(classes, counts, color='steelblue', edgecolor='black')
    
    # Add value labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_feature_embeddings(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    method: str = 'tsne',
    title: str = "Feature Embeddings",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot 2D feature embeddings using dimensionality reduction.
    
    Args:
        features: Feature array (N, D)
        labels: Label array (N,)
        class_names: List of class names
        method: Reduction method ('tsne' or 'pca')
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Reduce dimensions
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        reducer = PCA(n_components=2)
    
    embeddings = reducer.fit_transform(features)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        label_name = class_names[label] if class_names else str(label)
        ax.scatter(
            embeddings[mask, 0], embeddings[mask, 1],
            c=[color], label=label_name, alpha=0.7, s=30
        )
    
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    ax.set_title(f'{title} ({method.upper()})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_feature_embeddings_plotly(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    method: str = 'tsne',
    title: str = "Feature Embeddings"
) -> go.Figure:
    """
    Create interactive feature embeddings plot with Plotly.
    
    Args:
        features: Feature array
        labels: Label array
        class_names: List of class names
        method: Reduction method
        title: Plot title
        
    Returns:
        Plotly figure
    """
    # Reduce dimensions
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        reducer = PCA(n_components=2)
    
    embeddings = reducer.fit_transform(features)
    
    # Create labels for plotting
    if class_names:
        label_names = [class_names[l] for l in labels]
    else:
        label_names = [str(l) for l in labels]
    
    fig = px.scatter(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        color=label_names,
        title=f'{title} ({method.upper()})',
        labels={'x': 'Component 1', 'y': 'Component 2', 'color': 'Class'}
    )
    
    fig.update_layout(height=600, width=800)
    
    return fig
