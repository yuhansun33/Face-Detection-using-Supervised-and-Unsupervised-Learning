import numpy as np
import matplotlib.pyplot as plt

def arch(scores, metrics):
    fig, ax = plt.subplots(figsize=(8, 4))
    
    means = [np.mean(scores[metric]) for metric in metrics]
    stds = [np.std(scores[metric]) for metric in metrics]
    ax.barh(metrics, means, xerr=stds, align='center', alpha=0.8)
    ax.set_xlabel('Score')
    ax.set_title('Clustering Evaluation Metrics')
    plt.tight_layout()
    plt.show()
def box(scores, metrics):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot([scores[metric] for metric in metrics], labels=metrics)
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('Clustering Evaluation Metrics')
    plt.tight_layout()
    plt.show()

def scatter(X, pred_labels, kmeans, n_clusters):
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ['red', 'blue']
    for i in range(n_clusters):
        mask = pred_labels == i
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[i], label=f'Cluster {i}', alpha=0.8)
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', marker='x', s=100, label='Centroids')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Clustering Result')
    ax.legend()
    plt.tight_layout()
    plt.show()
    
def show_cluster(X, pred_labels, n_clusters):
    # 顯示分類為0和1的各一張照片
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    for i in range(n_clusters):
        cluster_samples = X[pred_labels == i]
        sample_image = cluster_samples[0].reshape(36, 49)  # 重塑為(36, 49)的形狀
        axs[i].imshow(sample_image, cmap='gray')
        axs[i].set_title(f'Cluster {i}')
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()