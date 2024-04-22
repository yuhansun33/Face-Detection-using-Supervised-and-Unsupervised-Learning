import dataset
import plots
import detection
import matplotlib.pyplot as plt
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, mutual_info_score, homogeneity_score, completeness_score, v_measure_score
from sklearn.model_selection import KFold

def evaluate_clustering(X, labels, n_clusters, n_splits):
    kf = KFold(n_splits=n_splits)
    scores = {
        'Adjusted Rand Index': [],
        'Mutual Information': [],
        'Homogeneity': [],
        'Completeness': [],
        'V-measure': []
    }

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]

        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        kmeans.fit(X_train)
        pred_labels = kmeans.predict(X_test)

        scores['Adjusted Rand Index'].append(adjusted_rand_score(labels_test, pred_labels))
        scores['Mutual Information'].append(mutual_info_score(labels_test, pred_labels))
        scores['Homogeneity'].append(homogeneity_score(labels_test, pred_labels))
        scores['Completeness'].append(completeness_score(labels_test, pred_labels))
        scores['V-measure'].append(v_measure_score(labels_test, pred_labels))

    for metric, score_list in scores.items():
        print(f"{metric}: {np.mean(score_list):.3f} +/- {np.std(score_list):.3f}")

    metrics = list(scores.keys())
    # 繪製聚類評估指標的橫條圖
    plots.arch(scores, metrics)
    # 繪製聚類評估指標的箱線圖
    plots.box(scores, metrics)



def main():
    print('Loading images')
    img_dataset, labels = dataset.create_dataset()
    X = np.array(img_dataset)
    labels = np.array(labels)
    print(f'The number of samples loaded: {len(X)}')

    n_clusters = 2
    n_splits = 15

    print(f"Evaluating clustering with {n_splits}-fold cross-validation:")
    evaluate_clustering(X, labels, n_clusters, n_splits)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(X)
    pred_labels = kmeans.predict(X)

    # 輸出每個聚類的樣本數量
    for i in range(n_clusters):
        cluster_labels = labels[pred_labels == i]
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        print(f"Cluster {i}:")
        for label, count in zip(unique_labels, counts):
            print(f"  Label {label}: {count} samples")

    # 繪製聚類結果的散點圖（假設數據為2D）
    plots.scatter(X, pred_labels, kmeans, n_clusters)

    # 顯示HOG結果
    # plots.show_cluster(X, pred_labels, n_clusters)

    # 預測
    print('Detecting faces')
    detection.detect(kmeans, "detect/detectData.txt")
    detection.detect(kmeans, "detect/yourOwnImages.txt")


if __name__ == "__main__":
    main()