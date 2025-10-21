import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import math
import numpy as np

# iris 데이터 로드
iris = load_iris()
X = iris.data

# 거리 계산 함수 (유클리드)
def cal_dist(x, y):
    diffs = []
    for a, b in zip(x, y):
        diff = a - b
        squared = diff ** 2
        diffs.append(squared)
    total = sum(diffs)
    result = math.sqrt(total)
    return result

# KMeans 함수 (중심점 이동 기록 없이 최종 중심만 반환)
def kmeans(X, k, max_iter):
    n_samples, n_features = X.shape

    # 초기 중심점 랜덤 선택
    centroids = []  # 중심점을 담을 리스트

    for _ in range(k):
        centroid = []
        for i in range(n_features):
            min_val = X[:, i].min()
            max_val = X[:, i].max()
            rand_val = random.uniform(min_val, max_val)
            centroid.append(rand_val)
        centroids.append(centroid)

    for iteration in range(max_iter):
        clusters = [[] for _ in range(k)]

        # 각 데이터에 대해 가장 가까운 중심점으로 할당
        for x in X:
            distances=[]
            for centroid in centroids:
                distance = cal_dist(x,centroid)
                distances.append(distance)

            cluster_idx = distances.index(min(distances))
            clusters[cluster_idx].append(x)

        # 중심점 이동
        new_centroids = []
        for cluster_idx, cluster in enumerate(clusters):
            if cluster:
                new_centroid = np.mean(cluster, axis=0).tolist()
            else:
                new_centroid = centroids[cluster_idx]  # 클러스터가 비어있으면 기존 중심 유지
            new_centroids.append(new_centroid)

        if np.allclose(centroids, new_centroids, atol=1e-4):
            break

        centroids = new_centroids

    # 최종 클러스터 레이블
        labels = np.zeros(n_samples, dtype=int)
    for i, x in enumerate(X):
        distances = [cal_dist(x, centroid) for centroid in centroids]
        labels[i] = distances.index(min(distances))

    return clusters, centroids, labels

# 시각화 (모든 feature 쌍별 subplot)
def drawKMeans(X, clusters, centroids):
    X = np.array(X)
    n_features = X.shape[1]
    k = len(clusters)
    colors = plt.cm.get_cmap('tab10', k)

    # 모든 feature 쌍
    pairs = [(i, j) for i in range(n_features) for j in range(i+1, n_features)]
    n_pairs = len(pairs)
    n_cols = min(3, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    fig.suptitle("K-means", fontsize=20, fontweight='bold')
    axes = np.array(axes).reshape(-1)

    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx]
        # 데이터 시각화
        for cluster_idx, cluster in enumerate(clusters):
            cluster = np.array(cluster)
            if len(cluster) > 0:
                ax.scatter(cluster[:, i], cluster[:, j], color=colors(cluster_idx), alpha=0.5, label=f'Cluster {cluster_idx}')
        # 최종 중심점 표시
        cent = np.array(centroids)
        ax.scatter(cent[:, i], cent[:, j], marker='X', s=150, color=[colors(c) for c in range(k)], edgecolor='black')
        ax.set_xlabel(f'Feature {i}')
        ax.set_ylabel(f'Feature {j}')
        ax.set_title(f'Feature {i} vs Feature {j}')
        ax.legend()

    # 남는 subplot 제거
    for ax in axes[n_pairs:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    clusters, centroids, labels = kmeans(X, k=3, max_iter=100)
    drawKMeans(X, clusters, centroids)
