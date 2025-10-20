import matplotlib.pyplot as plt
from keras.src.ops import shape
from sklearn.datasets import load_iris
import math
import numpy as np

iris = load_iris()
x = iris.data
X = list(x)


def cal_center(data):
    mean_data = np.mean(data, axis=0).tolist() #평균으로 그룹의 중심점 구하기
    return mean_data


def cal_dist(data1, data2):# 데이터 사이 거리 구하기
    if isinstance(data1[0], (list, np.ndarray)):
        data1 = cal_center(data1)

    if isinstance(data2[0], (list, np.ndarray)):
        data2 = cal_center(data2)

    data1 = np.array(data1, dtype=float).flatten()
    data2 = np.array(data2, dtype=float).flatten()

    diffs = []
    for a, b in zip(data1, data2):
        diff = a - b
        squared = diff ** 2
        diffs.append(squared)
    total = sum(diffs)
    result = math.sqrt(total)
    return result


def clustering(X, k):
    X_copy = X.copy()
    clusters = [[x] for x in X_copy] # 모든 그룹 리스트

    while len(clusters) > k: # 모든 데이터 사이 거리 계산
        dists = {}
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = cal_dist(clusters[i], clusters[j])# 거리 계산
                dists[(i, j)] = dist # {(데이터1,데이터2):데이터 사이 거리} 형식으로 저장

        min_key = min(dists, key=dists.get) # 가장 거리가 짧은 데이터 선택
        index1, index2 = min_key

        new_cluster = clusters[index1] + clusters[index2]#거리가 가장 짧은 데이터끼리 합치기
        print(clusters[index1])
        print(clusters[index2])
        print(new_cluster)
        print("=================")
        clusters.append(new_cluster)

        # #합친 데이터 원본은 삭제 (큰 인덱스부터)
        for idx in sorted([index1, index2], reverse=True):
            del clusters[idx]

    drawClusters(X,clusters)

def drawClusters(X, clusters):
    X = np.array(X)
    n_features = X.shape[1]
    k = len(clusters)

    colors_list = ['red', 'blue', 'green']

    # 모든 feature 쌍
    pairs = [(i, j) for i in range(n_features) for j in range(i + 1, n_features)]
    n_pairs = len(pairs)
    n_cols = min(3, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx]
        for cluster_idx, cluster in enumerate(clusters):
            cluster = np.array(cluster)
            if len(cluster) > 0:
                color = colors_list[cluster_idx % len(colors_list)]
                ax.scatter(cluster[:, i], cluster[:, j], color=color, alpha=0.7, label=f'Cluster {cluster_idx}')
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
    clustering(X, 3)

