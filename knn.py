import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import math
import numpy as np

iris = load_iris()
X = iris.data
Y = iris.target

def cal_dist(x, y):
    diffs = []
    for a, b in zip(x, y):
        diff = a - b
        squared = diff ** 2
        diffs.append(squared)
    total = sum(diffs)
    result = math.sqrt(total)
    return result

def knn_predict(X_train, y_train, k, data):
    dists = []
    count_label = {}
    labels = set(y_train)

    for train, label in zip(X_train,y_train):# 임의 데이터와 모든 데이터사이 거리 계산
        dist = cal_dist(data, train)
        dists.append((dist,label))

    dists.sort(key=lambda x: x[0])# 가까운 데이터 k개만 추출
    k_neighbors = dists[:k]


    for label in labels:
        count_label[label] = 0

    for _, neighbor_label in k_neighbors:# 가까운 데이터의 클래스 세기
        count_label[neighbor_label] +=1

    predict = max(count_label, key=count_label.get)# 가장 많은 클래스로 분류
    print("count_label:", count_label)
    print("predict:", predict)
    drawResult(X_train,y_train,data,predict)


def drawResult(X_train, y_train, data, predict):
    colors = ['red', 'green', 'blue']
    X_train = np.array(X_train)
    n_features = X_train.shape[1]

    # feature 쌍 생성
    pairs = [(i, j) for i in range(n_features) for j in range(i + 1, n_features)]
    n_pairs = len(pairs)

    # subplot 크기 동적 계산
    n_cols = min(3, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols

    # 그래프 기본 설정
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    fig.suptitle("KNN", fontsize=20, fontweight='bold')
    axes = np.array(axes).reshape(-1)

    # 각 feature 쌍별 시각화
    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx]
        # 학습 데이터
        for x, y in zip(X_train, y_train):
            ax.scatter(x[i], x[j], color=colors[y], alpha=0.5)
        # 예측 데이터
        ax.scatter(data[i], data[j], color=colors[predict], marker='X', s=150,
                   edgecolor='black', label=f'Predict: {predict}')
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

    num0 = random.uniform(X[:,0].min(), X[:,0].max())
    num1 = random.uniform(X[:,1].min(), X[:,1].max())
    num2 = random.uniform(X[:,2].min(), X[:,2].max())
    num3 = random.uniform(X[:,3].min(), X[:,3].max())

    data = [num0,num1,num2,num3]
    print(data)
    knn_predict(X,Y,15,data)