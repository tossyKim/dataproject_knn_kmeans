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

    for train, label in zip(X_train,y_train):
        dist = cal_dist(data, train)
        dists.append((dist,label))

    dists.sort(key=lambda x: x[0])
    k_neighbors = dists[:k]


    for label in labels:
        count_label[label] = 0

    for _, neighbor_label in k_neighbors:
        count_label[neighbor_label] +=1

    predict = max(count_label, key=count_label.get)
    print("count_label:", count_label)
    print("predict:", predict)
    drawResult(X_train,y_train,data,predict)


def drawResult(X_train, y_train, data, predict):
    colors = ['red', 'green', 'blue']
    feature_indices = [0, 1, 2, 3]
    pairs = []

    # 모든 조합 생성
    for i in range(len(feature_indices)):
        for j in range(i + 1, len(feature_indices)):
            pairs.append((i, j))

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx]
        for x, y in zip(X_train, y_train):
            ax.scatter(x[i], x[j], color=colors[y], alpha=0.5)
        # 예측 데이터 표시
        ax.scatter(data[i], data[j], color=colors[predict], marker='X', s=150, label=f'Predict: {predict}')
        ax.set_xlabel(f'Feature {i}')
        ax.set_ylabel(f'Feature {j}')
        ax.set_title(f'Feature {i} vs Feature {j}')
        ax.legend()

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
