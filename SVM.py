import numpy as np
import matplotlib.pyplot as plt


class LinearSVC:
    def __init__(self, C=1.0):
        self.C = C
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape

        # 初始化模型参数
        self.weights = np.zeros(num_features)
        self.bias = 0

        # 计算Gram矩阵
        gram = np.dot(X, X.T)

        # 使用SMO算法更新模型参数
        alphas = np.zeros(num_samples)
        num_iterations = 1000  # 迭代次数
        for _ in range(num_iterations):
            changed_alphas = 0
            for i in range(num_samples):
                error_i = self._decision_function(X[i]) - y[i]
                # 当第i个样本被错误分类（即预测值与真实标签相反),且当前的拉格朗日乘子小于上限C
                if (y[i] * error_i < -1e-3 and alphas[i] < self.C) or (y[i] * error_i > 1e-3 and alphas[i] > 0):
                    j = self._select_random_index(i, num_samples)
                    error_j = self._decision_function(X[j]) - y[j]
                    old_alpha_i, old_alpha_j = alphas[i], alphas[j]

                    # 计算上下界
                    if y[i] != y[j]:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(self.C, self.C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[i] + alphas[j] - self.C)
                        H = min(self.C, alphas[i] + alphas[j])

                    # 计算学习率 eta
                    eta = 2 * gram[i, j] - gram[i, i] - gram[j, j]
                    if eta >= 0:
                        continue

                    # 更新 alpha[j]
                    alphas[j] -= y[j] * (error_i - error_j) / eta
                    alphas[j] = np.clip(alphas[j], L, H)

                    if abs(alphas[j] - old_alpha_j) < 1e-5:
                        continue

                    # 更新 alpha[i]
                    alphas[i] += y[i] * y[j] * (old_alpha_j - alphas[j])

                    # 更新模型参数
                    self.weights = np.dot(alphas * y, X)
                    self.bias = np.mean(y - np.dot(X, self.weights))
                    changed_alphas += 1

            if changed_alphas == 0:
                break

    def _decision_function(self, X):
        return np.dot(X, self.weights) + self.bias

    def _select_random_index(self, exclude_index, num_samples):
        index = exclude_index
        while index == exclude_index:
            index = np.random.randint(num_samples)
        return index

    def predict(self, X):
        return np.where(self._decision_function(X) >= 0, 1, -1)


if __name__ == '__main__':


    # 准备数据集
    X = np.array([[1, 4], [1, 3], [2, 3], [2, 5], [9, 1], [61, 12]])
    y = np.array([0, 0, 0, 0, 1, 1])

    # 预测数据集中的异常值
    X_test = np.array([[3, 3], [4, 4], [12, 12], [13, 13], [20, 20]])

    # 创建LinearSVC模型
    clf = LinearSVC()
    # 拟合数据
    clf.fit(X, y)
    # 预测
    y_pred = clf.predict(X_test)

    # 绘制数据点
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, label='Training Samples')
    plt.plot(X_test[:, 0], X_test[:, 1], 'k-', linewidth=2, label='Predicted Outliers')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Linear SVM for Outlier Detection')

    plt.legend()
    plt.show()