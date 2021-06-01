import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

# Distribution parameters
priors = np.array([0.3, 0.3, 0.4])
m1 = np.array([0, 0, 0])
C1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
m2 = np.array([2, 2, 0])
C2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
m01 = np.array([0, 1, 2])
C01 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
m02 = np.array([1, 2, 2])
C02 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Generate samples
N = 10000
X = np.zeros(shape=(3, N))
true_labels = []
for i in range(N):
    s = random.uniform(0, 1)
    if s < 0.3:
        true_labels.append(1)
    elif s < 0.6:
        true_labels.append(2)
    else:
        true_labels.append(3)
for i in range(N):
    if true_labels[i] == 1:
        X[:, i] = np.random.multivariate_normal(mean=m1, cov=C1).T
    elif true_labels[i] == 2:
        X[:, i] = np.random.multivariate_normal(mean=m2, cov=C2).T
    elif true_labels[i] == 3:
        if random.uniform(0, 1) < 0.5:
            X[:, i] = np.random.multivariate_normal(mean=m01, cov=C01).T
        else:
            X[:, i] = np.random.multivariate_normal(mean=m02, cov=C02).T

# Plot samples
fig = plt.figure()
ax = plt.axes(projection='3d')
colors = []
for i in range(N):
    if true_labels[i] == 1:
        colors.append('red')
    elif true_labels[i] == 2:
        colors.append('green')
    elif true_labels[i] == 3:
        colors.append('blue')
ax.scatter(X[0], X[1], X[2], color=colors)
plt.title('True Class Labels')
plt.legend(handles=[Line2D([0], [0], marker='o', color='w', label='Class 1', markerfacecolor='r', markersize=10),
                    Line2D([0], [0], marker='o', color='w', label='Class 2', markerfacecolor='g', markersize=10),
                    Line2D([0], [0], marker='o', color='w', label='Class 3', markerfacecolor='b', markersize=10)])
plt.show()

# Calculate posteriors
pxg1 = priors[0]*stats.multivariate_normal.pdf(X.T, m1, C1)
pxg2 = priors[1]*stats.multivariate_normal.pdf(X.T, m2, C2)
pxg3 = priors[2]*(0.5*stats.multivariate_normal.pdf(X.T, m01, C01) + 0.5*stats.multivariate_normal.pdf(X.T, m02, C02))
px = pxg1 + pxg2 + pxg3
posteriors = np.array([np.divide(pxg1, px), np.divide(pxg2, px), np.divide(pxg3, px)])

# MAP classifications
D = np.array([np.argmax([posteriors[0][i], posteriors[1][i], posteriors[2][i]]) + 1 for i in range(N)])
D1 = D[[i == 1 for i in true_labels]]
D2 = D[[i == 2 for i in true_labels]]
D3 = D[[i == 3 for i in true_labels]]
confusion_matrix = np.zeros(shape=(3, 3))
for d in range(3):
    for l in range(3):
        decision_count = np.count_nonzero([D1, D2, D3][l] == d + 1)
        label_count = len(D)
        confusion_matrix[d][l] = decision_count / label_count
print('MAP Confusion Matrix:')
print(confusion_matrix)
fig = plt.figure()
ax = plt.axes(projection='3d')
L1 = X[:, [i == 1 for i in true_labels]]
L2 = X[:, [i == 2 for i in true_labels]]
L3 = X[:, [i == 3 for i in true_labels]]
colors1 = ['green' if D1[i] == 1 else 'red' for i in range(len(D1))]
colors2 = ['green' if D2[i] == 2 else 'red' for i in range(len(D2))]
colors3 = ['green' if D3[i] == 3 else 'red' for i in range(len(D3))]
ax.scatter(L1[0], L1[1], L1[2], marker='o', color=colors1)
ax.scatter(L2[0], L2[1], L2[2], marker='s', color=colors2)
ax.scatter(L3[0], L3[1], L3[2], marker='^', color=colors3)
plt.title('MAP Classifications')
plt.legend(handles=[Line2D([0], [0], marker='o', color='w', label='Class 1', markerfacecolor='black', markersize=10),
                    Line2D([0], [0], marker='s', color='w', label='Class 2', markerfacecolor='black', markersize=10),
                    Line2D([0], [0], marker='^', color='w', label='Class 3', markerfacecolor='black', markersize=10)])
plt.show()

# ERM classifications (cares 10 times more about L3)
loss_matrix = np.array([[0, 1, 10], [1, 0, 10], [1, 1, 0]])
risk_matrix = np.matmul(loss_matrix, posteriors)
D = np.array([np.argmin([risk_matrix[0][i], risk_matrix[1][i], risk_matrix[2][i]]) + 1 for i in range(N)])
D1 = D[[i == 1 for i in true_labels]]
D2 = D[[i == 2 for i in true_labels]]
D3 = D[[i == 3 for i in true_labels]]
confusion_matrix = np.zeros(shape=(3, 3))
for d in range(3):
    for l in range(3):
        D = [D1, D2, D3][l]
        decision_count = np.count_nonzero(D == d + 1)
        label_count = len(D)
        confusion_matrix[d][l] = decision_count / label_count
print('ERM 10 Confusion Matrix:')
print(confusion_matrix)
fig = plt.figure()
ax = plt.axes(projection='3d')
L1 = X[:, [i == 1 for i in true_labels]]
L2 = X[:, [i == 2 for i in true_labels]]
L3 = X[:, [i == 3 for i in true_labels]]
colors1 = ['green' if D1[i] == 1 else 'red' for i in range(len(D1))]
colors2 = ['green' if D2[i] == 2 else 'red' for i in range(len(D2))]
colors3 = ['green' if D3[i] == 3 else 'red' for i in range(len(D3))]
ax.scatter(L1[0], L1[1], L1[2], marker='o', color=colors1)
ax.scatter(L2[0], L2[1], L2[2], marker='s', color=colors2)
ax.scatter(L3[0], L3[1], L3[2], marker='^', color=colors3)
plt.title('ERM 10 Classifications')
plt.legend(handles=[Line2D([0], [0], marker='o', color='w', label='Class 1', markerfacecolor='black', markersize=10),
                    Line2D([0], [0], marker='s', color='w', label='Class 2', markerfacecolor='black', markersize=10),
                    Line2D([0], [0], marker='^', color='w', label='Class 3', markerfacecolor='black', markersize=10)])
plt.show()

# ERM classifications (cares 10 times more about L3)
loss_matrix = np.array([[0, 1, 100], [1, 0, 100], [1, 1, 0]])
risk_matrix = np.matmul(loss_matrix, posteriors)
D = np.array([np.argmin([risk_matrix[0][i], risk_matrix[1][i], risk_matrix[2][i]]) + 1 for i in range(N)])
D1 = D[[i == 1 for i in true_labels]]
D2 = D[[i == 2 for i in true_labels]]
D3 = D[[i == 3 for i in true_labels]]
confusion_matrix = np.zeros(shape=(3, 3))
for d in range(3):
    for l in range(3):
        D = [D1, D2, D3][l]
        decision_count = np.count_nonzero(D == d + 1)
        label_count = len(D)
        confusion_matrix[d][l] = decision_count / label_count
print('ERM 100 Confusion Matrix:')
print(confusion_matrix)
fig = plt.figure()
ax = plt.axes(projection='3d')
L1 = X[:, [i == 1 for i in true_labels]]
L2 = X[:, [i == 2 for i in true_labels]]
L3 = X[:, [i == 3 for i in true_labels]]
colors1 = ['green' if D1[i] == 1 else 'red' for i in range(len(D1))]
colors2 = ['green' if D2[i] == 2 else 'red' for i in range(len(D2))]
colors3 = ['green' if D3[i] == 3 else 'red' for i in range(len(D3))]
ax.scatter(L1[0], L1[1], L1[2], marker='o', color=colors1)
ax.scatter(L2[0], L2[1], L2[2], marker='s', color=colors2)
ax.scatter(L3[0], L3[1], L3[2], marker='^', color=colors3)
plt.title('ERM 100 Classifications')
plt.legend(handles=[Line2D([0], [0], marker='o', color='w', label='Class 1', markerfacecolor='black', markersize=10),
                    Line2D([0], [0], marker='s', color='w', label='Class 2', markerfacecolor='black', markersize=10),
                    Line2D([0], [0], marker='^', color='w', label='Class 3', markerfacecolor='black', markersize=10)])
plt.show()
