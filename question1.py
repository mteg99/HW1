import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

# Distribution parameters
priors = np.array([0.65, 0.35])
m01 = np.array([3, 0])
C01 = np.array([[2, 0], [0, 1]])
m02 = np.array([0, 3])
C02 = np.array([[1, 0], [0, 2]])
m1 = np.array([2, 2])
C1 = np.array([[1, 0], [0, 1]])

# Generate samples
N = 10000
X = np.zeros(shape=(2, N))
true_labels = [0 if random.uniform(0, 1) < 0.65 else 1 for i in range(N)]
for i in range(N):
    if true_labels[i] == 0:
        if random.uniform(0, 1) < 0.5:
            X[:, i] = np.random.multivariate_normal(mean=m01, cov=C01).T
        else:
            X[:, i] = np.random.multivariate_normal(mean=m02, cov=C02).T
    else:
        X[:, i] = np.random.multivariate_normal(mean=m1, cov=C1).T

# Plot samples
plt.scatter(X[0], X[1], color=['blue' if i == 0 else 'red' for i in true_labels])
plt.title('Question 1 True Class Labels')
plt.legend(handles=[Line2D([0], [0], marker='o', color='w', label='Class 0', markerfacecolor='b', markersize=10),
                    Line2D([0], [0], marker='o', color='w', label='Class 1', markerfacecolor='r', markersize=10)])
plt.show()

# Estimate mean and covariance
L0 = X[:, [i == 0 for i in true_labels]]
L1 = X[:, [i != 0 for i in true_labels]]
mu0 = np.mean(L0, axis=1)
mu1 = np.mean(L1, axis=1)
sigma0 = np.cov(L0)
sigma1 = np.cov(L1)

# Calculate scatter matrices and w
mu_diff = np.subtract(mu0, mu1)
sb = np.matmul(mu_diff, mu_diff.T)
sw = np.add(sigma0, sigma1)
values, vectors = np.linalg.eig(np.linalg.inv(sw)*sb)
w = vectors.T[np.argmax(values)]
if np.mean(np.matmul(w.T, L1)) <= np.mean(np.matmul(w.T, L0)):
    w = -w

# Plot LDA
wTX = np.matmul(w.T, X)
plt.scatter(wTX, np.zeros(N), color=['blue' if i == 0 else 'red' for i in true_labels])
plt.title('LDA Projection')
plt.legend(handles=[Line2D([0], [0], marker='o', color='w', label='Class 0', markerfacecolor='b', markersize=10),
                    Line2D([0], [0], marker='o', color='w', label='Class 1', markerfacecolor='r', markersize=10)])
plt.show()

# Plot ROC curve for LDA
FPR = []
TPR = []
taus = []
p_error = []
wTL0 = np.matmul(w.T, L0)
wTL1 = np.matmul(w.T, L1)
tau_min = np.floor(np.min(wTX))
tau_max = np.ceil(np.max(wTX))
tau_inc = (tau_max - tau_min) / 100
tau = tau_min
for i in range(100):
    FP = np.count_nonzero(wTL0 >= tau)
    N = np.size(wTL0)
    FPR.append(FP / N)
    TP = np.count_nonzero(wTL1 >= tau)
    P = np.size(wTL1)
    TPR.append(TP / P)
    FN = np.count_nonzero(wTL1 < tau)
    p_error.append((FP / N)*priors[0] + (FN / P)*priors[1])
    taus.append(tau)
    tau = tau + tau_inc
plt.plot(FPR, TPR, color='blue')
i = np.argmin(p_error)
plt.scatter(FPR[i], TPR[i], color='green')
plt.legend(handles=[Line2D([0], [0], marker='o', color='w', label='Optimal Tau', markerfacecolor='g', markersize=10)])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()
print('Optimal tau: ')
print(taus[i])
print('Min error: ')
print(p_error[i])

# plot ROC curve for ERM
FPR = []
TPR = []
gammas = []
p_error = []
pxL0 = stats.multivariate_normal.pdf(X.T, mu0, sigma0)
pxL1 = stats.multivariate_normal.pdf(X.T, mu1, sigma1)
d = np.divide(pxL1, pxL0)
dL0 = d[[i == 0 for i in true_labels]]
dL1 = d[[i != 0 for i in true_labels]]
gamma_min = 0
gamma_max = np.ceil(np.max(d))
gamma_inc = (gamma_max - gamma_min) / 100
gamma = gamma_min

# vary from 0 to inf
for i in range(100):
    FP = np.count_nonzero(dL0 >= gamma)
    N = np.size(dL0)
    FPR.append(FP / N)
    TP = np.count_nonzero(dL1 >= gamma)
    P = np.size(dL1)
    TPR.append(TP / P)
    FN = np.count_nonzero(dL1 < gamma)
    p_error.append((FP / N)*priors[0] + (FN / P)*priors[1])
    gammas.append(gamma)
    gamma = gamma + gamma_inc
plt.plot(FPR, TPR, color='blue')

# theoretically optimal
gamma = priors[0] / priors[1]
FP = np.count_nonzero(dL0 >= gamma)
N = np.size(dL0)
TP = np.count_nonzero(dL1 >= gamma)
P = np.size(dL1)
FN = np.count_nonzero(dL1 < gamma)
plt.scatter(FP / N, TP / P, color='green')
print('Theoretical gamma: ')
print(gamma)
print('Theoretical min error: ')
print((FP / N)*priors[0] + (FN / P)*priors[1])

# empirically optimal
i = np.argmin(p_error)
plt.scatter(FPR[i], TPR[i], color='red')
print('Empirical gamma: ')
print(gammas[i])
print('Empirical min error: ')
print(p_error[i])


plt.title('ERM ROC Curve')
plt.legend(handles=[Line2D([0], [0], marker='o', color='w', label='Theoretical Gamma', markerfacecolor='g', markersize=10),
                    Line2D([0], [0], marker='o', color='w', label='Empirical Gamma', markerfacecolor='r', markersize=10)])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()
