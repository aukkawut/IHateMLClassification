import numpy as np
from scipy.stats import f
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

digits = load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
cmeans = {}
cprecisions = {}
r = 1e-5

for c in np.unique(y_train):
    class_data = X_train[y_train == c]
    cmeans[c] = np.mean(class_data, axis=0)
    cov = np.cov(class_data, rowvar=False) + np.eye(class_data.shape[1]) * r
    cprecisions[c] = np.linalg.inv(cov)

nc = {c: len(X_train[y_train == c]) for c in np.unique(y_train)}
pc = X_train.shape[1]

def p_compute(x, cmeans, cprecisions, nc, pc, tstat = False):
    classes = list(cmeans.keys())
    p_values = []
    t2_stats = []
    for c in classes:
        mean = cmeans[c]
        inv_cov = cprecisions[c]
        diff = x - mean
        t2_stat = np.dot(diff, np.dot(inv_cov, diff))
        n = nc[c]
        F = (n - pc) / (pc * n) * t2_stat
        p_value = 1 - f.cdf(F, pc, n-pc)
        p_values.append(p_value)
        t2_stats.append((c,t2_stat))
    if tstat:
        return p_values, t2_stats
    return p_values

def classify(x, cmeans, cprecisions, nc, pc, sig=0.05, verbose = False):
    p_values = p_compute(x, cmeans, cprecisions, nc, pc)
    predicted_class = np.argmax(p_values)
    if all(p < sig for p in p_values):
        return -1  # "unknown" class
    return predicted_class

predicted_classes = [classify(p, cmeans, cprecisions, nc, pc) for p in X_test]
accuracy = np.mean(predicted_classes == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

def visualize(i, X_test, y_test, cmeans, cprecisions, nc, pc):
    x = X_test[i]
    true_label = y_test[i]
    p_values, t_stats = p_compute(x, cmeans, cprecisions, nc, pc, tstat = True)
    print(t_stats)
    predicted_class = np.argmax(p_values)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(x.reshape(8, 8), cmap='gray')
    plt.title(f"True label: {true_label}")
    plt.subplot(1, 2, 2)
    classes = np.arange(10)
    colors = ['red' if c == predicted_class else 'blue' for c in classes]
    plt.bar(classes, p_values, color=colors, tick_label=classes)
    plt.ylim(0, 1)
    plt.xlabel("Class")
    plt.ylabel("p-value")
    plt.title("p-values for each class")
    plt.tight_layout()
    plt.show()
i = np.random.randint(0, X_test.shape[0])
visualize(i, X_test, y_test, cmeans, cprecisions, nc, pc)


# now, try on iris dataset

from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)
cmeans = {}
cprecisions = {}
r = 1e-5

for c in np.unique(y_train):
    class_data = X_train[y_train == c]
    cmeans[c] = np.mean(class_data, axis=0)
    cov = np.cov(class_data, rowvar=False) + np.eye(class_data.shape[1]) * r
    cprecisions[c] = np.linalg.inv(cov)

nc = {c: len(X_train[y_train == c]) for c in np.unique(y_train)}
pc = X_train.shape[1]

predicted_classes = [classify(p, cmeans, cprecisions, nc, pc) for p in X_test]
accuracy = np.mean(predicted_classes == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# how about wisconsin breast cancer dataset?

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)
cmeans = {}
cprecisions = {}
r = 1e-5

for c in np.unique(y_train):
    class_data = X_train[y_train == c]
    cmeans[c] = np.mean(class_data, axis=0)
    cov = np.cov(class_data, rowvar=False) + np.eye(class_data.shape[1]) * r
    cprecisions[c] = np.linalg.inv(cov)

nc = {c: len(X_train[y_train == c]) for c in np.unique(y_train)}
pc = X_train.shape[1]

predicted_classes = [classify(p, cmeans, cprecisions, nc, pc) for p in X_test]
accuracy = np.mean(predicted_classes == y_test)

print(f"Accuracy: {accuracy * 100:.2f}%")

