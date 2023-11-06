#   now we consider the case where an image is flatten and each entry of that matrix is a dependent data point
#   drawn from the class distribution D.

#   We can use the t-test to compare the mean of the class distribution D to the mean of the test image.

#   Now, this method will not work well at all, as most of the time, the mean of the data is close to four to five.
#   Because it is just a linear transformation of the data, its mean does not change at all.

import numpy as np
from scipy.stats import t
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

digits = load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# Precalculating class mean and standard deviation for each class down to one scalar (for each class)
means, stds = [],[]
for i in range(10):
    class_data = X_train[y_train == i]
    # first, flatten down the train data from n*64*1 to 64n dimensional vector
    class_data = np.concatenate(class_data).ravel()
    # then calculate the mean and standard deviation
    means.append(np.mean(class_data[np.nonzero(class_data)]))
    stds.append(np.std(class_data[np.nonzero(class_data)]))
# Draw a test image
i = np.random.randint(len(X_test))
x = X_test[i]
true_label = y_test[i]

# Calculate the t-statistic for each class
t_statistics = []
for i in range(10):
    mean = means[i]
    std = stds[i]
    t_statistic = (np.mean(x[np.nonzero(x)]) - mean) / std
    t_statistics.append(t_statistic)
    pval = 1 - t.cdf(t_statistic, df=64)
    print(f"Class {i}: t-statistic: {t_statistic}, p-value: {pval}")

print(f"True label: {true_label}")