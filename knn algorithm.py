# Description: K-Nearest Neighbors (K-NN) Algorithm

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import mode
from sklearn.preprocessing import StandardScaler

# Importing the dataset
dataset = pd.ExcelFile("Need_For_Speed_Unbound_Dataset.xlsx")
train = pd.read_excel(dataset, 'Train')
test = pd.read_excel(dataset, 'Test')

# Data Visualization
sns.set(style="ticks", color_codes=True)

# Scatterplot
g = sns.PairGrid(train, vars=['Horsepower', 'Top Speed', 'Overpowered'],
                 hue='Car Brand', palette='Paired')
g.map(plt.scatter, alpha=0.8)
g.add_legend()

# Boxplot
sns.boxplot(data=train[["Horsepower", "Top Speed"]], palette="Paired")

# Splitting the dataset into the Training set and Test set
x_train = train[["Horsepower", "Top Speed"]]
y_train = train["Overpowered"]

x_test = test[["Horsepower", "Top Speed"]]

# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
print(x_train, x_test)

# Euclidean Distance


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# K-Nearest Neighbors (K-NN) Algorithm


def predict(x_train, y, x_input, k):
    op_labels = []

    # Loop through the Datapoints to be classified
    for item in x_input:

        # Array to store distances
        point_dist = []

        # Loop through each training Data
        for j in range(len(x_train)):
            distances = euclidean_distance(np.array(x_train[j, :]), item)
            # Calculating the distance
            point_dist.append(distances)
        point_dist = np.array(point_dist)

        # Sorting the array while preserving the index
        # Keeping the first K datapoints
        dist = np.argsort(point_dist)[:k]

        # Labels of the K datapoints from above
        labels = y[dist]

        # Majority voting
        lab = mode(labels)
        lab = lab.mode[0]
        op_labels.append(lab)

    return op_labels


# Ignore the warnings
warnings.filterwarnings("ignore")

# Predicting the Test set results


def main():
    k = 3
    while k <= 21:
        y_pred = predict(x_train, y_train, x_test, k)
        print("K = ", k)
        print("Prediksi: ", y_pred, "\n")
        k += 2


if __name__ == "__main__":
    main()
