import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from collections import Counter

matplotlib.use('TkAgg')

# Load the datasets (update file paths as necessary)
X_train = np.load('cucumbers_X_train.npy')
Y_train = np.load('cucumbers_Y_train.npy')
X_test = np.load('cucumbers_X_test.npy')
Y_test = np.load('cucumbers_Y_test.npy')


# Visualizing distributions of classes in 2D feature planes
def plot_feature_distribution(X_train, Y_train):
    feature_names = ["Bending Angle", "Weight", "Diameter", "Color"]
    plt.figure(figsize=(15, 10))
    plot_idx = 1
    for i in range(4):
        for j in range(i + 1, 4):
            plt.subplot(3, 2, plot_idx)
            for label in np.unique(Y_train):
                plt.scatter(X_train[Y_train == label, i], X_train[Y_train == label, j], label=f"Class {label}",
                            alpha=0.6)
            plt.xlabel(feature_names[i])
            plt.ylabel(feature_names[j])
            plt.legend()
            plt.title(f"{feature_names[i]} vs {feature_names[j]}")
            plot_idx += 1
    plt.tight_layout()
    plt.show()


plot_feature_distribution(X_train, Y_train)

# Standardizing the features
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)


# Function to find the K nearest neighbors using L2 norm
def find_k_nearest_neighbors(X_train, y_train, x_test, k):
    distances = distance.cdist([x_test], X_train, 'euclidean')[0]
    nearest_neighbors_indices = np.argsort(distances)[:k]
    nearest_neighbors_labels = y_train[nearest_neighbors_indices]
    return nearest_neighbors_labels


# Function to predict class based on majority vote
def predict_class(nearest_neighbors_labels):
    label_count = Counter(nearest_neighbors_labels)
    return label_count.most_common(1)[0][0]


# Function to compute accuracy of predictions
def compute_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    accuracy = correct_predictions / len(y_true)
    return accuracy


# Testing KNN model for K values from 1 to 10 and plotting accuracy vs K
accuracies = []
for k in range(1, 11):
    predictions = []
    for x_test in X_test_std:
        nearest_neighbors = find_k_nearest_neighbors(X_train_std, Y_train, x_test, k)
        predicted_label = predict_class(nearest_neighbors)
        predictions.append(predicted_label)
    accuracy = compute_accuracy(Y_test, np.array(predictions))
    accuracies.append(accuracy)

# Plotting accuracy vs K
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), accuracies, marker='o')
plt.xlabel("K (Number of Nearest Neighbors)")
plt.ylabel("Model Accuracy")
plt.title("KNN Model Accuracy vs K")
plt.grid(True)
plt.show()