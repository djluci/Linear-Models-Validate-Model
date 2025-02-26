import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import itertools
from sklearn.metrics import classification_report

# Function to generate interaction feature names
def generate_interaction_feature_names(original_feature_names):
    interaction_feature_names = []
    for i, j in itertools.combinations(original_feature_names, 2):
        interaction_feature_names.append(f"{i}*{j}")
    return original_feature_names + interaction_feature_names

def plot_model_weights(model, feature_names):
    # Check if  model has the 'coef_' attribute (scikit-learn model)
    if hasattr(model, 'coef_'):
        weights = model.coef_.flatten()
    elif hasattr(model, 'weights'):
        weights = model.weights.flatten()
    else:
        raise AttributeError("The model does not have 'coef_' or 'weights' attributes")
    
    # Add interaction feature names
    if model.interactions:
        interaction_names = []
        for i, j in itertools.combinations(feature_names, 2):  # Pairwise combinations of features
            interaction_names.append(f"{i}*{j}")
        all_feature_names = feature_names + interaction_names
    else:
        all_feature_names = feature_names

    # Bar plot
    plt.figure(figsize=(12, 8))
    plt.bar(np.arange(len(weights)), weights, tick_label=all_feature_names)

    # Title & labels
    plt.title("Model Weights", fontsize=16)
    plt.ylabel("Feature Weight", fontsize=12)

    # Feature names on x-axis are rotated
    plt.xticks(rotation=90, ha='right', fontsize=10)

    # Padding to avoid label overlap
    plt.subplots_adjust(bottom=0.3)

    # Show plot
    plt.show()
    
def print_model_evaluation(model, x_train, y_train, x_test, y_test):
    # Printing the fraction of positive labels in the dataset using numpy mean
    print(f"Fraction of positive labels in train set: {np.mean(y_train == 1):.2f}")
    print(f"Fraction of positive labels in test set: {np.mean(y_test == 1):.2f}")
    
    # Accuracy of the model using scikit-learn function
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.2f}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.2f}")
    
    # AUC score of model
    if hasattr(model, 'predict_proba'):
        y_train_prob = model.predict_proba(x_train)
        y_test_prob = model.predict_proba(x_test)
        
        # Checking if output is 2D and if so, index [:, 1] for positive class
        if len(y_train_prob.shape) == 2:
            y_train_prob = y_train_prob[:, 1]
            y_test_prob = y_test_prob[:, 1]
            
        # Calculating and printing AUC scores
        train_auc = roc_auc_score(y_train, y_train_prob)
        test_auc = roc_auc_score(y_test, y_test_prob)
        print(f"Train AUC: {train_auc:.2f}")
        print(f"Test AUC: {test_auc:.2f}")
        
        # Returning the test AUC for comparison
        return test_auc
    
    else:
        print("Model doesn't allow probability predictions")
        return None