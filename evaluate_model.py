import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Function to evaluate MSE, RMSE, and MAE
def evaluate_model(interaction_matrix, svd_model, latent_matrix):
    # Reconstruct the interaction matrix from SVD
    reconstructed_matrix = np.dot(latent_matrix, svd_model.components_)
    
    # Flatten both matrices for comparison
    actual = interaction_matrix.values.flatten()
    predicted = reconstructed_matrix.flatten()
    
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(actual, predicted)
    
    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    
    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(actual, predicted)
    
    return mse, rmse, mae

# Function to calculate Precision@K
def precision_at_k(reconstructed_matrix, actual_matrix, k):
    precision_scores = []
    
    for user_idx in range(reconstructed_matrix.shape[0]):
        # Get top K predicted items
        top_k_items = np.argsort(-reconstructed_matrix[user_idx, :])[:k]
        
        # Get top K actual items
        actual_items = np.argsort(-actual_matrix[user_idx, :])[:k]
        
        # Calculate precision for the user
        relevant_and_recommended = np.intersect1d(top_k_items, actual_items)
        precision = len(relevant_and_recommended) / k
        precision_scores.append(precision)
    
    return np.mean(precision_scores)

# Function to calculate Recall@K
def recall_at_k(reconstructed_matrix, actual_matrix, k):
    recall_scores = []
    
    for user_idx in range(reconstructed_matrix.shape[0]):
        # Get top K predicted items
        top_k_items = np.argsort(-reconstructed_matrix[user_idx, :])[:k]
        
        # Get all actual relevant items (non-zero entries in the interaction matrix)
        actual_items = np.where(actual_matrix[user_idx, :] > 0)[0]
        
        if len(actual_items) == 0:
            continue
        
        # Calculate recall for the user
        relevant_and_recommended = np.intersect1d(top_k_items, actual_items)
        recall = len(relevant_and_recommended) / len(actual_items)
        recall_scores.append(recall)
    
    return np.mean(recall_scores)

# Function to calculate F1-Score
def f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

if __name__ == '__main__':
    # Load the interaction matrix, SVD model, and latent matrix
    interaction_matrix = pd.read_pickle('data/interaction_matrix.pkl')
    svd_model = pd.read_pickle('models/svd_model.pkl')
    latent_matrix = pd.read_pickle('models/latent_matrix.pkl')
    
    # Evaluate the model using multiple metrics
    mse, rmse, mae = evaluate_model(interaction_matrix, svd_model, latent_matrix)
    print(f"Model Mean Squared Error: {mse}")
    print(f"Model Root Mean Squared Error: {rmse}")
    print(f"Model Mean Absolute Error: {mae}")
    
    # Reconstruct the interaction matrix from latent factors
    reconstructed_matrix = np.dot(latent_matrix, svd_model.components_)
    
    # Calculate Precision@2 and Recall@2
    precision_k = precision_at_k(reconstructed_matrix, interaction_matrix.values, k=5)
    recall_k = recall_at_k(reconstructed_matrix, interaction_matrix.values, k=5)
    
    # Calculate F1-Score
    f1_k = f1_score(precision_k, recall_k)
    
    print(f"Precision@5: {precision_k:.4f}")
    print(f"Recall@5: {recall_k:.4f}")
    print(f"F1-Score@5: {f1_k:.4f}")
