import pandas as pd
import numpy as np

def recommend_products(user_id, interaction_matrix, svd_model, latent_matrix, n_recommendations=5):
    # Reconstruct the original interaction matrix using the SVD model
    reconstructed_matrix = np.dot(latent_matrix, svd_model.components_)
    
    # Get the user’s interaction vector (row corresponding to the user)
    user_index = interaction_matrix.index.get_loc(user_id)
    user_interactions = reconstructed_matrix[user_index]
    
    # Get top N recommendations
    product_indices = np.argsort(-user_interactions)[:n_recommendations]
    recommended_products = interaction_matrix.columns[product_indices]
    
    return recommended_products

if __name__ == '__main__':
    # Load the interaction matrix, SVD model, and latent matrix
    interaction_matrix = pd.read_pickle('data/interaction_matrix.pkl')
    svd_model = pd.read_pickle('models/svd_model.pkl')
    latent_matrix = pd.read_pickle('models/latent_matrix.pkl')
    
    # Specify a user for whom to make recommendations
    user_id = 17850
    
    # Get recommendations
    recommendations = recommend_products(user_id, interaction_matrix, svd_model, latent_matrix)
    
    print(f"Recommended products for User {user_id}:")
    for product in recommendations:
        print(f"- {product}")
