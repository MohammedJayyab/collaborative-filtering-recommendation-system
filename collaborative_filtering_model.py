import pandas as pd
from sklearn.decomposition import TruncatedSVD

def build_svd_model(interaction_matrix, n_components=50):
    # Initialize the SVD model
    svd = TruncatedSVD(n_components=n_components)
    
    # Fit the model to the interaction matrix
    latent_matrix = svd.fit_transform(interaction_matrix)
    
    return svd, latent_matrix

if __name__ == '__main__':
    # Load the interaction matrix
    interaction_matrix = pd.read_pickle('data/interaction_matrix.pkl')
    
    # Build the SVD model
    svd_model, latent_matrix = build_svd_model(interaction_matrix)
    
    # Save the model and latent features
    pd.to_pickle(svd_model, 'models/svd_model.pkl')
    pd.to_pickle(latent_matrix, 'models/latent_matrix.pkl')
    
    print("SVD model and latent matrix saved.")
