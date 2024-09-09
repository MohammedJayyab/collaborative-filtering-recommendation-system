# Collaborative Filtering Recommendation System Using SVD

## Table of Contents

1. [Project Overview](#overview)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Steps to Run](#steps-to-run)
5. [Modeling and Predictions](#modeling)
6. [Evaluation Metrics](#evaluation)
7. [Results](#results)
8. [Future Improvements](#improvements)

## 1. Project Overview <a name="overview"></a>

This project demonstrates the use of **Collaborative Filtering** with **Singular Value Decomposition (SVD)** for building a recommendation system. The goal of this project is to provide personalized product recommendations to customers based on their historical interaction data (e.g., purchases). The system predicts which products a customer might like or buy next by identifying patterns in the customer-product interaction data.

The dataset used in this project consists of user transactions from an online retail store, and the model is trained to recommend products to customers based on past purchases.

## 2. Installation <a name="installation"></a>

To set up the environment for this project, follow these steps:

### Step 1: Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install the required packages

Install the necessary Python libraries by running:

```bash
pip install -r requirements.txt
```

The required packages include:

- pandas
- numpy
- scikit-learn
- openpyxl

## 3. Data Preparation <a name="data-preparation"></a>

Data preparation is an essential step before building the recommendation system. The steps involved include:

- **Loading the Dataset**: The raw dataset contains customer transactions, including the product IDs, product descriptions, and customer IDs.
- **Creating the Interaction Matrix**: We create a matrix where rows represent customers and columns represent products. The matrix cells contain the interaction count, such as the number of times a customer bought a specific product.

```python
M = np.array([
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 1]
])
```

The interaction matrix will be used to train the recommendation model.

## 4. Steps to Run <a name="steps-to-run"></a>

To execute the recommendation system, follow these steps:

### Step 1: Prepare the Data

Run the `data_processing.py` script to clean and prepare the dataset:

```bash
python data_processing.py
```

This script will load the raw dataset, filter out irrelevant transactions, and create the interaction matrix (`data/interaction_matrix.pkl`).

### Step 2: Build the Collaborative Filtering Model

Run the `collaborative_filtering_model.py` to build the recommendation model using SVD:

```bash
python collaborative_filtering_model.py
```

This script will train the SVD model and save the trained model (`models/svd_model.pkl`) and latent matrix (`models/latent_matrix.pkl`).

### Step 3: Make Predictions

Run the `make_predictions.py` to generate product recommendations for specific users:

```bash
python make_predictions.py
```

This will recommend products for users based on the learned latent factors.

### Step 4: Evaluate the Model

Run the `evaluate_model.py` to evaluate the performance of the recommendation system using metrics like MSE, Precision@K, Recall@K, and F1-Score:

```bash
python evaluate_model.py
```

## 5. Modeling and Predictions <a name="modeling"></a>

We use **Singular Value Decomposition (SVD)** to decompose the interaction matrix into three matrices: \( U \), \( \Sigma \), and \( V^T \). This allows us to capture the latent relationships between customers and products.

- **Matrix U**: Contains latent factors representing customers.
- **Matrix \( \Sigma \)**: Diagonal matrix representing the importance of each latent factor.
- **Matrix V^T**: Contains latent factors representing products.

We then reconstruct the interaction matrix to predict how much a customer will like a product they haven't purchased before.

### Prediction Example

For a customer \( C_1 \) and product \( P_3 \), the predicted interaction score is calculated by taking the dot product between the customer’s latent features and the product’s latent features.

## 6. Evaluation Metrics <a name="evaluation"></a>

We evaluate the recommendation system using various metrics:

- **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted values.
- **Root Mean Squared Error (RMSE)**: Provides the error in the original scale of the interaction data.
- **Mean Absolute Error (MAE)**: Measures the average magnitude of the error between actual and predicted values.
- **Precision@K**: Measures how many of the top K recommended products were relevant.
- **Recall@K**: Measures how many of the relevant products were recommended in the top K.
- **F1-Score**: A harmonic mean of Precision and Recall, balancing both metrics.

### Example Results:

- **MSE**: 4.98
- **RMSE**: 2.23
- **MAE**: 0.29
- **Precision@5**: 16.53%
- **Recall@5**: 5.68%
- **F1-Score@5**: 8.46%

## 7. Results <a name="results"></a>

The recommendation system suggests personalized products for each customer. For example:

- **Customer 17850**: Recommended products:
  - ASSORTED COLOUR BIRD ORNAMENT
  - WHITE HANGING HEART T-LIGHT HOLDER

These recommendations are based on past purchase patterns and the latent relationships identified through the SVD model.

## 8. Future Improvements <a name="improvements"></a>

Several improvements can be made to enhance the performance of the recommendation system:

1. **Tuning the Number of Latent Factors**: Adjust the number of latent factors to capture more complex user-product relationships.
2. **Alternative Algorithms**: Experiment with **Alternating Least Squares (ALS)** or **Matrix Factorization with Bias** for better performance.
3. **Hybrid Models**: Consider combining content-based filtering with collaborative filtering using models like **LightFM**.
4. **Increasing K**: Evaluate using Precision@10, Recall@10, or higher to explore different recommendation scenarios.

---

This project demonstrates the application of collaborative filtering with SVD to build a personalized recommendation system that leverages customer-product interaction data to generate meaningful suggestions. The model can be further enhanced through parameter tuning, improved algorithms, or hybrid approaches.
