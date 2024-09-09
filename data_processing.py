import pandas as pd

def load_and_prepare_data(input_file):
    # Load the Excel data into a DataFrame
    all_sheets = pd.read_excel(input_file, sheet_name=None)
    data = pd.concat(all_sheets.values(), ignore_index=True)

    # Filter out canceled transactions
    data = data[~data['Invoice'].astype(str).str.startswith('C', na=False)]

    # Create User-Item Interaction Matrix (Implicit Feedback)
    interaction_matrix = data.pivot_table(index='Customer ID', columns='Description', values='Quantity', fill_value=0)
    
    return interaction_matrix

if __name__ == '__main__':
    input_file = 'data/online_retail_II.xlsx'
    interaction_matrix = load_and_prepare_data(input_file)
    
    # Save the prepared data for use in the recommendation model
    interaction_matrix.to_pickle('data/interaction_matrix.pkl')
    print("Interaction matrix saved to 'data/interaction_matrix.pkl'")
