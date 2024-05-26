import pandas as pd
import numpy as np

# Example DataFrame
data = {
    'list of company': [
        ['Apple', 'Google', 'Facebook'],
        ['Microsoft', 'Apple'],
        ['Google', 'Amazon']
    ]
}
df = pd.DataFrame(data)

# Example 1D array
companies = np.array(['Apple', 'Google', 'Facebook', 'Amazon', 'Microsoft'])

# Convert the array into a Series where index is company name and value is the index in the original array
company_index = pd.Series(index=companies, data=np.arange(len(companies)))

# Function to convert company names to indices
def map_to_index(company_list):
    return [company_index[company] for company in company_list]

# Apply the function to the DataFrame column
df['indexed_companies'] = df['list of company'].apply(map_to_index)

print(df)