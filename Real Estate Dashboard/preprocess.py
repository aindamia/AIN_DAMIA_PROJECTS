# Import the required libraries
import os
import pandas as pd

# Define the directory and file name
directory = 'C:\\Users\\aiman\\Documents\\SQIT3073\\PyClass\\PROJECT'
file_name = 'houses.csv'  # Adjust the file name with the correct one

# File Path
file_path = os.path.join(directory, file_name)

# Try to read the CSV file
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")

# Data Transforming
df['price'] = df['price'].str.replace('RM', '').str.replace(' ', '').astype(float)
df['Property Size'] = df['Property Size'].str.replace('sq.ft.', '').str.replace(' ', '').astype(float)

# Process the 'Address' column
df['Address'] = df['Address'].str.split(',').str[-1]

# Rename 'price' to 'Price' and 'Address' to 'State'
df.rename(columns={'price': 'Price', 'Address': 'State'}, inplace=True)

# Process the 'Facilities' column
facilities = df['Facilities'].str.get_dummies(sep=', ')

# Concatenate the original DataFrame with the new 'facilities' DataFrame
df = pd.concat([df, facilities], axis=1)

# Drop unnecessary columns
df.drop(columns=['description', 'Ad List', '-', '10','Completion Year','# of Floors','Total Units','Floor Range','Facilities','Building Name','Developer','Category','Nearby School','Nearby Mall','Hospital','Bus Stop','Mall','Park','School','Highway','Nearby Railway Station','Railway Station','Firm Type','Firm Number','REN Number'], inplace=True)

# Write DataFrame to an Excel file
try:
    df.to_excel('output.xlsx', index=False)
    print("Excel file created successfully.")
except Exception as e:
    print(f"An error occurred while creating the Excel file: {e}")

# Print the DataFrame
print(df)
