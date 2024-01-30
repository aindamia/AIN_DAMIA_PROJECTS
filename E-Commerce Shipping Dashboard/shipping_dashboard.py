# Data manipulation library
import numpy as np
import pandas as pd
from scipy import stats

# Data visualization library
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Hide warnings
import warnings 
warnings.filterwarnings("ignore")

# Load your dataset
file_path = r'C:\Users\user\Desktop\Python\shipping.csv'
df = pd.read_csv(file_path)

# Display basic information about the dataset
print("Dataset Information:")
print(df.info())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values)

# Check for outliers using z-scores for all numerical variables
z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
outliers = np.where(z_scores > 3)

# Visualize outliers using box plots for all numerical variables
plt.figure(figsize=(15, 8))
sns.boxplot(data=df.select_dtypes(include=[np.number]), orient='h')
plt.title('Box Plots for All Numerical Variables')
plt.show()

# Replace values in the 'Reached.on.Time_Y.N.' column for better labels
df['Reached.on.Time_Y.N'] = df['Reached.on.Time_Y.N'].replace({0: 'On Time', 1: 'Not On Time'})

# Set the page
st.set_page_config(page_title="E-Commerce Shipping", page_icon=":package:")

# Set the dashboard title at the top of the page 
st.title("E-Commerce Shipping Insights: Descriptive Analysis")

# Use st.sidebar for filters
with st.sidebar:
    # Add title to the sidebar
    st.sidebar.title("Filters")
    
    # Target Variable Checkbox Filter
    st.sidebar.subheader("Status of Delivery")
    on_time_filter = st.checkbox("On Time", value=True)
    not_on_time_filter = st.checkbox("Not On Time", value=True)
    
    # Discount Range Slider
    st.sidebar.subheader("Discount Range")
    discount_range = st.slider("Select Discount Range", min_value=0, max_value=65, value=(0, 10))

    # Weight Range Slider
    st.sidebar.subheader("Weight Range")
    weight_range = st.slider("Select Weight Range", min_value=df['Weight_in_gms'].min(), max_value=df['Weight_in_gms'].max(), value=(df['Weight_in_gms'].min(), df['Weight_in_gms'].max()))

# Apply Filters
if on_time_filter and not_on_time_filter:
    # Include both On Time and Not On Time
    filtered_df = df
elif on_time_filter:
    # Include only On Time
    filtered_df = df[df['Reached.on.Time_Y.N'] == 'On Time']
elif not_on_time_filter:
    # Include only Not On Time
    filtered_df = df[df['Reached.on.Time_Y.N'] == 'Not On Time']
else:
    # Exclude both On Time and Not On Time
    filtered_df = pd.DataFrame()

# Apply Discount Range Filter
filtered_df = filtered_df[(filtered_df['Discount_offered'] >= discount_range[0]) & (filtered_df['Discount_offered'] <= discount_range[1])]

# Apply Weight Range Filter
filtered_df = filtered_df[(filtered_df['Weight_in_gms'] >= weight_range[0]) & (filtered_df['Weight_in_gms'] <= weight_range[1])]


# Warehouse Analysis
# ==============================
st.header("1. Warehouse Analysis")

# Chart 1: Warehouse Block 
st.markdown("<h3 style='text-align: center;'>Warehouse Block</h3>", unsafe_allow_html=True)
fig, ax = plt.subplots()
warehouse_order = ['A', 'B', 'C', 'D', 'E', 'F']
sns.countplot(x='Warehouse_block', hue='Reached.on.Time_Y.N', data=filtered_df, ax=ax, palette=['red', 'green'], order=warehouse_order)
ax.set_xlabel('Warehouse Block')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# Conclusion for Warehouse Analysis
st.markdown("""
**Conclusion for Warehouse Analysis:**
- Most product comes from warehouse F
- All warehouse block shows that most of the product do not arrive on time
- There is no significant influence between warehouse block and status of delivery
""")


# Shipment Analysis
# ==============================
st.header("2. Shipment Analysis")

## Chart 2: Mode of Shipment
st.markdown("<h3 style='text-align: center;'>Mode of Shipment</h3>", unsafe_allow_html=True)
fig, ax = plt.subplots()
sns.countplot(x='Mode_of_Shipment', hue='Reached.on.Time_Y.N', data=filtered_df, ax=ax, palette=['red', 'green'])
ax.set_xlabel('Mode of Shipment')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# Conclusion for Shipment Analysis
st.markdown("""
**Conclusion for Shipment Analysis:**
- Most product are delivered using ship
- All mode of shipment shows that most of the product do not arrive on time
- There is no significant influence between mode of shipment and status of delivery
""")


# Customer Analysis
# ==============================
st.header("3. Customer Analysis")

# Create two columns for side-by-side charts
col1, col2 = st.columns(2)

## Chart 3: Customer Ratings
col1.markdown("<h3 style='text-align: center;'>Customer Ratings</h3>", unsafe_allow_html=True)
fig, ax = plt.subplots()
sns.countplot(x='Customer_rating', hue='Reached.on.Time_Y.N', data=filtered_df, ax=ax, palette=['red', 'green'])
ax.set_xlabel('Customer Rating')
ax.set_ylabel('Frequency')
col1.pyplot(fig)

## Chart 4: Customer Care Calls
col2.markdown("<h3 style='text-align: center;'>Customer Care Calls</h3>", unsafe_allow_html=True)
fig, ax = plt.subplots()
sns.countplot(x='Customer_care_calls', hue='Reached.on.Time_Y.N', data=filtered_df, ax=ax, palette=['red', 'green'])
ax.set_xlabel('Customer Care Calls')
ax.set_ylabel('Frequency')
col2.pyplot(fig)

# Conclusion for Customer Analysis
st.markdown("""
**Conclusion for Customer Analysis:**
- **Customer Ratings**
  - All customer rating shows that most of the product do not arrive on time
  - There is no significant influence between customer ratings and status of delivery
- **Customer Care Calls**
  - Most product have 2 to 3 number of customer care calls
  - Every number of customer care call shows that most of the product do not arrive on time
  - There is no significant influence between customer care calls and status of delivery
""")


# Product Analysis
# ==============================
st.header("4. Product Analysis")

# Create two columns for side-by-side charts
col3, col4 = st.columns(2)

## Chart 5: Cost of the Product
col3.markdown("<h3 style='text-align: center;'>Cost of the Product</h3>", unsafe_allow_html=True)
fig, ax = plt.subplots()
sns.boxplot(x='Reached.on.Time_Y.N', y='Cost_of_the_Product', data=filtered_df, ax=ax, palette=['red', 'green'])
ax.set_xlabel('Reached on Time')
ax.set_ylabel('Cost of the Product')
col3.pyplot(fig)

## Chart 6: Prior Purchases
col4.markdown("<h3 style='text-align: center;'>Prior Purchases</h3>", unsafe_allow_html=True)
fig, ax = plt.subplots()
sns.countplot(x='Prior_purchases', hue='Reached.on.Time_Y.N', data=filtered_df, ax=ax, palette=['red', 'green'])
ax.set_xlabel('Prior Purchases')
ax.set_ylabel('Frequency')
col4.pyplot(fig)

# Create two columns for side-by-side charts
col5, col6 = st.columns(2)

## Chart 7: Product Importance
col5.markdown("<h3 style='text-align: center;'>Product Importance</h3>", unsafe_allow_html=True)
fig, ax = plt.subplots()
sns.countplot(x='Product_importance', hue='Reached.on.Time_Y.N', data=filtered_df, ax=ax, palette=['red', 'green'])
ax.set_xlabel('Product Importance')
ax.set_ylabel('Frequency')
col5.pyplot(fig)

## Chart 8: Discount Offered
col6.markdown("<h3 style='text-align: center;'>Discount Offered</h3>", unsafe_allow_html=True)
fig, ax = plt.subplots()
sns.boxplot(x='Reached.on.Time_Y.N', y='Discount_offered', data=filtered_df, ax=ax, palette=['red', 'green'])
ax.set_xlabel('Reached on Time')
ax.set_ylabel('Discount Offered')
col6.pyplot(fig)

## Chart 9: Weight In Grams
st.markdown("<h3 style='text-align: center;'>Weight of Product (gram)</h3>", unsafe_allow_html=True)
fig, ax = plt.subplots()
sns.boxplot(x='Reached.on.Time_Y.N', y='Weight_in_gms', data=filtered_df, ax=ax, palette=['red', 'green'])
ax.set_xlabel('Reached on Time')
ax.set_ylabel('Weight of Product (gram)')
st.pyplot(fig)

# Conclusion for Product Analysis
st.markdown("""
**Conclusion for Product Analysis:**
- **Cost of the Product**
  - There is no significant influence between cost of the product and status of delivery
- **Prior Purchases**
  - Most customers who have made prior purchase mainly 2-3 times, tend to receive product not on time
  - There is no significant influence between prior purchases and status of delivery
- **Product Importance**
  - All level of product importance shows that more product do not arrive on time
  - There is no significant influence between product importance and status of delivery
- **Discount Offered**
  - Product with discounts ranging from 0-10% tend to arrive on time
  - Product with discounts above 10% are more likely to not arrive on time
  - There is significant influence between discount offered and status of delivery
- **Weight of Product (gram)**
  - Product weighing below about 4000 grams tend not to not arrive on time
  - Product weighing above 4000 grams tend to arrive on time
  - There is significant influence between weight of product and status of delivery       
""")
