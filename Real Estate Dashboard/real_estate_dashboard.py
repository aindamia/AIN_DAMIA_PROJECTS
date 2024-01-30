# Importing necessary libraries
import streamlit as st
import plotly.express as px
import pandas as pd
import os
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from PIL import Image

# Streamlit settings and warnings configuration
st.set_option('deprecation.showPyplotGlobalUse', False)
warnings.filterwarnings('ignore')

# Streamlit page configuration
st.set_page_config(page_title="🏠 MYHomes: Your Trusted Partner in Malaysian Real Estate", page_icon="🏠", layout="wide")

# Title and introductory markdown
st.title("🏠 MYHomes: Your Trusted Partner in Malaysian Real Estate")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

# Reading the dataset
df = pd.read_excel(r"C:\\Users\\user\\Desktop\\Python\\Output.xlsx")

# Identify columns with missing values
columns_with_missing_values = df.columns[df.isnull().any()].tolist()

# Input missing values for numeric columns
numeric_cols_with_missing = df[columns_with_missing_values].select_dtypes(include=np.number).columns
df[numeric_cols_with_missing] = df[numeric_cols_with_missing].fillna(df[numeric_cols_with_missing].mean())

# Round up the inputed values to the nearest whole number
df[numeric_cols_with_missing] = df[numeric_cols_with_missing].round(0)

# Sidebar for filtering data
st.sidebar.header("🔍 Filter Here:")
state = st.sidebar.multiselect("Select the State", df["State"].unique())
if not state:
    df2 = df.copy()
else:
    df2 = df[df["State"].isin(state)]

property_type=st.sidebar.multiselect("Choose the Property Type",df2["Property Type"].unique())
if not property_type:
    df3 = df2.copy()
else:
    df3=df2[df2["Property Type"].isin(property_type)]

# Sidebar for page selection
st.sidebar.header("📄 Pages")
page = st.sidebar.selectbox("Choose a page", ["🏢 Home Page", "📊 Exploratory Data Analysis", "📈 Real Estate Price Estimator"])

# Home Page
if page == "🏢 Home Page":
    st.title("Discover MYHomes")

    st.header("📝 About Us")
    st.write("""
    Welcome to MYHomes, your trusted partner in the Malaysian real estate landscape. As a Malaysia-based online platform, MYHomes is dedicated to simplifying the home buying and selling experience for individuals and families. Our commitment to innovation and customer satisfaction sets us apart in the dynamic world of real estate.
    """)

    st.header("🎯 Vision")
    st.write("""
    To be the leading online destination for Malaysians in search of their dream homes, offering a seamless and empowering real estate experience.
    """)

    st.header("🚀 Mission")
    st.write("""
    MYHomes is on a mission to revolutionize the Malaysian real estate industry by providing a user-centric platform that simplifies the home buying and selling process. We aim to empower our users with information, cutting-edge technology, and personalized services, ensuring they make informed decisions and find homes that perfectly match their dreams.
    """)

    st.header("🎯 Objectives")
    st.write("""
    - **User-Centric Experience:** Enhance our platform continually to provide an intuitive and user-friendly experience for both buyers and sellers in the Malaysian real estate market.
    - **Comprehensive Listings:** Maintain an extensive and up-to-date database of property listings, ensuring our Malaysian users have access to a diverse range of homes.
    - **Innovation:** Embrace technological advancements to stay ahead in the Malaysian real estate industry, offering innovative tools and features that enhance the overall real estate experience.
    """)

    st.write("""
    MYHomes is not just an online platform; it's a dedicated partner for Malaysians on the exciting journey of finding their perfect home. Join us as we redefine the real estate experience in Malaysia!
    """)

    image = Image.open('C:\\Users\\user\\Desktop\\Python\\house\\Real-Estate-2.png') 
    st.image(image)

# Exploratory Data Analysis Page
elif page == "📊 Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    # Bar Chart: Average Price vs State
    with col1:
        st.subheader("📊 Bar Chart: Average Price vs State")
        avg_price_state = df3.groupby('State')['Price'].mean().reset_index()
        fig = px.bar(avg_price_state, x='State', y='Price', title='Average Price by State', color='Price')
        st.plotly_chart(fig)

    # Donut Chart: Count of Property Type
    with col2:
        st.subheader("🍩 Donut Chart: Count of Property Type")
        property_type_count = df3['Property Type'].value_counts().reset_index()
        property_type_count.columns = ['Property Type', 'Count']
        fig = px.pie(property_type_count, values='Count', names='Property Type', title='Count of Property Type', hole=0.4)
        st.plotly_chart(fig)
    
    col3, col4 = st.columns(2)

    # Scatter Plot: Price vs Property Size
    with col3:
        st.subheader("🔵 Scatter Chart: Price vs Property Size")
        fig = px.scatter(df3, x='Property Size', y='Price', title='Price vs Property Size', color='Price')
        st.plotly_chart(fig)


    # Bar Chart: Distribution of Bedroom Counts
    with col4:
        st.subheader("📊 Bar Chart: Average Price vs Number of Bedrooms")
        avg_price_bedroom = df3.groupby('Bedroom')['Price'].mean().reset_index()
        fig = px.bar(avg_price_bedroom, x='Bedroom', y='Price', title='Average Price vs Number of Bedrooms', color='Price')
        st.plotly_chart(fig)

    col5, col6 = st.columns(2)

    # Bar Chart: Average Price vs Number of Bathrooms
    with col5:
        st.subheader("📊 Bar Chart: Average Price vs Number of Bathrooms")
        avg_price_bathroom = df3.groupby('Bathroom')['Price'].mean().reset_index()
        fig = px.bar(avg_price_bathroom, x='Bathroom', y='Price', title='Average Price vs Number of Bathrooms', color='Price')
        st.plotly_chart(fig)

    # Heatmap: Correlation of Amenities with Price
    st.header("🔥 Heatmap: Correlation of Amenities with Price")
    amenities = ['Barbeque area', 'Club house', 'Gymnasium', 'Jogging Track', 'Lift', 'Minimart', 'Multipurpose hall', 'Parking', 'Playground', 'Sauna', 'Security', 'Squash Court', 'Swimming Pool', 'Tennis Court', 'Price']
    corr = df3[amenities].corr()
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Real Estate Price Estimator Page
elif page == "📈 Real Estate Price Estimator":
    st.title("📈 Real Estate Price Estimator")

    # Encoding categorical features
    df_encoded = pd.get_dummies(df3, columns=['Tenure Type', 'State', 'Property Type', 'Land Title'])
    X = df_encoded.drop('Price', axis=1)
    y = df_encoded['Price']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth' : [6, 8],
    }

    CV_rfc = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=5)
    CV_rfc.fit(X_train, y_train)

    # Best model from GridSearchCV
    model = CV_rfc.best_estimator_

    # Making predictions on the test set
    y_pred = model.predict(X_test)

    # Model evaluation metrics
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)

    # User input for making predictions
    st.header("🏡 Let's Predict Your Property Value!")
    col1, col2, col3 = st.columns(3)

    with col1:
        state = st.selectbox("State", df["State"].unique())
        property_type = st.selectbox("Property Type", df["Property Type"].unique())
        property_size = st.number_input("Property Size", min_value=1, max_value=10000, value=1000)
        land_title = st.selectbox("Land Title", df["Land Title"].unique())           

    with col2:
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
        parking_lot = st.number_input("Parking Lot", min_value=1, max_value=10, value=1)
        
    with col3:
        st.subheader("Choose Amenities")
        barbeque_area = st.checkbox("Barbeque Area")
        club_house = st.checkbox("Club house")
        gymnasium = st.checkbox("Gymnasium")
        jogging_track = st.checkbox("Jogging Track")
        lift = st.checkbox("Lift")
        minimart = st.checkbox("Minimart")
        multipurpose_hall = st.checkbox("Multipurpose Hall")
        parking = st.checkbox("Parking")
        playground = st.checkbox("Playground")
        sauna = st.checkbox("Sauna")
        security = st.checkbox("Security")
        squash_court = st.checkbox("Squash Court")
        swimming_pool = st.checkbox("Swimming Pool")
        tennis_court = st.checkbox("Tennis Court")

    # Prediction button
    if st.button("Predict"):
        # Creating input data for prediction
        input_data = {'Bedrooms': [bedrooms], 'Bathrooms': [bathrooms], 'Property Size': [property_size], 'State': [state], 'Property Type': [property_type], 'Parking Lot': [parking_lot], 'Land Title': [land_title], 'Barbeque Area': [barbeque_area], 'Club house':[club_house], 'Gymnasium': [gymnasium], 'Jogging Track': [jogging_track], 'Lift': [lift], 'Minimart': [minimart], 'Multipurpose Hall': [multipurpose_hall], 'Parking': [parking], 'Playground': [playground], 'Sauna': [sauna], 'Security': [security], 'Squash Court':[squash_court], 'Swimming Pool': [swimming_pool], 'Tennis Court':[tennis_court]}
        input_df = pd.DataFrame(input_data)

        # Encoding input data and filling missing columns
        input_df = pd.get_dummies(input_df).reindex(columns=X.columns, fill_value=0)

        # Making prediction using the trained model
        prediction = model.predict(input_df)[0]
        st.write(f"<p style='font-size:20px; font-weight:bold;'>Predicted Price: RM {prediction:.2f}</p>", unsafe_allow_html=True)

        # Displaying evaluation values
        st.subheader("Prediction Evaluation")
        st.write(f"Mean Absolute Percentage Error: {mape:.2f}")
        st.write(f"R-squared (R2) Score: {r2:.2f}")

