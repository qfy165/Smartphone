import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
@st.cache
def load_data():
    df = pd.read_csv('smartphone.csv')  # Replace with your file path
    return df

# Preprocess the data
def preprocess_data(df):
    # Convert 'Price' from MYR format to numeric (remove 'MYR ' and convert to float)
    df['Price'] = df['Price'].str.replace('MYR ', '').str.replace(',', '').astype(float)
    
    # Select numerical columns for similarity computation
    features = ['Price', 'rating', 'battery_capacity', 'ram_capacity', 'internal_memory', 'screen_size']
    
    # Fill missing values with the mean
    df[features] = df[features].apply(pd.to_numeric, errors='coerce')  # Convert to numeric
    df[features] = df[features].fillna(df[features].mean())

    # Normalize the feature values to a range of [0,1] for comparison
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    
    return df, features

# Recommend smartphones based on similarity
def recommend_smartphones(df, user_preferences, features, top_n=5):
    # Add user preferences as a new row in the dataset
    user_preferences_scaled = MinMaxScaler().fit_transform([user_preferences])
    df_with_user = pd.concat([df, pd.DataFrame(user_preferences_scaled, columns=features)], ignore_index=True)
    
    # Compute cosine similarity between user preferences and smartphones
    similarity = cosine_similarity(df_with_user[features])
    
    # Get the top N most similar smartphones (excluding the user preference row)
    similar_indices = similarity[-1, :-1].argsort()[-top_n:][::-1]
    
    return df.iloc[similar_indices]

# Streamlit App
def main():
    st.title('Smartphone Recommender System (MYR)')
    
    # Load and preprocess the data
    df = load_data()
    df, features = preprocess_data(df)

    # User input: preferences for smartphone features
    st.sidebar.header('Set Your Preferences')
    price = st.sidebar.slider('Max Price (MYR)', min_value=int(df['Price'].min()), max_value=int(df['Price'].max()), value=1500)
    rating = st.sidebar.slider('Min Rating', min_value=0, max_value=100, value=80)
    battery_capacity = st.sidebar.slider('Min Battery Capacity (mAh)', min_value=int(df['battery_capacity'].min()), max_value=int(df['battery_capacity'].max()), value=4000)
    ram_capacity = st.sidebar.slider('Min RAM (GB)', min_value=int(df['ram_capacity'].min()), max_value=int(df['ram_capacity'].max()), value=6)
    internal_memory = st.sidebar.slider('Min Internal Memory (GB)', min_value=int(df['internal_memory'].min()), max_value=int(df['internal_memory'].max()), value=128)
    screen_size = st.sidebar.slider('Min Screen Size (inches)', min_value=float(df['screen_size'].min()), max_value=float(df['screen_size'].max()), value=6.0)
    
    # Store user preferences
    user_preferences = [price, rating, battery_capacity, ram_capacity, internal_memory, screen_size]
    
    # Recommend smartphones
    recommendations = recommend_smartphones(df, user_preferences, features)
    
    # Display recommendations
    st.subheader('Recommended Smartphones for You')
    st.write(recommendations[['brand_name', 'model', 'Price', 'rating', 'battery_capacity', 'ram_capacity', 'internal_memory', 'screen_size']])

if __name__ == "__main__":
    main()
