import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Set Streamlit to use wide layout
st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('smartphone.csv')  # Replace with your file path
    return df

# Preprocess the data
def preprocess_data(df):
    df['price'] = df['Price'].str.replace('MYR ', '', regex=False).str.replace(',', '', regex=False).astype(float)
    
    # Include front and rear camera in features list
    features = ['price', 'battery_capacity', 'ram_capacity', 'internal_memory', 'screen_size', 'rear_camera', 'front_camera']
    df[features] = df[features].apply(pd.to_numeric, errors='coerce')  # Convert to numeric
    
    df[features] = df[features].fillna(df[features].mean())
    df_original = df.copy()
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    return df, df_original, features, scaler

# Recommend smartphones based on similarity
def recommend_smartphones(df, user_preferences, features, scaler, top_n=10):
    user_preferences_scaled = scaler.transform([user_preferences])
    user_preferences_df = pd.DataFrame(user_preferences_scaled, columns=features)
    df_with_user = pd.concat([df, user_preferences_df], ignore_index=True)
    similarity = cosine_similarity(df_with_user[features])
    similar_indices = similarity[-1, :-1].argsort()[-top_n:][::-1]
    
    # Return only valid indices (remove rows where indices exceed DataFrame length)
    similar_indices = [i for i in similar_indices if i < len(df)]
    
    return similar_indices

# Streamlit App
def main():
    st.title('Smartphone Recommender System')

    df = load_data()
    df_scaled, df_original, features, scaler = preprocess_data(df)

    st.sidebar.header('Set Your Preferences')
    
    # Dropdown for brand selection
    brand_list = ['Every Brand'] + df_original['brand_name'].unique().tolist()
    selected_brand = st.sidebar.selectbox('Choose a brand', options=brand_list, index=0)

    # Dropdown for processor brand selection
    processor_list = ['Every Processor'] + df_original['processor_brand'].unique().tolist()
    selected_processor = st.sidebar.selectbox('Choose a processor brand', options=processor_list, index=0)

    df_filtered = df_scaled.copy()
    df_original_filtered = df_original.copy()
    
    if selected_brand != 'Every Brand':
        df_filtered = df_filtered[df_original['brand_name'] == selected_brand]
        df_original_filtered = df_original[df_original['brand_name'] == selected_brand]
    
    if selected_processor != 'Every Processor':
        df_filtered = df_filtered[df_original['processor_brand'] == selected_processor]
        df_original_filtered = df_original[df_original['processor_brand'] == selected_processor]

    # User input: preferences for features including front and rear cameras
    price = st.sidebar.slider('Max Price (MYR)', min_value=int(df_original_filtered['price'].min()), max_value=int(df_original_filtered['price'].max()), value=1500)
    battery_capacity = st.sidebar.slider('Min Battery Capacity (mAh)', min_value=int(df_original_filtered['battery_capacity'].min()), max_value=int(df_original_filtered['battery_capacity'].max()), value=4000)
    ram_capacity = st.sidebar.slider('Min RAM (GB)', min_value=int(df_original_filtered['ram_capacity'].min()), max_value=int(df_original_filtered['ram_capacity'].max()), value=6)
    internal_memory = st.sidebar.slider('Min Internal Memory (GB)', min_value=int(df_original_filtered['internal_memory'].min()), max_value=int(df_original_filtered['internal_memory'].max()), value=128)
    screen_size = st.sidebar.slider('Min Screen Size (inches)', min_value=float(df_original_filtered['screen_size'].min()), max_value=float(df_original_filtered['screen_size'].max()), value=6.5)

    # Dropdown for rear and front camera megapixels
    rear_camera_options = ['Any'] + sorted(df_original_filtered['rear_camera'].dropna().unique().tolist())
    selected_rear_camera = st.sidebar.selectbox('Min Rear Camera (MP)', options=rear_camera_options, index=0)
    
    front_camera_options = ['Any'] + sorted(df_original_filtered['front_camera'].dropna().unique().tolist())
    selected_front_camera = st.sidebar.selectbox('Min Front Camera (MP)', options=front_camera_options, index=0)
    
    # Filter the dataframe by selected camera preferences
    if selected_rear_camera != 'Any':
        df_filtered = df_filtered[df_original['rear_camera'] >= float(selected_rear_camera)]
        df_original_filtered = df_original_filtered[df_original_filtered['rear_camera'] >= float(selected_rear_camera)]
    
    if selected_front_camera != 'Any':
        df_filtered = df_filtered[df_original['front_camera'] >= float(selected_front_camera)]
        df_original_filtered = df_original_filtered[df_original_filtered['front_camera'] >= float(selected_front_camera)]
    
    # Store user preferences
    user_preferences = [price, battery_capacity, ram_capacity, internal_memory, screen_size]
    
    # Add rear and front camera to user preferences if they are not 'Any'
    if selected_rear_camera != 'Any':
        user_preferences.append(float(selected_rear_camera))
    else:
        user_preferences.append(df_original_filtered['rear_camera'].mean())
    
    if selected_front_camera != 'Any':
        user_preferences.append(float(selected_front_camera))
    else:
        user_preferences.append(df_original_filtered['front_camera'].mean())
    
    similar_indices = recommend_smartphones(df_filtered, user_preferences, features, scaler)
    
    # Display only non-empty rows based on filtered indices
    recommendations = df_original_filtered.iloc[similar_indices]
    
    st.subheader(f'Recommended Smartphones for Brand: {selected_brand} and Processor: {selected_processor}')
    
    # Display front and rear camera in the result
    st.dataframe(recommendations[['brand_name', 'model', 'price', 'processor_brand', 'battery_capacity', 'ram_capacity', 'internal_memory', 'screen_size', 'rear_camera', 'front_camera']], height=600, width=1200)

if __name__ == "__main__":
    main()
