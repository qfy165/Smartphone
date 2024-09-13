import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('smartphone.csv')  # Replace with your file path
    return df

# Preprocess the data
def preprocess_data(df):
    df['price'] = df['Price'].str.replace('MYR ', '', regex=False).str.replace(',', '', regex=False).astype(float)
    features = ['price', 'battery_capacity', 'ram_capacity', 'internal_memory', 'screen_size', 'primary_camera_rear', 'primary_camera_front']
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
    return similar_indices

# Enhanced UI for recommender system 1
def recommender_system_1(df_original, df_scaled, features, scaler):
    st.subheader('💡 Recommender System 1: Select a Phone')
    selected_phone = st.selectbox('📱 Choose a Phone', df_original['model'])
    selected_phone_row = df_original[df_original['model'] == selected_phone].iloc[0]
    selected_phone_features = selected_phone_row[features].values.reshape(1, -1)
    similarity = cosine_similarity(selected_phone_features, df_scaled[features])
    top_indices = similarity.argsort()[0][-6:-1][::-1]
    
    st.write("📋 Similar Phones You Might Like:")
    st.table(df_original.iloc[top_indices][['brand_name', 'model', 'price', 'battery_capacity', 'ram_capacity', 'internal_memory', 'screen_size', 'primary_camera_rear', 'primary_camera_front']])

# Enhanced UI for recommender system 2
def recommender_system_2(df_original, df_scaled, features, scaler):
    st.subheader('⚙️ Recommender System 2: Customize Your Preferences')
    st.sidebar.header('🎛️ Set Your Preferences')

    # Get unique brand and processor brand lists
    brand_list = df_original['brand_name'].unique().tolist()
    processor_list = df_original['processor_brand'].unique().tolist()

    # Add "Any Brand" and "Any Processor Brand" options
    brand_list_with_any = ['Any Brand'] + brand_list
    processor_list_with_any = ['Any Processor Brand'] + processor_list

    # Select brand and processor brand
    selected_brand = st.sidebar.selectbox('Choose a Smartphone Brand', options=brand_list_with_any)
    if selected_brand != 'Any Brand':
        filtered_processor_list = df_original[df_original['brand_name'] == selected_brand]['processor_brand'].unique().tolist()
        processor_list_with_any = ['Any Processor Brand'] + filtered_processor_list
    selected_processor_brand = st.sidebar.selectbox('Choose a Processor Brand', options=processor_list_with_any)

    if selected_processor_brand != 'Any Processor Brand':
        filtered_brand_list = df_original[df_original['processor_brand'] == selected_processor_brand]['brand_name'].unique().tolist()
        brand_list_with_any = ['Any Brand'] + filtered_brand_list

    with st.sidebar.form(key='preferences_form'):
        price = st.slider('Max Price (MYR)', min_value=int(df_original['price'].min()), max_value=int(df_original['price'].max()), value=1500)
        battery_capacity = st.slider('Min Battery Capacity (mAh)', min_value=int(df_original['battery_capacity'].min()), max_value=int(df_original['battery_capacity'].max()), value=4000)
        ram_capacity = st.slider('Min RAM (GB)', min_value=int(df_original['ram_capacity'].min()), max_value=int(df_original['ram_capacity'].max()), value=6)
        internal_memory = st.slider('Min Internal Memory (GB)', min_value=int(df_original['internal_memory'].min()), max_value=int(df_original['internal_memory'].max()), value=128)
        screen_size = st.slider('Min Screen Size (inches)', min_value=float(df_original['screen_size'].min()), max_value=float(df_original['screen_size'].max()), value=6.5)
        rear_camera = st.selectbox('Choose Min Rear Camera MP', sorted(df_original['primary_camera_rear'].unique()))
        front_camera = st.selectbox('Choose Min Front Camera MP', sorted(df_original['primary_camera_front'].unique()))
        submit_button = st.form_submit_button(label='🔍 Find Smartphones')

    user_preferences = [price, battery_capacity, ram_capacity, internal_memory, screen_size, rear_camera, front_camera]

    if submit_button:
        df_filtered = df_scaled.copy()
        df_original_filtered = df_original.copy()

        if selected_brand != 'Any Brand':
            df_filtered = df_filtered[df_original['brand_name'] == selected_brand]
            df_original_filtered = df_original[df_original['brand_name'] == selected_brand]

        if selected_processor_brand != 'Any Processor Brand':
            df_filtered = df_filtered[df_original['processor_brand'] == selected_processor_brand]
            df_original_filtered = df_original[df_original['processor_brand'] == selected_processor_brand]

        similar_indices = recommend_smartphones(df_filtered, user_preferences, features, scaler)
        recommendations = df_original_filtered.iloc[similar_indices]

        st.subheader(f'📱 Recommended Smartphones for Brand: {selected_brand} and Processor: {selected_processor_brand}')
        st.table(recommendations[['brand_name', 'model', 'price', 'battery_capacity', 'processor_brand', 'ram_capacity', 'internal_memory', 'screen_size', 'primary_camera_rear', 'primary_camera_front']])

# Main function to organize the UI
def main():
    st.set_page_config(page_title="Smartphone Recommender", page_icon="📱", layout="centered")
    st.title('📱 Smartphone Recommender System')

    st.markdown("""
    Welcome to the **Smartphone Recommender**! 🎉 Use the options in the sidebar to set your preferences, and we'll recommend the best smartphones based on your needs. 
    """)

    df = load_data()
    df_scaled, df_original, features, scaler = preprocess_data(df)

    st.sidebar.title('🧭 Choose Recommender System')
    system_choice = st.sidebar.selectbox('Select Recommender System', ['Recommender System 1', 'Recommender System 2'])

    if system_choice == 'Recommender System 1':
        recommender_system_1(df_original, df_scaled, features, scaler)
    elif system_choice == 'Recommender System 2':
        recommender_system_2(df_original, df_scaled, features, scaler)

if __name__ == "__main__":
    main()
