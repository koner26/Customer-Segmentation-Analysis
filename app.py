import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


# Title of the app
st.title('Customer Segmentation and Classification App')
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])


if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    df = df.dropna()
    st.write("Data preview:")
    st.dataframe(df.head())

    # Preprocess data
    st.sidebar.header('Data Preprocessing')
    features = st.sidebar.multiselect('Select features for segmentation:', df.columns)


    if len(features) > 0:
        X = df[features]
        try:
            # K-Means Clustering
            st.sidebar.header('K-Means Clustering')
            n_clusters = st.sidebar.slider('Number of clusters:', 2, 10, 3)
            kmeans = KMeans(n_clusters=n_clusters)
            df['Segment'] = kmeans.fit_predict(X)

        except Exception as e:
            st.error("Select Appropriate Value.")

        st.write("Segmented data:")
        st.dataframe(df.head())

        # Visualize clusters
        st.sidebar.header('Cluster Visualization')
        if len(features) >= 2:
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=features[0], y=features[1], hue='Segment', palette='viridis', ax=ax)
            st.pyplot(fig)

        # Classification
        st.sidebar.header('Classification Model')
        target = st.sidebar.selectbox('Select target variable:', df.columns)

        if target:
            y = df['Segment']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

            # Feature importance
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            feature_names = [features[i] for i in indices]

            fig, ax = plt.subplots()
            sns.barplot(x=importances[indices], y=feature_names, ax=ax)
            ax.set_title('Feature Importance')
            st.pyplot(fig)

            # User input for prediction
            st.sidebar.header('Predict New Data')
            input_data = []
            for feature in features:
                if df[feature].dtype == 'object':
                    unique_vals = df[feature].unique()
                    value = st.sidebar.selectbox(f"Input {feature}", unique_vals)
                else:
                    value = st.sidebar.number_input(f"Input {feature}", float(df[feature].min()),
                                                    float(df[feature].max()))
                input_data.append(value)

            if st.sidebar.button('Predict'):
                input_df = pd.DataFrame([input_data], columns=features)
                segment_prediction = model.predict(input_df)
                st.write(f"Predicted Segment: {segment_prediction[0]}")
else:
    st.write("Please upload a CSV file to get started.")


