import streamlit as st
import logging
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_squared_error
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import re
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
import base64
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline





import matplotlib.pyplot as plt
st.title("Execution of Machine Learning Algorithm on Different Datasets")

# Read the CSV file
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])






# Add a dropdown box to choose file type
file_type = st.selectbox("Choose Algorithm Type", ("NONE","Classification Type", "Regression Type"))
# Load the appropriate file type based on user selection

if file_type=="NONE":
    pass
elif file_type=="Classification Type":
    algo=st.selectbox("Choose Algorithm",("K NEAREST NEIGHBOUR","SVMC","DECISION TREE","K-MEANS"))
else:
    algo=st.selectbox("Choose Algorithm",("LINEAR REGRESSION","SVMR","LASSO REGRESSION"))

# function to compare dataset

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    le = LabelEncoder()

    if len(df.select_dtypes("object").columns.tolist())>0:
        for column_name in df.select_dtypes("object").columns.tolist():
            df[column_name] = le.fit_transform(df[column_name])
    else: 
        pass
    num_rows, num_cols = df.shape
    st.write(f"Number of Rows: {num_rows}")
    st.write(f"Number of Columns: {num_cols}")
    object_cols = df.select_dtypes(include=['object'])
    st.write(df)
    X = df.iloc[:, :-1]  
    y = df.iloc[:, -1]



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if algo=="SVMR":
        # image_path = 'SVM-model.jpg'  # Path to your SVMR image
        # st.markdown(
        #     f"""
        #     <div style="display: flex; justify-content: flex-end;">
        #         <img src="data:image/png;base64,{base64.b64encode(open(image_path, "rb").read()).decode()}" width="400" />
        #         <p style="font-size: 28px;">{"SUPPORT VECTOR MACHINE ALGO FLOW DIAGRAM"}</p>
        #     </div>
        #     """,
        #     unsafe_allow_html=True
        # )
        svm_regressor = SVR(kernel='rbf')
        svm_regressor.fit(X_train, y_train)
        y_pred = svm_regressor.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"The Mean Squared Error is: {mse}", font=("Arial", 34))
        st.write(f"The Mean Absolute Error is: {mae}", font=("Arial", 34))
        st.write(f"The R² Score is: {r2}", font=("Arial", 34))

        # Optionally, plot y_test vs y_pred to visualize regression performance
        plt.figure(figsize=(6, 4))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted')
        st.pyplot(plt)

        image_path = 'SVM-model.jpg'  # Path to your SVMR image
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end;">
                <img src="data:image/png;base64,{base64.b64encode(open(image_path, "rb").read()).decode()}" width="400" />
                <p style="font-size: 28px;">{"SUPPORT VECTOR MACHINE ALGO FLOW DIAGRAM"}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # accuracy = accuracy_score(y_test, y_pred)
        # f1_scores = cross_val_score(svm_classifier, X, y, cv=5, scoring='f1_macro')
        # st.write(f"The Accuracy is : {accuracy}", font=("Arial", 34))
        # st.write(f"The F1 Score is : {f1_scores.mean()}", font=("Arial", 34))
        
        # cm = confusion_matrix(y_test, y_pred)
        # plt.figure(figsize=(6,4))
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        # plt.xlabel('Predicted')
        # plt.ylabel('Actual')
        # plt.title('Confusion Matrix for SVM')
        # st.pyplot(plt)
    elif algo=="K NEAREST NEIGHBOUR":
        k_value = st.slider('Select k value for KNN:', min_value=1, max_value=10, value=5)
        knn_classifier = KNeighborsClassifier(n_neighbors=k_value)
        knn_classifier.fit(X_train, y_train)
        y_pred = knn_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        st.write(f"Accuracy: {accuracy}")
        f1 = f1_score(y_test, y_pred, average='macro')
        st.write(f"F1 Score (macro): {f1.mean()}")
        # Plotting confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix for KNN')
        st.pyplot(plt)
        image_path = 'KNN.png'  # Path to your SVR image
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end;">
                <img src="data:image/png;base64,{base64.b64encode(open(image_path, "rb").read()).decode()}" width="400" />
                <p style="font-size: 24px;">{"KNN ALGO FLOW DIAGRAM"}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif algo=="DECISION TREE":
        dt_classifier = DecisionTreeClassifier()
        dt_classifier.fit(X_train, y_train)
        y_pred = dt_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy}")
        # Calculating F1 score
        f1 = f1_score(y_test, y_pred, average='macro')
        st.write(f"F1 Score (macro): {f1}")
        # Plotting confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix for Decision Tree')
        st.pyplot(plt)
        image_path = 'decision-tree-algo.png'  # Path to your SVR image
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end;">
                <img src="data:image/png;base64,{base64.b64encode(open(image_path, "rb").read()).decode()}" width="400" />
                <p style="font-size: 24px;">{"DECISION TREE ALGO FLOW DIAGRAM"}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif algo=="K-MEANS":
        image_path = 'k-means-cluster.png'  # Path to your SVR image
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end;">
                <img src="data:image/png;base64,{base64.b64encode(open(image_path, "rb").read()).decode()}" width="400" />
                <p style="font-size: 24px;">{"K MEANS ALGO FLOW DIAGRAM"}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.write("Clustered DataFrame:")
        st.write(df)
        wscc = []
        range_values = range(1,11)
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)
        for i in range_values:
            kmeans = KMeans(n_clusters = i)
            kmeans.fit(df_scaled)
            wscc.append(kmeans.inertia_)

        plt.figure(figsize = (5,4), facecolor = 'grey') 
        plt.plot(wscc, 'rx-', marker = '*', color = 'red')
        plt.xlabel('Cluster Number', fontsize = 10)
        plt.title('ELBOW',fontsize = 15)
        st.pyplot(plt)

        num_clusters = st.slider("Select number of clusters:", min_value=1, max_value=10, value=2)
        # K-Means clustering
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(df)
        labels = kmeans.labels_
        # Add cluster labels to DataFrame
        df['cluster'] = labels
        # Display clustered DataFrame

        plt.figure(figsize=(8, 6))
        plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis', label='Cluster')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', color='red', label='Centroids')
        plt.title('K-Means Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        st.pyplot(plt)

    elif algo=="LINEAR REGRESSION":
        linear_reg = LinearRegression()
        linear_reg.fit(X_train, y_train)
        y_pred = linear_reg.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Mean Squared Error (MSE): {mse}")
        mae = mean_absolute_error(y_test, y_pred)
        st.write(f"Mean Absolute Error (MAE): {mae}")
        #plotting the graph
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='red')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs. Predicted')
        st.pyplot(plt)
        image_path = 'LREGRESS.png'  # Path to your SVR image
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end;">
                <img src="data:image/png;base64,{base64.b64encode(open(image_path, "rb").read()).decode()}" width="400" />
                <p style="font-size: 24px;">{"LINEAR REGRESSION ALGO FLOW DIAGRAM"}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif algo=="LASSO REGRESSION":
        alpha = st.slider("Select regularization parameter (alpha):", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_train, y_train)
        # Making predictions
        y_pred = lasso.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Mean Squared Error: {mse}")
        r2_test = r2_score(y_test, y_pred)
        r2_train = r2_score(y_train, lasso.predict(X_train))
        st.write(f"R² Score on Test Set: {r2_test}")
        st.write(f"R² Score on Training Set: {r2_train}")
        # Plotting the regression line (for simple linear regression or 2D case)
        plt.figure(figsize=(6, 4))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted')
        st.pyplot(plt)

        # Plotting R² scores vs Alpha
        alphas = np.linspace(0.01, 1.0, 100)
        r2_train_scores = []
        r2_test_scores = []

        for a in alphas:
            lasso = Lasso(alpha=a)
            lasso.fit(X_train, y_train)
            r2_train_scores.append(r2_score(y_train, lasso.predict(X_train)))
            r2_test_scores.append(r2_score(y_test, lasso.predict(X_test)))

        plt.figure(figsize=(6, 4))
        plt.plot(alphas, r2_train_scores, label='Train R²')
        plt.plot(alphas, r2_test_scores, label='Test R²')
        plt.xlabel('Alpha')
        plt.ylabel('R² Score')
        plt.title('R² Score vs Alpha')
        plt.legend()
        st.pyplot(plt)
        image_path = 'lasso.jpg'  # Path to your SVR image
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end;">
                <img src="data:image/png;base64,{base64.b64encode(open(image_path, "rb").read()).decode()}" width="600" />
                <p style="font-size: 18px;">{"LASSO REGRESSION ALGO FLOW DIAGRAM"}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    if algo=="SVMC":
      
        # Assuming X_train, X_test, y_train, y_test are already defined
        svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale')
        svm_classifier.fit(X_train, y_train)
        y_pred = svm_classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1_scores = cross_val_score(svm_classifier, X_train, y_train, cv=5, scoring='f1_macro')

        st.write(f"The Accuracy is: {accuracy}")
        st.write(f"The F1 Score is: {f1_scores.mean()}")

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix for SVMC')
        st.pyplot(plt)
        image_path = 'svmc.png'  # Path to your SVMR image
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end;">
                <img src="data:image/png;base64,{base64.b64encode(open(image_path, "rb").read()).decode()}" width="400" />
                <p style="font-size: 28px;">{"SUPPORT VECTOR MACHINE CLASSIFIER ALGO FLOW DIAGRAM"}</p>
            </div>
            """,
            unsafe_allow_html=True
        )