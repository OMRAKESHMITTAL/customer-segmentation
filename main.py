import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from io import BytesIO

# Function to generate a sample Excel file
def generate_sample_file():
    sample_data = {
        "InvoiceNo": ["A0001", "A0002", "A0003"],
        "StockCode": ["S001", "S002", "S003"],
        "Description": ["Product 1", "Product 2", "Product 3"],
        "Quantity": [10, 5, 15],
        "InvoiceDate": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
        "UnitPrice": [20.0, 35.0, 10.0],
        "CustomerID": ["C001", "C002", "C003"],
        "Country": ["USA", "UK", "Canada"]
    }
    df = pd.DataFrame(sample_data)
    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    return buffer

# Title and instructions
st.title("Customer Segmentation with RFM Analysis")

st.markdown("""
### Instructions
Please upload an Excel file with the following mandatory columns:
- `InvoiceNo`
- `StockCode`
- `Description`
- `Quantity`
- `InvoiceDate`
- `UnitPrice`
- `CustomerID`
- `Country`

If you don't have a file ready, you can download a [sample file here](https://example.com/sample_file.xlsx). 
""")

# Provide a download link for the sample file

# Path to the pre-made sample file
sample_file_path = "Online Retail.xlsx"

# Provide a download link for the sample file
with open(sample_file_path, "rb") as file:
    st.sidebar.subheader("Download Sample Excel File")
    st.sidebar.download_button(
        label="Download Sample Excel File",
        data=file,
        file_name="sample_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
# Upload file
uploaded_file = st.file_uploader("Choose a file", type="xlsx")

if uploaded_file is not None:
    # Read the uploaded file
    try:
        data = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        st.stop()

    # Validate required columns
    required_columns = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        st.error(f"The uploaded file is missing the following mandatory columns: {', '.join(missing_columns)}")
        st.stop()

    # Data cleaning steps
    data.dropna(subset=["CustomerID"], axis=0, inplace=True)
    data = data[~data['InvoiceNo'].str.contains('C', na=False)]
    data = data.drop_duplicates(keep="first")

    # Remove outliers
    def replace_with_threshold(dataframe, variable):
        q1 = dataframe[variable].quantile(0.01)
        q3 = dataframe[variable].quantile(0.99)
        iqr = q3 - q1
        up_limit = q3 + 1.5 * iqr
        dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit

    replace_with_threshold(data, "Quantity")
    replace_with_threshold(data, "UnitPrice")

    data["Revenue"] = data["Quantity"] * data["UnitPrice"]
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

    Latest_Date = dt.datetime(2011, 12, 10)
    RFM = data.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (Latest_Date - x.max()).days,
        'InvoiceNo': lambda x: x.nunique(),
        'Revenue': lambda x: x.sum()
    })
    RFM.rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'Revenue': 'Monetary'}, inplace=True)
    RFM = RFM[RFM["Frequency"] > 1]

    Shopping_Cycle = data.groupby('CustomerID').agg({'InvoiceDate': lambda x: (x.max() - x.min()).days})
    RFM["Shopping_Cycle"] = Shopping_Cycle
    RFM["Interpurchase_Time"] = RFM["Shopping_Cycle"] // RFM["Frequency"]

    RFMT = RFM[["Recency", "Frequency", "Monetary", "Interpurchase_Time"]]

    # Finding optimal number of clusters using Elbow Method
    SSD = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, max_iter=50)
        kmeans.fit(RFMT)
        SSD.append(kmeans.inertia_)

    st.subheader("Optimal Number of Clusters")
    st.write("""
    The Elbow Method is used to determine the optimal number of clusters for KMeans. 
    The 'elbow' point in the SSD (Sum of Squared Distances) plot indicates the best number of clusters. 
    Adding more clusters beyond this point does not significantly reduce the SSD.
    """)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(K_range, SSD, marker='o')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Sum of Squared Distances (SSD)')
    ax.set_title('Elbow Method for Optimal K')
    st.pyplot(fig)

    # Fit KMeans Model with the chosen number of clusters
    kmeans = KMeans(n_clusters=4, max_iter=50)
    kmeans.fit(RFMT)
    RFMT["Clusters"] = kmeans.labels_

    # Plotting Clusters
    st.subheader("Cluster Visualization")
    st.write("""
    This scatter plot shows the distribution of customers in terms of Recency and Frequency, colored by cluster. 
    Recency represents the number of days since the customer's last purchase, while Frequency indicates the number of purchases.
    Centroids are marked to visualize the center of each cluster.
    """)
    fig, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(RFMT["Recency"], RFMT["Frequency"], c=RFMT["Clusters"], cmap='viridis')
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color="red", marker="*", label="Centroids")
    ax.set_xlabel("Recency")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

    # Additional Visualizations
    st.subheader("Recency vs Frequency by Cluster")
    st.write("""
    This scatter plot displays Recency versus Frequency for each cluster. 
    Recency measures the number of days since the last purchase, and Frequency indicates the number of purchases. 
    This helps in understanding how recently and frequently customers purchase within each cluster.
    """)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(x="Recency", y="Frequency", hue="Clusters", data=RFMT, palette='viridis', ax=ax)
    ax.set_title('Recency vs Frequency by Cluster')
    ax.set_xlabel('Recency (Days Since Last Purchase)')
    ax.set_ylabel('Frequency (Number of Purchases)')
    st.pyplot(fig)

    st.subheader("Monetary Value Distribution")
    st.write("""
    This histogram shows the distribution of Monetary Values among customers. 
    Monetary Value represents total spending by a customer. The Kernel Density Estimate (KDE) curve provides a smooth estimate of the distribution.
    """)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(RFM['Monetary'], bins=30, kde=True, ax=ax)
    ax.set_title('Distribution of Monetary Value')
    ax.set_xlabel('Monetary Value (Total Spending)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    st.subheader("Frequency vs Monetary Value by Cluster")
    st.write("""
    This scatter plot shows the relationship between Frequency (number of purchases) and Monetary Value (total spending) across different clusters. 
    It helps in understanding how the number of purchases relates to total spending within each cluster.
    """)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(x="Frequency", y="Monetary", hue="Clusters", data=RFMT, palette='viridis', ax=ax)
    ax.set_title('Frequency vs Monetary Value by Cluster')
    ax.set_xlabel('Frequency (Number of Purchases)')
    ax.set_ylabel('Monetary Value (Total Spending)')
    st.pyplot(fig)

    # Silhouette Score
    score = silhouette_score(RFMT, kmeans.labels_, metric='euclidean')
    st.write(f"Silhouette Score: {score:.2f}")

    # Display the final RFM table with clusters
    st.subheader("RFM Table with Clusters")
    st.write(RFMT.head())
