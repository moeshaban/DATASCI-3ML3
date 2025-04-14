# Customer Segmentation Using K-Means and Autoencoders

## Overview
This project performs customer segmentation using the Wholesale Customers dataset from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/index.php). The goal is to uncover natural groupings within customer spending patterns and derive actionable business insights. The analysis leverages both traditional clustering (via K-Means) and modern deep learning approaches (using an autoencoder for feature extraction). Dimensionality reduction techniques like PCA and t-SNE help visualize the clusters, while evaluation metrics such as the silhouette score validate the clustering quality.

## Files Included
- **Final_Project_3ML3.ipynb** – The complete Jupyter Notebook with executable code and detailed markdown explanations.
- **Final_Project_3ML3.pdf** – A PDF export of the Jupyter Notebook (combined code and writeup, limited to 10 pages).
- **wholesale+customers.zip** – The dataset archive containing the "Wholesale customers data.csv" file.
- **Wholesale_Customers_Clustered.csv** – The final dataset with cluster labels added.
- (Optional) **kmeans_model.pkl** – The saved K-Means clustering model.
- (Optional) **autoencoder_model.h5** – The saved autoencoder model.

## Project Structure and Approach

### Data Loading and Preprocessing
- The dataset is extracted from a zip file and loaded into a Pandas DataFrame.
- Basic exploratory data analysis (EDA) is performed using `df.info()`, `df.describe()`, and a correlation heatmap.
- Numerical features are normalized using `StandardScaler` (excluding the categorical columns "Channel" and "Region").

### Clustering with K-Means
- The optimal number of clusters is determined using the Elbow Method.
- K-Means is applied to the scaled dataset, and cluster labels are added to the DataFrame.
- Clusters are visualized in a 2D space using PCA.

### Cluster Analysis and Visualization
- A heatmap of the mean feature values per cluster provides insights into the spending profiles of each segment.
- Silhouette scores are calculated to quantitatively evaluate the separation and cohesion of the clusters.

### Autoencoder-Based Feature Extraction
- A simple autoencoder is built and trained to compress the 6-dimensional data into a lower-dimensional representation.
- The autoencoder’s latent features, when visualized via PCA, offer an alternative perspective on cluster separation.
- A comparison of silhouette scores demonstrates slight improvements in cluster quality using the learned features.

### Additional Visualizations and Model Saving
- t-SNE is employed for a nonlinear reduction to further visualize the cluster structures.
- A pairplot is generated to explore feature interactions across clusters.
- Both the K-Means and autoencoder models are saved, and the final clustered dataset is exported to CSV for reproducibility.

## Requirements
- **Python 3.x**
- **Jupyter Notebook/Google Colab**

### Required Libraries
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow (or keras)
- joblib

## How to Run
1. **Setup:** Download or clone the repository and ensure all files (including the dataset zip file) are in the working directory.
2. **Open the Notebook:** Launch the `Final_Project_3ML3.ipynb` file in Jupyter Notebook or Google Colab.
3. **Execute Cells:** Run the notebook sequentially to reproduce the entire analysis—from data extraction to final visualizations and model saving.
4. **Review Outputs:** Verify that all cells execute without errors and that figures and markdown explanations are correctly displayed.
5. **Export:** If needed, export the notebook as a PDF ensuring that the final document is well-formatted and within the page limit.

## Future Directions
Future enhancements might include:
- Integrating additional customer attributes or external data (e.g., customer satisfaction metrics, sales history) for finer segmentation.
- Exploring alternative clustering methods such as DBSCAN or hierarchical clustering.
- Experimenting with more advanced deep learning architectures, such as variational autoencoders, to capture more nuanced patterns.

## Acknowledgments
- The Wholesale Customers dataset is provided by the UCI Machine Learning Repository.
- This project was developed as part of the Datasci 3ML3 final project at McMaster University.
- Appreciation is extended to the documentation and resources of the libraries used in this project.
