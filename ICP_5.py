import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Question 1: Principal Component Analysis
print("QUESTION 1: PRINCIPAL COMPONENT ANALYSIS")

# Read the file
CCG_data = pd.read_csv('C:\\Users\\augie\\OneDrive\\Desktop\\2024 Spring\\Machine Learning\\ICP #5\\datasets\\CC GENERAL.csv')

# Prune the csv file of any data we don't need
CCG_data.dropna(inplace=True)

# Apply Scaling to normalize data
standardScaler = StandardScaler()
CCG_scaled = standardScaler.fit_transform(CCG_data.drop(columns=['CUST_ID']))

# Acquire a "before" sample of data to reference later
kmeans_raw = KMeans(n_clusters=3)
kmeans_raw.fit(CCG_scaled)
kmeans_raw = kmeans_raw.predict(CCG_scaled)

# Apply k-means to "raw" result
CCG_raw_silhouette = silhouette_score(CCG_scaled, kmeans_raw)

# Used normalized data for PCA
pca = PCA(n_components=2)
CCG_PCAd = pca.fit_transform(CCG_scaled)

# Create a semi-duplication of data preparing for k-means integration
kmeans_PCA = KMeans(n_clusters=3)
kmeans_PCA.fit(CCG_PCAd)

# Apply k-means to the PCA result
kmeans_PCA = kmeans_PCA.predict(CCG_PCAd)

# Compare the two results to report performance
CCG_silhouette = silhouette_score(CCG_PCAd, kmeans_PCA)
print("Silhouette score before PCA: ", CCG_raw_silhouette)
print("Silhouette score after PCA: ", CCG_silhouette)

# Question 2: Tinkering with pd_speech_features.csv
print("\nQUESTION 2: Tinkering with pd_speech_features.csv")

# Read the file
speech_data = pd.read_csv('C:\\Users\\augie\\OneDrive\\Desktop\\2024 Spring\\Machine Learning\\ICP #5\\datasets\\pd_speech_features.csv')

# Apply Scaling to normalize the data
speech_scaled = standardScaler.fit_transform(speech_data.drop(columns = ['class']))

# Use PCA with specified k=3
speech_PCAd = PCA(n_components=3)
speech_PCAd = speech_PCAd.fit_transform(speech_scaled)

# Find the SVM Score on the PCA'd data set
x_train, x_test, y_train, y_test = train_test_split(speech_PCAd, speech_data['class'], test_size=0.2, random_state=42)
speech_svm = SVC()
speech_svm.fit(x_train, y_train)
svmScore = speech_svm.score(x_test, y_test)

# Report the performance using SVM score
print('SVM score after PCA: ', svmScore)

# Question 3: Apply LDA on Iris.csv -> reduce dimensionality to 2=k
print("\nQUESTION 3: Apply LDA on Iris.csv -> reduce dimensionality to 2=k\n (see code, not much to display)")

# Read the file
iris_data = pd.read_csv('C:\\Users\\augie\\OneDrive\\Desktop\\2024 Spring\\Machine Learning\\ICP #5\\datasets\\Iris.csv')

# Apply LDA to the
iris_LDAd = LDA(n_components=2)
iris_LDAd = iris_LDAd.fit_transform(iris_data.drop(columns=['Species']), iris_data['Species'])

#  Question 4: Identify the Difference between PCA and LDA
print("\nQuestion 4: Identify the Difference between PCA and LDA")
print("  Both function to help reduce the dimensionality of a given dataset.")
print("  While both achieve the same goal, they are specifically catered to individual tasks.")
print("    PCA is unsupervised, and help to find the largest degree of variability in the dataset its given.")
print("    LDA is supervised, and uses this to find the largest separation of data dependant on classes.")
print("  The contrast is in supervision, and how they use the data their given. "
      "\n    In a way, PCA works with 'naked', unnamed data that needs degrees of separation,"
      "\n    while LDA works with 'tagged', named data that can use said tags for greater distinction")
