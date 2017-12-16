"""
@author: RT Rakesh
"""

import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.externals import joblib


start = time.time() # Start time

#  we have to declare directory path where our data is present.
document_path="Path where the documents are stored."

# We are reading all the filenames and creating a list filenames.
filenames=[os.path.join(document_path, each)
            for each in os.listdir(document_path)]

# We have to instantiate the Ft-Idf vectorize along with this preprocessing is built in.
vectorizer = TfidfVectorizer(input='filename',
                             token_pattern='(?u)\\b[a-zA-Z]\\w{2,}\\b',
                             stop_words='english',
                             ngram_range=(1, 2))
# We fit the vectorizer with the training sample data.
vec=vectorizer.fit(filenames)
# Next we are using the vectorizer to transform the sample data.
tfidf_result = vec.transform(filenames)

# Remove the comments below for fing optimal K.
"""
# In order to find the optimal number of K we use elbow method.
cluster_error = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(tfidf_result)
    Cluster_error.append( kmeanModel.inertia_ )
    
clusters_df = pd.DataFrame( { "num_clusters":K, "cluster_errors": cluster_errors } )
Print(clusters_df)

# Plot the elbow Plot
plt.figure(figsize=(12,6))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )
plt.xlabel('k')
plt.ylabel('Clusters_Error')
plt.title('The Elbow Method showing the optimal k')
plt.show()
"""

# From the elbow plot we find that the optimal k should be 5 post this there is not much information gain.    
num_clusters=5
kmeanModel = KMeans(n_clusters=num_clusters)
kmeanModel.fit(tfidf_result)

#We want see top features for all the clusters.
print("Top terms per cluster:")
order_centroids = kmeanModel.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(num_clusters):
    print("Cluster %d:" % i)
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print("\n")

#Saving the model for future application
# Setting the output path.
output_path="Path to store the model and vectorizer"

#Saving the Vectorizer and the kmeans model for future prediction.
joblib.dump(kmeanModel, output_path + 'finalized_Kmeans_model')
joblib.dump(vec, output_path + "finalized_tfidf_vectorizer") 

# Just for the fun of finding out the total elapsed time.
end = time.time()
elapsed = end - start
print "Time taken for K Means clustering and saving the model: ", elapsed, "seconds."
