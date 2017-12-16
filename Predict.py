"""
@author: RT Rakesh
"""

import os
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from sklearn.externals import joblib

start = time.time() # Start time

# We need to provide the path where the Test data documents are saved.
# The test data must be saved as individual files and stored in the path mentioned. 
test_document_path="Path where the test/new documents are stored"

# We are reading all the filenames and creating a list filenames.
filenames=[os.path.join(test_document_path, each)
            for each in os.listdir(test_document_path)]

# We need to provide the path where the saved Vectorizer and Kmean Model is Saved.
trained_model_path="path where the trained model and vectorizer are stored."

#Load the vectorizer and tranforming the test file.
loaded_vectorizer = joblib.load(trained_model_path + "finalized_tfidf_vectorizer")
tfidf_result = loaded_vectorizer.transform(filenames)

features=loaded_vectorizer.get_feature_names()

# load the model and predict for the vectorized test data.
loaded_model = joblib.load(trained_model_path + 'finalized_Kmeans_model')
labels = loaded_model.predict(tfidf_result)


#to print the classification of all the documents and also the top 10 features of the document.
l=-1
for i in os.listdir(test_document_path):
    l=l+1
    print("The test document %s belongs to cluster %d.\n" %(i,labels[l,]))
    tf=[os.path.join(test_document_path,i)]
    vectorizer = TfidfVectorizer(input='filename', 
                             max_features=10,
                             token_pattern='(?u)\\b[a-zA-Z]\\w{2,}\\b',
                             stop_words='english',
                             ngram_range=(1, 2))
    tf_vectors=vectorizer.fit_transform(tf)
    tf_features = vectorizer.get_feature_names()
    print ("The Top 10 features for document %s are\n %s\n\n" %(i,tf_features))

# Just for the fun of finding out the total elapsed time.
end = time.time()
elapsed = end - start
print "Time taken for predicting the class and getting top features of new test documents: ", elapsed, "seconds."
