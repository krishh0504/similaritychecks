from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.linalg import Vectors, DenseVector
from pyspark.ml import Pipeline
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType

# Initialize Spark session
spark = SparkSession.builder.master("local[*]").appName("Advanced Similarity Search").getOrCreate()

# Load the dataset (adjust the file path as necessary)
df = spark.read.csv("dataset.csv", header=True, inferSchema=True)

# Show the first few records to verify the dataset
df.show(5, truncate=False)

# Text Preprocessing: Tokenization and TF-IDF
tokenizer = Tokenizer(inputCol="description", outputCol="words")
hashingTF = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=1000)
idf = IDF(inputCol="raw_features", outputCol="features")

# Build the pipeline
pipeline = Pipeline(stages=[tokenizer, hashingTF, idf])

# Fit and transform the data
model = pipeline.fit(df)
result = model.transform(df)

# Cache the result to speed up further operations
result.cache()

# Show the result with TF-IDF features
result.select("product_id", "description", "features").show(5, truncate=False)

# Define a UDF to convert sparse vector to dense vector
def to_dense_udf(sparse_vector):
    return sparse_vector.toArray().tolist()

# Register the UDF
to_dense = F.udf(to_dense_udf, ArrayType(DoubleType()))

# Apply the UDF to convert sparse vector to dense vector
result = result.withColumn("dense_features", to_dense("features"))

# Show the result with dense features
result.select("product_id", "description", "dense_features").show(5, truncate=False)

# Define similarity metrics

# Cosine Similarity (Efficient version)
def cosine_similarity(df):
    similarities = []
    features = df.select("dense_features").rdd.map(lambda row: row[0]).collect()
    
    for i, vec1 in enumerate(features):
        for j, vec2 in enumerate(features):
            if i < j:
                similarity = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
                similarities.append((i + 1, j + 1, similarity))
    
    return similarities

# Euclidean Distance (Efficient version)
def euclidean_distance(df):
    distances = []
    features = df.select("dense_features").rdd.map(lambda row: row[0]).collect()
    
    for i, vec1 in enumerate(features):
        for j, vec2 in enumerate(features):
            if i < j:
                distance = float(np.linalg.norm(np.array(vec1) - np.array(vec2)))
                distances.append((i + 1, j + 1, distance))
    
    return distances

# Jaccard Similarity (Efficient version)
def jaccard_similarity(df):
    similarities = []
    features = df.select("dense_features").rdd.map(lambda row: row[0]).collect()
    
    for i, vec1 in enumerate(features):
        for j, vec2 in enumerate(features):
            if i < j:
                intersection = float(np.sum(np.minimum(vec1, vec2)))
                union = float(np.sum(np.maximum(vec1, vec2)))
                similarity = intersection / union if union != 0 else 0
                similarities.append((i + 1, j + 1, similarity))
    
    return similarities

# Manhattan Distance (Efficient version)
def manhattan_distance(df):
    distances = []
    features = df.select("dense_features").rdd.map(lambda row: row[0]).collect()
    
    for i, vec1 in enumerate(features):
        for j, vec2 in enumerate(features):
            if i < j:
                distance = float(np.sum(np.abs(np.array(vec1) - np.array(vec2))))
                distances.append((i + 1, j + 1, distance))
    
    return distances

# Hamming Distance (for binary data)
def hamming_distance(df):
    distances = []
    features = df.select("dense_features").rdd.map(lambda row: row[0]).collect()
    
    for i, vec1 in enumerate(features):
        for j, vec2 in enumerate(features):
            if i < j:
                distance = float(np.sum(np.abs(np.array(vec1) - np.array(vec2))) != 0)
                distances.append((i + 1, j + 1, distance))
    
    return distances

# Execute similarity algorithms and measure time
start_time = time.time()
cosine_sim = cosine_similarity(result)
cosine_time = time.time() - start_time
print("Cosine Similarity Time:", cosine_time)

start_time = time.time()
euclidean_dist = euclidean_distance(result)
euclidean_time = time.time() - start_time
print("Euclidean Distance Time:", euclidean_time)

start_time = time.time()
jaccard_sim = jaccard_similarity(result)
jaccard_time = time.time() - start_time
print("Jaccard Similarity Time:", jaccard_time)

start_time = time.time()
manhattan_dist = manhattan_distance(result)
manhattan_time = time.time() - start_time
print("Manhattan Distance Time:", manhattan_time)

start_time = time.time()
hamming_dist = hamming_distance(result)
hamming_time = time.time() - start_time
print("Hamming Distance Time:", hamming_time)

# Plot execution times
metrics = ['Cosine', 'Jaccard', 'Euclidean', 'Manhattan', 'Hamming']
times = [cosine_time, jaccard_time, euclidean_time, manhattan_time, hamming_time]

plt.bar(metrics, times)
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time of Similarity Metrics')
plt.show()