#
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession


# Generate synthetic dataset
data = [(Vectors.dense([2.0, 3.0, 5.0, 7.0, 11.0]),),
        (Vectors.dense([3.0, 5.0, 7.0, 11.0, 13.0]),),
        (Vectors.dense([5.0, 7.0, 11.0, 13.0, 17.0]),)]
df = spark.createDataFrame(data, ["features"])


# Apply PCA
pca = PCA(k=2, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(df)
result = model.transform(df).select("pcaFeatures")


# Show result
result.show(truncate=False)
