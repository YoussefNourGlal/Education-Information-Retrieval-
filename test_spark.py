## file da test bs 34an tt2kd en spark sh8al 3la lap bta3k 
from pyspark.sql import SparkSession

# Create Spark Session
spark = (
    SparkSession.builder
    .appName("EducationProjectTest")
    .master("local[*]")
    .getOrCreate()
)

print("Spark Started Successfully!")

# Create a simple DataFrame
data = [
    (1, "online learning"),
    (2, "student assessment"),
    (3, "education technology")
]

columns = ["document_id", "content"]

df = spark.createDataFrame(data, columns)

# Display DataFrame
df.show()

# Count documents
print("Number of documents:", df.count())

# Stop Spark
spark.stop()
