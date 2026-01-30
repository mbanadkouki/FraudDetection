import pyspark
from xgboost import spark as xgb_spark

# Initialize a local Spark session
spark = pyspark.sql.SparkSession.builder.master("local[*]").appName("FraudCheck").getOrCreate()

print(f"Spark Version: {spark.version}")
print("XGBoost Spark Integration: Ready")