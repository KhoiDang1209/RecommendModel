import pandas as pd
from pymongo.mongo_client import MongoClient
uri = "mongodb+srv://khoibk123123:khoibk123@recommenddtb.4in6a.mongodb.net/?retryWrites=true&w=majority&appName=RecommendDTB"
# Create a new client and connect to the server
client = MongoClient(uri)
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
db = client["Recommend"]

# Collections
user_collection = db["User"]
rating_collection = db["Rating"]
product_collection = db["Product"]
association_collection = db["Association"]
user_df = pd.DataFrame(list(user_collection.find()))
rating_df = pd.DataFrame(list(rating_collection.find()))
product_df = pd.DataFrame(list(product_collection.find()))
association_df = pd.DataFrame(list(association_collection.find()))
print(association_df.head(5))
print(user_df.head(5))
print(rating_df.head(5))
print(product_df.head(5))
