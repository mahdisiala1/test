from pymongo import MongoClient

uri = "mongodb+srv://mahdisiala3:man7ebbech@cluster0.cmzo8fb.mongodb.net/"

try:
    client = MongoClient(uri)
    client.admin.command("ping")
    print("✅ Connection successful")
except Exception as e:
    print("❌ Connection failed:", e)
