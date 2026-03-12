from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URL = "mongodb://localhost:27017"

client = AsyncIOMotorClient(MONGO_URL)

database = client.finops_db

users_collection = database["users"]
organizations_collection = database["organizations"]
transactions_collection = database["transactions"]
budgets_collection = database["budgets"]
alerts_collection = database["alerts"]
audit_logs_collection = database["audit_logs"]