from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from pydantic import BaseModel
from typing import Optional


app = FastAPI()


client = MongoClient('mongodb://localhost:27017/')
db = client['internwork']  
collection = db['items']


class Item(BaseModel):
    name: str
    description: str
    price: float

# Home endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to my FastAPI app with MongoDB!"}

# Endpoint to get an item by its ID
@app.get("/items/{item_id}")
def read_item(item_id: int):
    item = collection.find_one({"_id": item_id})
    if item:
        return {"item_id": item["_id"], "name": item["name"], "description": item["description"], "price": item["price"]}
    raise HTTPException(status_code=404, detail="Item not found")

# Endpoint to create a new item
@app.post("/items/")
def create_item(item_id: int, item: Item):
    if collection.find_one({"_id": item_id}):
        raise HTTPException(status_code=400, detail="Item ID already exists")
    collection.insert_one({"_id": item_id, "name": item.name, "description": item.description, "price": item.price})
    return {"item_id": item_id, "name": item.name, "description": item.description, "price": item.price}

# Endpoint to update an existing item
@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    result = collection.update_one(
        {"_id": item_id},
        {"$set": {"name": item.name, "description": item.description, "price": item.price}}
    )
    if result.matched_count:
        return {"item_id": item_id, "name": item.name, "description": item.description, "price": item.price}
    raise HTTPException(status_code=404, detail="Item not found")

# Endpoint to delete an item
@app.delete("/items/{item_id}")
def delete_item(item_id: int):
    result = collection.delete_one({"_id": item_id})
    if result.deleted_count:
        return {"message": "Item deleted"}
    raise HTTPException(status_code=404, detail="Item not found")
