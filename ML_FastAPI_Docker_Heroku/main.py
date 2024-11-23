from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from ML_FastAPI_Docker_Heroku.model import recommend
app = FastAPI()
class Item(BaseModel):
    text: str
class Recommendation(BaseModel):
    list: str
@app.get("/")
async def home():
    return {"This is Home"}
@app.post("/recommend",response_model=Recommendation)
def search_recommendations(item: Item):
    recommendations=recommend(item)
    return {"Search RecommendationsL" :recommendations}
