from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = FastAPI()
data = pd.read_csv("Notebook/test_data.csv")
user = pd.read_csv("Notebook/user_data.csv")
rating = pd.read_csv("Notebook/ratings_test.csv")

class SearchRecommendation:
    def __init__(self, data):
        self.data = data
        self.data['processed_name'] = self.data['name'].str.lower()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.data['processed_name'])

    def search(self, query, top_n=20):
        query_vector = self.vectorizer.transform([query.lower()])
        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        results_df = pd.DataFrame({
            'id': self.data['id'].astype(str),
            'name': self.data['name'],
            'cosine_similarity': cosine_similarities
        })
        results_df = results_df.sort_values(by='cosine_similarity', ascending=False)
        return results_df[['id', 'name']].head(top_n).to_dict(orient='records')



search_model = SearchRecommendation(data)

class SearchQuery(BaseModel):
    query: str
    top_n: int = 10

class RecommendTrendModel:
    def __init__(self, user_data, ratings_data, product_data):
        self.user_data = user_data
        self.ratings_data = ratings_data
        self.product_data = product_data

    @staticmethod
    def calculate_similarity(user1, user2):
        gender_sim = 1 if user1['gender'] == user2['gender'] else 0
        age_sim = 1 - abs(user1['age'] - user2['age']) / 100
        if user1['city'] == user2['city']:
            location_sim = 1  # Same city
        elif user1['country'] == user2['country']:
            location_sim = 0.5
        else:
            location_sim = 0
        total_similarity = 0.2 * gender_sim + 0.6 * age_sim + 0.2 * location_sim
        return total_similarity

    def find_top5_similar_users(self, user_info):
        similarities = []
        for _, other_user in self.user_data.iterrows():
            similarity = self.calculate_similarity(user_info, other_user)
            similarities.append((other_user['userID'], similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [user[0] for user in similarities[:5]]

    def collaborative_filtering_recommendations(self, user_info, top_n=10):
        top5_similar_users = self.find_top5_similar_users(user_info)
        filtered_ratings_data = self.ratings_data[self.ratings_data['userid'].isin(top5_similar_users)]
        user_item_matrix = filtered_ratings_data.pivot_table(
            index='userid',
            columns='productid',
            values='rating',
            aggfunc='mean'
        ).fillna(0)
        user_similarity = cosine_similarity(user_item_matrix)
        recommended_items = set()
        for target_user_id in top5_similar_users:
            try:
                target_user_index = user_item_matrix.index.get_loc(target_user_id)
            except KeyError:
                continue
            user_similarities = user_similarity[target_user_index]
            similar_users_indices = user_similarities.argsort()[::-1][1:]
            for user_index in similar_users_indices:
                rated_by_similar_user = user_item_matrix.iloc[user_index]
                not_rated_by_target_user = (rated_by_similar_user > 0) & (user_item_matrix.iloc[target_user_index] == 0)
                recommended_items.update(user_item_matrix.columns[not_rated_by_target_user][:top_n])
                if len(recommended_items) >= top_n:
                    break
        recommended_items_details = self.ratings_data[
            self.ratings_data['productid'].isin(recommended_items)
        ][['productid', 'rating']].drop_duplicates()
        recommended_items_with_names = recommended_items_details.merge(
        self.product_data[['id', 'name']],
        left_on='productid',
        right_on='id',
        how='inner')
        top_recommendations = recommended_items_with_names.sort_values(by='rating', ascending=False ).head(top_n)
        return top_recommendations[['productid', 'name']].values.tolist()

    def most_trending_products(self, top_n=20):
        trending_products = self.product_data.sort_values(
            by=['ratings', 'no_of_ratings'], ascending=False
        ).head(top_n)
        return trending_products[['id', 'name']].values.tolist()

    def recommend(self, user_info, top_n=20):
        collaborative_recommendations = self.collaborative_filtering_recommendations(user_info, top_n=top_n)
        trending_recommendations = self.most_trending_products(top_n=top_n)
        recommendations = collaborative_recommendations + trending_recommendations
        return recommendations

trend_recommendation_model=RecommendTrendModel(user,rating,data)
class TrendRecommendation(BaseModel):
    age:int
    gender:str
    city:str
    country:str

@app.post("/search")
def get_search_recommendations(search_query: SearchQuery):
    results = search_model.search(search_query.query, top_n=search_query.top_n)
    return {"results": results}
@app.post("/trend")
def get_trend_recommendations(trend_recommendations: TrendRecommendation):
    user_info = trend_recommendations.model_dump()
    results = trend_recommendation_model.recommend(user_info)
    return {"results": results}
@app.get("/")
def read_root():
    return {"message": "Welcome to the Search Recommendation API!"}
