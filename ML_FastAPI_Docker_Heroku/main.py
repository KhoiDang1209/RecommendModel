from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = FastAPI()
data = pd.read_csv("Notebook/test_data.csv")
user = pd.read_csv("Notebook/user_data.csv")
rating = pd.read_csv("Notebook/ratings_test.csv")
rule = pd.read_csv("Notebook/association_rules.csv")

class SearchRecommendation:
    def __init__(self, product):
        self.product = product
        self.product['processed_name'] = self.product['name'].str.lower()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.product['processed_name'])

    def search(self, query, top_n=20):
        query_vector = self.vectorizer.transform([query.lower()])
        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        results_df = pd.DataFrame({
            'id': self.product['id'].astype(str),
            'name': self.product['name'],
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
        return [user_list[0] for user_list in similarities[:5]]

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

class InterestRecommendationModel:
    def __init__(self, user_data, ratings_data, product_data):
        # Store the datasets in attributes
        self.user_data = user_data
        self.ratings_data = ratings_data
        self.product_data = product_data

    @staticmethod
    def calculate_similarity(user1, user2):
        gender_sim = 1 if user1['gender'] == user2['gender'] else 0
        age_sim = 1 - abs(user1['age'] - user2['age']) / 100
        if user1['city'] == user2['city']:
            location_sim = 1
        elif user1['country'] == user2['country']:
            location_sim = 0.5
        else:
            location_sim = 0
        total_similarity = 0.2 * gender_sim + 0.6 * age_sim + 0.2 * location_sim
        return total_similarity

    def find_top5_similar_users(self, user_info):
        similarities = []
        for i, other_user in self.user_data.iterrows():
            similarity = self.calculate_similarity(user_info, other_user)
            similarities.append((other_user['userID'], similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [user_list[0] for user_list in similarities[:5]]

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

    def highest_rated_products_by_interest(self, user_info, top_n=20):
        user_interest = list(map(int, user_info['interest'].split(',')))
        filtered_products = self.product_data[self.product_data['main_category_encoded'].isin(user_interest)]
        top_products = filtered_products.sort_values(
            by=['ratings', 'no_of_ratings'], ascending=False
        ).head(top_n)
        return top_products[['id', 'name']].values.tolist()

    def recommend(self, user_info, top_n=20):
        collaborative_recommendations = self.collaborative_filtering_recommendations(user_info, top_n=top_n)
        interest_based_recommendations = self.highest_rated_products_by_interest(user_info, top_n=top_n)

        recommendations = collaborative_recommendations+interest_based_recommendations
        return recommendations

interest_recommendation_model=InterestRecommendationModel(user,rating,data)
class InterestRecommendation(BaseModel):
    age: int
    gender: str
    city: str
    country: str
    interest: str

class AssociationRecommendationModel:
    def __init__(self, product, rules):
        self.product = product
        self.rules = rules

    def get_top_associated_categories(self, item_ids, n=3):
        """Get top associated categories for a list of item IDs."""
        top_associated_categories = set()

        for item_id in item_ids:
            # Retrieve item information
            item_info = self.product[self.product['id'] == item_id]
            if item_info.empty:
                print(f"Item with ID {item_id} not found in dataset.")
                continue

            # Construct item category combination
            item_category = f"{item_info.iloc[0]['main_category']} - {item_info.iloc[0]['sub_category']}"
            # Filter rules to find relevant antecedents
            relevant_rules = self.rules[self.rules['antecedents'].apply(lambda x: item_category in x)]
            # If no rules, continue
            if relevant_rules.empty:
                continue

            # Sort rules and collect associated categories
            relevant_rules = relevant_rules.sort_values(by='lift', ascending=False).head(n)
            for _, row in relevant_rules.iterrows():
                for category_set in [row['antecedents'], row['consequents']]:
                    if isinstance(category_set, str):  # Ensure it's a string
                        for category in category_set.split(", "):  # Split by delimiter
                            if category != item_category:  # Exclude the original category
                                top_associated_categories.add(category)
        # Return the top N categories as a list
        top_categories_list = list(top_associated_categories)[:n]
        return top_categories_list


    def get_top_rated_items(self, associated_categories, top_n=5):
        """Get top-rated items from the associated categories."""
        top_items = []
        for category in associated_categories:
            try:
                # Split the main and subcategories
                main_category, sub_category = category.split(" - ")
            except ValueError:
                print(f"Invalid category format: {category}")
                continue
            # Filter items within the given category
            category_items = self.product[
                (self.product['main_category'] == main_category) &
                (self.product['sub_category'] == sub_category)
            ]
            # Get top items based on ratings
            top_rated_items = category_items.sort_values(by='ratings', ascending=False).head(top_n)
            top_items.extend(top_rated_items['id'].tolist())
        return top_items

    def recommend(self, item_ids, associated_n=3, top_n=5):
        # Get top associated categories for the input items
        top_associated_categories = self.get_top_associated_categories(item_ids, n=associated_n)
        # Get highest-rated items from the associated categories
        top_rated_items = self.get_top_rated_items(top_associated_categories, top_n=top_n)
        return top_rated_items

associate_model=AssociationRecommendationModel(data,rule)
class AssociationRecommendation(BaseModel):
    item_ids: list[str]

class ItemRecommendationModel:
    def __init__(self, product):
        self.product = product
        self.product['processed_name'] = self.product['name'].str.lower()

        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.product['processed_name'])

    def find_similar_items(self, item_id, top_n=20):
        if item_id not in self.product['id'].values:
            print(f"Item ID {item_id} not found in dataset.")
            return []

        item_index = self.product[self.product['id'] == item_id].index[0]
        item_vector = self.tfidf_matrix[item_index]
        cosine_similarities = cosine_similarity(item_vector, self.tfidf_matrix).flatten()
        similar_indices = cosine_similarities.argsort()[-top_n-1:-1][::-1]
        similar_items = self.product.iloc[similar_indices][['id','name',]]

        similar_items_list = similar_items.to_dict(orient='records')

        return similar_items_list

class ItemRecommendation(BaseModel):
    id: str

item_recommendation_model = ItemRecommendationModel(data)
@app.post("/search")
async def get_search_recommendations(search_query: SearchQuery):
    results = search_model.search(search_query.query, top_n=search_query.top_n)
    return {"results": results}

@app.post("/trend")
async def get_trend_recommendations(trend_recommendations: TrendRecommendation):
    user_info = trend_recommendations.model_dump()
    results = trend_recommendation_model.recommend(user_info)
    return {"results": results}

@app.post("/interest")
async def get_interest_recommendations(interest_recommendations: InterestRecommendation):
    user_info=interest_recommendations.model_dump()
    results = interest_recommendation_model.recommend(user_info)
    return {"results": results}

@app.post("/item")
async def get_item_recommendation(item_recommendation: ItemRecommendation):
    item_id=item_recommendation.id
    result=item_recommendation_model.find_similar_items(item_id)
    return {"results": result}

@app.post("/association")
async def get_association_recommendations(associate_instances: AssociationRecommendation):
    item_ids = associate_instances.item_ids
    results = associate_model.recommend(item_ids)
    return {"results": results}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Search Recommendation API!"}
