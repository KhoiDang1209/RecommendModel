import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SearchRecommendation:

    def __init__(self, csv_path=None):
        # Default CSV path if none is provided
        self.csv_path = csv_path or 'test_data.csv'

        # Load the data
        self.data = pd.read_csv(self.csv_path, engine='python', encoding='utf-8')

        # Preprocess the data
        self.data['processed_name'] = self.data['name'].str.lower()

        # Fit the TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.data['processed_name'])

    def search(self, query, top_n=20):
        # Transform the query to a vector
        query_vector = self.vectorizer.transform([query.lower()])

        # Compute cosine similarities
        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        results_df = pd.DataFrame({
            'id': self.data['id'].astype(str),  # Ensure ID is treated as string for consistency
            'name': self.data['name'],
            'cosine_similarity': cosine_similarities
        })
        results_df = results_df.sort_values(by='cosine_similarity', ascending=False)

        # Select only the 'id' and 'name' columns and convert to list of dictionaries
        results_list = results_df[['id', 'name']].head(top_n).to_dict(orient='records')

        return results_list


# Create the model instance
model = SearchRecommendation(csv_path='/Notebook/test_data.csv')

# Save the model as a pickle file
with open('search_recommendation.pkl', 'wb') as file:
    pickle.dump(model, file)
