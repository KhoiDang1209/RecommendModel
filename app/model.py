import pickle
from pathlib import Path
BASE_DIR = Path(__file__).resolve(strict=True).parent

# Load the pickled model
try:
    with open(BASE_DIR / 'search_recommendation.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise FileNotFoundError(
        "The 'search_recommendation.pkl' file was not found. Ensure it is in the same directory as model.py.")
except Exception as e:
    raise RuntimeError(f"An error occurred while loading the model: {e}")


# Recommendation function
def recommend(query, top_n=20):

    if not isinstance(query, str) or not query.strip():
        raise ValueError("The search query must be a non-empty string.")

    try:
        # Assuming the method is named `search`
        recommendations = model.search(query, top_n=top_n)
        return recommendations
    except Exception as e:
        raise RuntimeError(f"An error occurred while getting recommendations: {e}")