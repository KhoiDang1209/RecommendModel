import pickle
from pathlib import Path
from
BASE_DIR = Path(__file__).resolve(strict=True).parent

# Load the pickled model
try:
    with open(BASE_DIR / 'search_recommendation_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise FileNotFoundError(
        "The 'search_recommendation_model.pkl' file was not found. Ensure it is in the same directory as model.py.")
except Exception as e:
    raise RuntimeError(f"An error occurred while loading the model: {e}")


# Recommendation function
def recommend(query, top_n=20):
    """
    Get recommendations based on a search query.

    :param query: The search text (string).
    :param top_n: Number of recommendations to return.
    :return: List of recommendations.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("The search query must be a non-empty string.")

    try:
        # Assuming the method is named `search`
        recommendations = model.search(query, top_n=top_n)
        return recommendations
    except Exception as e:
        raise RuntimeError(f"An error occurred while getting recommendations: {e}")