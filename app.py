
from flask import Flask, render_template, request, jsonify
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

# Load saved ML artifacts
vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")
X = joblib.load(BASE_DIR / "tfidf_matrix.pkl")
df = pd.read_csv(BASE_DIR / "hisense_products.csv").fillna("")

# Validate dataset
required_columns = {"model", "category", "description"}

if not required_columns.issubset(df.columns):
    raise ValueError("Dataset is missing required columns.")

if X.shape[0] != len(df):
    raise ValueError("TF-IDF matrix and dataset row counts do not match.")


def search_hisense(query, top_k=5):
    query = str(query).strip()

    if not query:
        return []

    # Exact model match first
    normalized_query = "".join(query.upper().split())

    exact_matches = df[
        df["model"].astype(str).str.upper().str.replace(
            r"\s+", "", regex=True
        ) == normalized_query
    ]

    if not exact_matches.empty:
        results = exact_matches.copy()
        results["similarity"] = 1.0

        return results[
            ["model", "category", "description", "similarity"]
        ].to_dict(orient="records")

    # TF-IDF search
    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector, X
    ).flatten()

    indices = np.argsort(similarities)[::-1][:top_k]

    results = df.iloc[indices].copy()
    results["similarity"] = similarities[indices]

    # Remove weak matches
    results = results[results["similarity"] >= 0.15]

    return results[
        ["model", "category", "description", "similarity"]
    ].to_dict(orient="records")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json(silent=True) or {}
    query = data.get("query", "")

    if not isinstance(query, str):
        return jsonify({"error": "Invalid query"}), 400

    if len(query) > 200:
        return jsonify({"error": "Query is too long"}), 400

    results = search_hisense(query)

    return jsonify({
        "query": query,
        "count": len(results),
        "results": results
    })


@app.route("/api/products", methods=["GET"])
def products():
    return jsonify({
        "count": len(df),
        "products": df[
            ["model", "category", "description"]
        ].to_dict(orient="records")
    })


if __name__ == "__main__":
    app.run(debug=True)
