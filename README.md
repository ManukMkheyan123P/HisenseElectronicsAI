Hisense Product Search 🔎
A lightweight product-search prototype built with Python, TF-IDF, and cosine similarity. Enter a Hisense model number or a short description to find the closest matching products in a small sample catalog.
Features
- Search by model number, such as 55U7KQ or a partial code like 55U7K.
- Search by product characteristics, such as Mini LED 65 inch.
- Return the top matching products with similarity scores.
- Run an interactive search loop in Jupyter Notebook.
- Export the vectorizer, TF-IDF matrix, and catalog for reuse in another application.
Tech Stack
- Python
- pandas, NumPy
- scikit-learn (TfidfVectorizer, cosine_similarity)
- joblib
- Jupyter Notebook
How It Works
1. A sample catalog is loaded into a pandas DataFrame.
2. The brand, model, category, and description are combined into a lowercase search_text field.
3. TfidfVectorizer(analyzer="char", ngram_range=(2, 4)) converts the catalog into character-level TF-IDF features.
4. The user query is transformed using the same vectorizer.
5. Cosine similarity ranks the catalog entries; search_hisense(query, top_k=3) returns the best matches.
Note: This is a TF-IDF similarity-search system, not a supervised machine-learning model or a generative AI chatbot. It only searches the products included in the catalog.
Dataset
The notebook contains 5 example Hisense TVs with these fields:
Field	Description
model	Product model number
category	Product category
brand	Manufacturer
description	Short product specification


The sample models are 55U7KQ, 55U6KQ, 55A6K, 65U7KQ, and 50A6K. Product descriptions are prototype data, not a verified live catalog.
Installation
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
pip install pandas numpy matplotlib scikit-learn joblib notebook
jupyter notebook Hisense.ipynb
Replace the placeholders with your repository URL and folder name. Run the notebook cells in order.
Usage
Search directly in the notebook:
search_hisense("Hisense 55U7KQ")
search_hisense("55U7K")
search_hisense("Mini LED 65 inch")
Results contain model, category, description, and similarity (a cosine similarity score, not a confidence probability).

The notebook also includes a terminal-style input loop. Type exit to stop. In that loop, the top match is treated as not found when its similarity is below 0.15; this is a prototype heuristic, not a validated threshold.
Exported Files
Running the final notebook cell generates:
Hisense.ipynb
hisense_products.csv
tfidf_vectorizer.pkl
tfidf_matrix.pkl
- hisense_products.csv — catalog with its search text.
- tfidf_vectorizer.pkl — fitted TF-IDF vectorizer.
- tfidf_matrix.pkl — vectorized catalog matrix.
These files can be loaded with pandas.read_csv() and joblib.load() in a separate Python or Flask application. Only load .pkl files from trusted sources.
Limitations and Next Steps
- The catalog currently has only five sample TV products.
- Search results are based on text similarity, so the closest result may still be incorrect.
- No live product updates, external API, or deployed website are included in this notebook.
- Possible improvements: expand the dataset, add exact-model matching, test queries systematically, and build a Flask web interface.
Project Status
Prototype / educational project — demonstrating information retrieval with TF-IDF and cosine similarity.
