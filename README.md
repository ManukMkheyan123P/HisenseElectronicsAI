
# Hisense AI Product Search

A Python-based product search system for Hisense electronics using Natural Language Processing (NLP), TF-IDF Vectorization, and Cosine Similarity.

The project allows users to search for Hisense products by model name, specifications, or keywords.

## Features

- Search Hisense products by model number
- Search products using natural language
- Find similar products using TF-IDF
- Calculate similarity scores
- Display product specifications
- Handle unknown product queries
- Save the trained vectorizer and product dataset

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity
- Joblib
- Jupyter Notebook

## How It Works

The system uses TF-IDF to convert product descriptions into numerical vectors.

### 1. Data Preparation

Product information includes:

- Brand
- Model
- Category
- Description

### 2. TF-IDF Vectorization

```python
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
    lowercase=True
)

X = vectorizer.fit_transform(df["search_text"])
```

Character-level n-grams help identify similar model names even when the user enters an incomplete model number.

### 3. Cosine Similarity

```python
query_vector = vectorizer.transform([query])

similarities = cosine_similarity(
    query_vector, X
).flatten()
```

The system calculates similarity scores and returns the most relevant products.

## Example

Input:

```text
Hisense 55U7KQ
```

Example output:

```text
HISENSE PRODUCT INFORMATION
--------------------------
Model: 55U7KQ
Category: TV
Description: 55 inch 4K Mini LED Smart TV, VIDAA, 144Hz
Similarity: 100.0 %
```

Other supported queries:

```text
55U7K
Mini LED 65 inch
Hisense 4K TV
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn joblib matplotlib jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook Hisense.ipynb
```

## Saving the Model

The project uses Joblib to save the fitted TF-IDF vectorizer and matrix.

```python
import joblib

joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(X, "tfidf_matrix.pkl")

df.to_csv("hisense_products.csv", index=False)
```

Saved files:

```text
Hisense.ipynb
tfidf_vectorizer.pkl
tfidf_matrix.pkl
hisense_products.csv
README.md
```

## Machine Learning Approach

This project uses an unsupervised text similarity approach.

Unlike supervised machine learning models, TF-IDF does not require labeled training data.

The search engine works by:

1. Converting product descriptions into TF-IDF vectors.
2. Transforming the user's query into the same vector space.
3. Calculating cosine similarity.
4. Ranking products by similarity score.
5. Returning the most relevant results.

## Future Improvements

- Expand the Hisense product dataset
- Add refrigerators, washing machines, and air conditioners
- Build a Flask web application
- Implement multilingual search
- Integrate a product database
- Add semantic search using Sentence Transformers
- Deploy the application online

## Project Status

Prototype — functional product search system with a small sample dataset.

## Author

**Manuk Mkheyan**

Python | Machine Learning | Data Analytics

GitHub: https://github.com/ManukMkheyan123P

