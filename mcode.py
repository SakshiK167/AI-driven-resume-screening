import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# dataset of resumes
# Make sure to update the path to your CSV file
df = pd.read_csv(r"C:\Users\saksh\OneDrive\Desktop")

# Check the first few rows of the dataframe to understand its structure
print(df.head())

# Extract resumes into a list
resumes = df['Category'].tolist()  # Adjust 'resume' to the actual column name

# Sample Job Description
job_description = "Looking for a software engineer with experience in Python and machine learning."

# Step 1: Preprocessing
vectorizer = TfidfVectorizer()
documents = [job_description] + resumes
tfidf_matrix = vectorizer.fit_transform(documents)

# Step 2: Calculate similarity
cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

# Step 3: Prepare results
results = pd.DataFrame({
    'Resume': resumes,
    'Score': cosine_similarities
})

# Sort by score
results = results.sort_values(by='Score', ascending=False).reset_index(drop=True)

# Display results
print("Matching Results:")
print(results)