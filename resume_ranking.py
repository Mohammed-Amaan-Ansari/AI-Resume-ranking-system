# Import required libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import re
import os
from PyPDF2 import PdfReader
from docx import Document

# Download NLTK stopwords
#nltk.download('stopwords')

# Step 1: Extract Text from Different File Formats
def extract_text_from_file(file_path):
    """
    Extract text from a file based on its format (PDF, DOCX, or TXT).
    """
    if file_path.endswith('.pdf'):
        # Extract text from PDF
        reader = PdfReader(file_path)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text
    elif file_path.endswith('.docx'):
        # Extract text from DOCX
        doc = Document(file_path)
        text = '\n'.join([para.text for para in doc.paragraphs])
        return text
    elif file_path.endswith('.txt'):
        # Extract text from TXT
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    else:
        raise ValueError("Unsupported file format. Please upload a PDF, DOCX, or TXT file.")

# Step 2: Preprocess Text
def preprocess_text(text):
    """
    Preprocess text by converting to lowercase, removing special characters, and removing stopwords.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text

# Step 3: Load Job Description
def load_job_description(job_desc_path):
    """
    Load and preprocess the job description.
    """
    with open(job_desc_path, 'r', encoding='utf-8') as file:
        job_desc_text = file.read()
    job_desc_processed = preprocess_text(job_desc_text)
    return job_desc_processed

# Step 4: Rank Resumes
def rank_resumes(resumes, job_desc_processed):
    """
    Rank resumes based on their similarity to the job description.
    """
    # Combine resumes and job description text
    all_texts = resumes['Processed_Text'].tolist() + [job_desc_processed]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Split the matrix back into resumes and job description
    resumes_tfidf = tfidf_matrix[:len(resumes)]
    job_desc_tfidf = tfidf_matrix[-1]
    
    # Calculate cosine similarity between resumes and job description
    similarity_scores = cosine_similarity(resumes_tfidf, job_desc_tfidf.reshape(1, -1))
    
    # Add similarity scores to the resumes dataframe
    resumes['Similarity_Score'] = similarity_scores
    
    # Rank resumes based on similarity scores
    ranked_resumes = resumes.sort_values(by='Similarity_Score', ascending=False)
    
    return ranked_resumes

# Step 5: Main Function
def main():
    """
    Main function to execute the resume screening and ranking system.
    """
    # Load job description
    job_desc_path = 'job_description.txt'  # Path to job description file
    job_desc_processed = load_job_description(job_desc_path)
    
    # Get resume files from user
    resume_files = []
    print("Welcome to the AI-powered Resume Screening and Ranking System!")
    print("Please upload your resume files (PDF, DOCX, or TXT). Enter 'done' when finished.")
    
    while True:
        file_path = input("Enter the path to a resume file (or 'done' to finish): ").strip()
        if file_path.lower() == 'done':
            break
        if not os.path.exists(file_path):
            print("File not found. Please try again.")
            continue
        resume_files.append(file_path)
    
    # Extract and preprocess text from resumes
    resumes = []
    for file_path in resume_files:
        text = extract_text_from_file(file_path)
        processed_text = preprocess_text(text)
        resumes.append({'File_Path': file_path, 'Processed_Text': processed_text})
    
    # Convert to DataFrame
    resumes_df = pd.DataFrame(resumes)
    
    # Rank resumes
    ranked_resumes = rank_resumes(resumes_df, job_desc_processed)
    
    # Display results
    print("\nRanked Resumes:")
    print(ranked_resumes[['File_Path', 'Similarity_Score']])
    
    # Save results to a CSV file
    ranked_resumes.to_csv('ranked_resumes.csv', index=False)
    print("\nRanked resumes have been saved to 'ranked_resumes.csv'.")

# Run the program
if __name__ == "__main__":
    main()
