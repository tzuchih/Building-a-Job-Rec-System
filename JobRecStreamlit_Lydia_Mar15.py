#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import io
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_text
from docx import Document

# 🔹 Set page layout to wide
st.set_page_config(page_title="NLP Team12 Job Recommendation System", layout="wide")

# 🔹 Load NLTK Stopwords & Lemmatizer
nltk_stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

@st.cache_data
def load_data():
    """Loads and processes the job dataset."""
    job_data = pd.read_csv('/Users/tzuchihsu/UCI/2025 Winter/NLP/LinkedIn Job Rec/data cleaning/linkedin_job_posts_skills.csv')
    job_data['job_text'] = job_data['job_title'] + " " + job_data['job_skills']
    job_data['clean_text'] = job_data['job_text'].apply(clean_text)
    return job_data

@st.cache_resource
def get_vectorizer_and_vectors(job_data):
    """Caches the TF-IDF vectorizer and job text embeddings."""
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), smooth_idf=True, sublinear_tf=True)
    
    if not job_data.empty:
        job_vectors = vectorizer.fit_transform(job_data['clean_text'])
    else:
        job_vectors = None  # Prevents errors if no jobs exist
    
    return vectorizer, job_vectors

def clean_text(text):
    """Cleans text by removing numbers, punctuation, and stopwords."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)  
    text = text.translate(str.maketrans('', '', string.punctuation))  
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in nltk_stopwords]
    return " ".join(words)

def extract_resume_text(uploaded_file):
    """Extracts text from resume (PDF or DOCX)."""
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        if uploaded_file.name.endswith('.pdf'):
            return extract_text(io.BytesIO(file_bytes))  
        elif uploaded_file.name.endswith('.docx'):
            doc = Document(io.BytesIO(file_bytes))  
            return "\n".join([para.text for para in doc.paragraphs])
    return ""

def get_top_jobs_with_match_score(resume_text, job_data, vectorizer, job_vectors, top_n=5):
    """Finds the best job matches using TF-IDF and cosine similarity, ensuring valid indices."""
    if job_vectors is None or job_data.empty:
        return pd.DataFrame()  # Return empty DataFrame if no jobs exist

    resume_vector = vectorizer.transform([clean_text(resume_text)])
    similarity_scores = cosine_similarity(resume_vector, job_vectors).flatten()
    
    max_score = max(similarity_scores) if max(similarity_scores) > 0 else 1  
    match_scores = (similarity_scores / max_score * 100).round(2)  

    # Get sorted indices based on similarity scores
    top_indices = np.argsort(similarity_scores)[::-1][:top_n]

    # Ensure valid indices (convert filtered job data into reindexed DataFrame)
    job_data = job_data.reset_index(drop=True)  # Reset index to match TF-IDF results
    valid_indices = [i for i in top_indices if i < len(job_data)]  # Ensure valid indices
    
    if not valid_indices:
        return pd.DataFrame()  # Return empty DataFrame if no valid matches found

    # Select the top matching jobs
    top_jobs = job_data.iloc[valid_indices].copy()
    top_jobs["match_score"] = match_scores[valid_indices]  

    return top_jobs

def skill_gap_analysis(resume_text, job_skills):
    """Compares resume skills with job-required skills and finds missing ones."""
    resume_words = set(clean_text(resume_text).split())
    job_words = set(clean_text(job_skills).split())
    missing_skills = job_words - resume_words
    matched_skills = job_words & resume_words  # Get common skills
    return matched_skills, missing_skills

def generate_cover_letter(job):
    """Generates a structured cover letter based on job details."""
    return f"""
    Dear Hiring Manager,

    I am excited to apply for the {job['job_title']} position at {job['company']}. 
    With my background in [Your Relevant Experience], I am eager to bring my skills in {job['job_skills']} to your team.

    In my previous role as [Your Most Relevant Job Title] at [Previous Company], 
    I [describe an achievement that demonstrates a key skill relevant to the job, using quantifiable impact if possible]. 
    This experience has equipped me with the ability to [explain how your skills will add value to the new role].

    What excites me most about this opportunity at {job['company']} is 
    [mention something specific about the company, its culture, mission, or projects that align with your goals]. 
    I am eager to contribute by [mention how you can help solve a challenge or add value based on your experience].

    I would love the opportunity to discuss how my skills and experience align with your needs. 
    Please feel free to contact me at your convenience.

    Best regards,  
    
    """


# In[ ]:


st.title("💼 NLP Team12 Job Recommendation System")
st.write("Upload your resume to get started!")

if "top_jobs" not in st.session_state:
    st.session_state.top_jobs = pd.DataFrame()  # ✅ Ensures it's always initialized

uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
selected_job = None  

if uploaded_file is not None:
    with st.status("🔄 Extracting text from resume...", expanded=True):
        resume_text = extract_resume_text(uploaded_file)
        st.write("✅ Resume text extracted!")

    job_data = load_data()

    with st.form("filter_form"):
        selected_country = st.multiselect("Select Country:", ["Australia", "Canada", "United Kingdom", "United States"], default=["United States"])
        selected_job_level = st.multiselect("Select Job Level:", ["Associate", "Mid senior"], default=["Mid senior"])
        selected_job_type = st.multiselect("Select Job Type:", ["Hybrid", "Onsite", "Remote"], default=["Remote"])
        submitted = st.form_submit_button("Find Matching Jobs")

    if submitted:
        filtered_jobs = job_data[
            (job_data["search_country"].isin(selected_country)) &
            (job_data["job_level"].isin(selected_job_level)) &
            (job_data["job_type"].isin(selected_job_type))
        ]
        vectorizer, job_vectors = get_vectorizer_and_vectors(filtered_jobs)
        st.session_state.top_jobs = get_top_jobs_with_match_score(resume_text, filtered_jobs, vectorizer, job_vectors)

col1, col2 = st.columns(2)

with col1:
    if not st.session_state.top_jobs.empty:
        st.write("### 🎯 Top 5 Matching Jobs:")
        job_options = [f"{job['job_title']} at {job['company']}" for _, job in st.session_state.top_jobs.iterrows()]
        
        for _, job in st.session_state.top_jobs.iterrows():
            matched_skills, missing_skills = skill_gap_analysis(resume_text, job['job_skills'])

            st.write(f"**{job['job_title']} at {job['company']}**")
            st.write(f"📍 **Country:** {job['search_country']} | 🏢 **Type:** {job['job_type']}")
            st.write(f"💯 **Match Score:** {job['match_score']}%")
            st.write(f"✅ **Matched Skills:** {', '.join(matched_skills)}")
            st.write(f"❌ **Missing Skills:** {', '.join(missing_skills)}")
            st.write("---")
    else:
        st.warning("⚠️ No matching jobs found. Please try different filters.")

with col2:
    if not st.session_state.top_jobs.empty:
        selected_job_title = st.selectbox("Select a Job for Cover Letter:", job_options)

        if st.button("Generate Cover Letter"):
            matching_jobs = st.session_state.top_jobs[
                st.session_state.top_jobs.apply(lambda job: f"{job['job_title']} at {job['company']}" == selected_job_title, axis=1)
            ]

            if not matching_jobs.empty:
                selected_job = matching_jobs.iloc[0]  # ✅ Assign the first matching job
                cover_letter_text = generate_cover_letter(selected_job)
                st.text_area("Cover Letter", cover_letter_text, height=400)
            else:
                st.warning("⚠️ No matching job found. Please select another job.")
    else:
        st.warning("⚠️ No jobs available for selection.")

