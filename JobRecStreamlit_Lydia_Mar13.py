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


# In[ ]:


# 🔹 Set page layout to wide
st.set_page_config(page_title="Job Recommendation System", layout="wide")

# 🔹 Load NLTK Stopwords & Lemmatizer
nltk_stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# 🔹 Function to Clean Text
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in nltk_stopwords]
    return " ".join(words)

# 🔹 Function to Extract Text from Resumes
def extract_resume_text(uploaded_file):
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()  # Read file in memory
        if uploaded_file.name.endswith('.pdf'):
            return extract_text(io.BytesIO(file_bytes))  # Read PDF from memory
        elif uploaded_file.name.endswith('.docx'):
            doc = Document(io.BytesIO(file_bytes))  # Read DOCX from memory
            return "\n".join([para.text for para in doc.paragraphs])
        else:
            return "Unsupported file format."
    return ""


# In[ ]:


# 🔹 Load Job Data (Use `st.cache_data` for Speed)
@st.cache_data
def load_data():
    job_data = pd.read_csv('/Users/tzuchihsu/UCI/2025 Winter/NLP/LinkedIn Job Rec/data cleaning/linkedin_job_posts_skills.csv')
    job_data['job_text'] = job_data['job_title'] + " " + job_data['job_skills']
    job_data['clean_text'] = job_data['job_text'].apply(clean_text)
    return job_data

# Display loading message
with st.status("🔄 Loading job dataset...", expanded=True):
    job_data = load_data()
    st.write("✅ Job dataset loaded!")

# 🔹 TF-IDF Vectorization (Convert Job Descriptions into Vectors)
st.write("🔄 **Processing job descriptions...**")
vectorizer = TfidfVectorizer(stop_words='english')
job_vectors = vectorizer.fit_transform(job_data['clean_text'])
st.write("✅ **Job descriptions processed!**")


# In[ ]:


# 🔹 Function to Match Resume to Jobs with Normalized Match Score
def get_top_jobs_with_match_score(resume_text, top_n=5):
    resume_vector = vectorizer.transform([clean_text(resume_text)])
    similarity_scores = cosine_similarity(resume_vector, job_vectors).flatten()
    
    # Normalize scores: Scale between 0% and 100% relative to the highest match
    max_score = max(similarity_scores) if max(similarity_scores) > 0 else 1  # Avoid division by zero
    match_scores = (similarity_scores / max_score * 100).round(2)  # Scale scores to a percentage
    
    # Select the top N jobs
    top_indices = np.argsort(similarity_scores)[::-1][:top_n]
    top_jobs = job_data.iloc[top_indices].copy()
    top_jobs["match_score"] = match_scores[top_indices]  # Add match score to dataframe
    
    return top_jobs


# In[ ]:


# 🔹 Cover Letter Generator with Improved Template
def generate_cover_letter(job, user_name="Candidate", user_experience="your relevant experience"):
    return f"""
    Dear Hiring Manager,

    I am excited to apply for the {job['job_title']} position at {job['company']}. 
    With my background in {user_experience}, I am eager to bring my skills in {job['job_skills']} to your team.

    In my previous role as a [Your Most Relevant Job Title] at [Previous Company], 
    I [describe an achievement that demonstrates a key skill relevant to the job, using quantifiable impact if possible]. 
    This experience has equipped me with the ability to [explain how your skills will add value to the new role].

    What excites me most about this opportunity at {job['company']} is 
    [mention something specific about the company, its culture, mission, or projects that align with your goals]. 
    I am eager to contribute by [mention how you can help solve a challenge or add value based on your experience].

    I would love the opportunity to discuss how my skills and experience align with your needs. 
    Please feel free to contact me at your convenience to schedule a conversation. 
    Thank you for your time and consideration—I look forward to hearing from you.

    Best regards,  
    {user_name}
    """


# In[ ]:


# 🔹 Streamlit UI Layout
st.title("💼 AI-Powered Job Recommendation System")
st.write("Upload your resume to get started!")

# Initialize session state to store job recommendations
if "top_jobs" not in st.session_state:
    st.session_state.top_jobs = None

# 🔹 Resume Upload Section
uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
selected_job = None  # Initialize selected job

if uploaded_file is not None:
    st.write(f"📂 **File Uploaded:** `{uploaded_file.name}`")

    with st.status("🔄 Extracting text from resume...", expanded=True):
        resume_text = extract_resume_text(uploaded_file)
        st.write("✅ Resume text extracted!")

    # Display Extracted Resume Content
    st.write("### 📄 Extracted Resume Text:")
    st.text_area("Resume Content", resume_text, height=200)

    # 🔹 Get Job Recommendations
    if st.button("Find Matching Jobs"):
        with st.status("🔄 Searching for best job matches...", expanded=True):
            st.session_state.top_jobs = get_top_jobs_with_match_score(resume_text)
            st.write("✅ Job matches found!")

# 🔹 Display Job Recommendations & Cover Letter Side-by-Side
if st.session_state.top_jobs is not None:
    col1, col2 = st.columns([3, 1])  # Left: 3x space (Jobs) | Right: 1x space (Cover Letter)

    # 🔹 Job Recommendations (Left Side - Wider)
    with col1:
        st.write("### 🎯 Recommended Jobs:")
        job_options = []
        for i, (_, job) in enumerate(st.session_state.top_jobs.iterrows()):
            job_title = f"{job['job_title']} at {job['company']}"
            job_options.append(job_title)

            # Display job details
            st.write(f"**{i+1}. {job_title}**")
            st.write(f"📍 {job['job_location']} | 🏢 {job['job_type']} | 💯 Match Score: **{job['match_score']}%**")
            st.write(f"**Required Skills:** {job['job_skills']}")
            st.write("---")



    # 🔹 Cover Letter Generator (Right Side - Smaller)
    with col2:
        # Let user select a job for cover letter
        selected_job_title = st.selectbox("Select a Job for Cover Letter:", job_options)
        selected_job = st.session_state.top_jobs[st.session_state.top_jobs.apply(
            lambda job: f"{job['job_title']} at {job['company']}" == selected_job_title, axis=1)].iloc[0]
        
        st.write("### 📝 Cover Letter Preview")
        if selected_job is not None and st.button("Generate Cover Letter"):
            with st.status("📝 Generating Cover Letter...", expanded=True):
                cover_letter = generate_cover_letter(selected_job, user_name="Your Name", user_experience="data analytics, product management, marketing")
                st.write("✅ Cover Letter Ready!")
            st.text_area("Cover Letter", cover_letter, height=400)


        

