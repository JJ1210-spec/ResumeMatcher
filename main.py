import streamlit as st
from pdfminer.high_level import extract_text
from similarity import  tfidf_similarity, sbert_similarity
import google.generativeai as genai
from section_parser import extract_sections


# Gemini Setup
genai.configure(api_key="your-google-api-key")
model = genai.GenerativeModel("models/gemini-1.5-flash")

def explain_fit(resume_text, jd_text):
    prompt = f"""You are an HR expert AI.

Here is a resume:
{resume_text}

Here is a job description:
{jd_text}

Analyze whether this resume is a good fit for the job, and explain your reasoning clearly. List strengths, weaknesses, and overall suitability.
"""
    response = model.generate_content(prompt)
    return response.text


def explain_section(section_name, resume_section, job_description):
    prompt = f"""
You are an expert HR assistant.

This is the **{section_name} section** of a resume:

{resume_section}

And here is the job description:

{job_description}

Evaluate how well this resume section aligns with the job description. 
Give a short explanation and a tip for improvement if necessary.
Be short and crisp , no need of giving a big explanation
"""

    response = model.generate_content(prompt)
    return response.text

# App layout
st.set_page_config(page_title="Resume Matcher", layout="wide")
st.title("📄 Resume vs Job Description Matcher")

# Sidebar for Inputs
with st.sidebar:
    uploaded_resume = st.file_uploader("📄 Upload your Resume (PDF only)", type=["pdf"])
    job_description = st.text_area("📋 Paste the Job Description here")
    


# Main Area for Output
if uploaded_resume and job_description:
    resume_text = extract_text(uploaded_resume)

    st.subheader("🧠 Similarity Scores")
    tfidf_score = tfidf_similarity(resume_text, job_description)
    sbert_score = sbert_similarity(resume_text, job_description)

    col1, col2 = st.columns(2)
    col1.metric(label="TF-IDF Score", value=f"{tfidf_score:.2f} %")
    col2.metric(label="SBERT Semantic Score", value=f"{sbert_score:.2f} %")

    st.markdown("---")
    st.write("**Interpretation:**")
    st.markdown("""
    - 🧮 **TF-IDF**: Measures exact keyword overlap.
    - 🧠 **SBERT**: Measures overall meaning/context similarity.
    """)

    # Gemini: AI Resume Fit Explanation
    with st.expander("💬 AI Explanation of Fit (Gemini)", expanded=True):
        with st.spinner("Analyzing fit with Gemini..."):
            explanation = explain_fit(resume_text, job_description)
        st.markdown(explanation)

    # Gemini: Suggested Additions
    prompt = f"""Suggest two bullet points this resume can add to better match the following job description.

Resume:
{resume_text}

Job Description:
{job_description}
"""
    
    st.subheader("📊 Section-Based Resume Analysis")

    sections = extract_sections(resume_text)
    missing_sections = []

# Only show the sections that exist
    for section, content in sections.items():
        if content.strip():
            continue
        else:
            missing_sections.append(section)


    for section, content in sections.items():
        if section not in missing_sections:
            if content.strip():
                score = sbert_similarity(content, job_description)
                st.markdown(f"**{section.capitalize()}** Match Score: {score:.2f}%")

            with st.expander(f"💬 Why is this score? ({section})"):
                explanation = explain_section(section, content, job_description)
                st.markdown(explanation)

    # Suggestions only for missing sections
    
   
    
    st.markdown("### ✨ Suggested Resume Improvements")
    response = model.generate_content(prompt)
    st.markdown(response.text)
    for section in missing_sections:
        st.markdown(f"- Consider adding a **{section.capitalize()}** section to better support your fit for the role.")
