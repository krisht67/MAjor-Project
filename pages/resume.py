import streamlit as st
from PyPDF2 import PdfReader
from io import BytesIO

def read_pdf(uploaded_file):
    content = uploaded_file.getvalue()
    with BytesIO(content) as f:
        reader = PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_resume_sections(resume_text):
    # Split the resume text into sections based on common headings or keywords
    sections = {
        "Education": "",
        "Experience": "",
        "Technical Skills": "",
        "Projects": "",
        "Leadership":""
        # Add more sections as needed
    }
    
    current_section = None
    for line in resume_text.split("\n"):
        line_lower = line.lower()
        for section_name in sections:
            if section_name.lower() in line_lower:
                current_section = section_name
                break
        if current_section is not None:
            sections[current_section] += line + "\n"
    
    return sections

def main():
    st.title("**Resume Reviewer**")

    uploaded_file = st.file_uploader("Upload a resume (PDF)", type=["pdf"])

    if uploaded_file is not None:
        resume_text = read_pdf(uploaded_file)
        sections = extract_resume_sections(resume_text)
        
        for section_name, section_content in sections.items():
            st.subheader(f"**{section_name}**")
            st.text_area("", section_content, height=200)

if __name__ == "__main__":
    main()
