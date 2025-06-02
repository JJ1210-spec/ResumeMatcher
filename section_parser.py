import re

def extract_sections(text):
    sections = {
        "skills": "",
        "experience": "",
        "education": "",
        "projects": ""
    }

    text = text.lower()
    lines = text.splitlines()
    current = None

    for line in lines:
        clean = line.strip().lower()
        if "skill" in clean:
            current = "skills"
        elif "experience" in clean or "work history" in clean:
            current = "experience"
        elif "education" in clean:
            current = "education"
        elif "project" in clean:
            current = "projects"
        elif current:
            sections[current] += line + " "

    return sections
