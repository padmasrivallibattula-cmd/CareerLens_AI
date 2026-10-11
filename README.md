# CareerLens AI — Python-only AI Resume Analyzer

A hackathon starter built entirely in Python. It uses Streamlit for the interface, PyMuPDF/python-docx for resume extraction, scikit-learn for a transparent baseline matching score, and a job listings API for live vacancies.

## Enhanced features

- Upload PDF or DOCX resumes
- Extract resume text without saving the uploaded file to disk
- Detect skills using a small editable role/skills taxonomy
- Suggest possible career roles
- Search live job listings through the Adzuna API
- Estimate resume-to-job match scores
- Show skills detected in common and potential gaps
- Analyze a specific job description pasted by the user

## Requirements

- Python 3.10 or newer recommended
- Internet connection for live job searches
- Job provider API credentials for live listings

## Setup (Windows)

Open a terminal in this project folder:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and replace the example values with your API ID and key from the provider. Do not commit `.env` to GitHub.

Run the app:

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Live job search

The app uses the Adzuna job search endpoint. Register with the provider and confirm that your account has access to the country you want to search. API availability, rate limits, terms, and geographic coverage may change. See:

- https://developer.adzuna.com/overview
- https://developer.adzuna.com/docs/search

If credentials are absent, the app does not invent fake live listings. Resume parsing and pasted-job matching can still be used.

## Important limitations

This is a prototype, not a validated hiring system.

- The built-in role taxonomy is small. Edit `ROLE_SKILLS` in `app.py` to add roles and skills.
- The matching score combines TF-IDF text similarity and recognized skill overlap. It is not a probability of getting hired.
- The prototype does not independently guarantee that a vacancy remains open. Open the original listing and verify its status before applying.
- Some PDFs are scanned images; OCR is not included in this starter.
- LLM-powered generation is not wired in yet. Suggestions are rule-based so the project works without an LLM key.
- Resume data is sensitive. Keep API keys private, obtain user consent before storing data, and avoid uploading real personal resumes to public repositories.

## Suggested next upgrades

1. Add OCR for scanned resumes.
2. Add more roles and skill synonyms.
3. Add job listing deduplication and caching.
4. Add an optional LLM API for grounded, evidence-based resume suggestions.
5. Add a user-controlled export of the analysis report.
6. Add tests for parsing, scoring, and API error handling.


## Improvements in this edition
- Resume quality checklist for contact details, skills, projects/experience, education, action verbs, and approximate length.
- Downloadable plain-text resume review report.
- Starter learning roadmap based on the top suggested role's unrecognized skills.
- Downloadable JSON report for a specific job match.
- Interview practice questions after job-description analysis.
- Clearer disclaimers: scores are heuristics, not hiring predictions or official ATS scores.

## Important API security note
Keep your real Adzuna credentials in your local `.env` file and never commit that file. `.env` is ignored by Git. For Streamlit Community Cloud, put credentials in the app's Secrets settings instead of uploading `.env`.
