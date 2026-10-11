
import os
import re
import io
import json
from datetime import datetime, timezone

import requests
import streamlit as st
import fitz  # PyMuPDF
from docx import Document
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

st.set_page_config(page_title="CareerLens AI", page_icon="🧭", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #f7f9ff 0%, #ffffff 46%, #f8fbff 100%); }
    [data-testid="stHeader"] { background: rgba(255,255,255,.72); }
    .block-container { padding-top: 1.7rem; padding-bottom: 3rem; max-width: 1180px; }
    .hero-wrap { position:relative; overflow:hidden; padding: clamp(1.6rem, 4vw, 3.5rem); border-radius: 28px;
      background: radial-gradient(circle at 86% 15%, rgba(129,140,248,.42), transparent 29%), radial-gradient(circle at 12% 100%, rgba(45,212,191,.22), transparent 34%), linear-gradient(125deg,#0b122e 0%,#172554 47%,#4338ca 100%);
      color:#fff; box-shadow: 0 24px 60px rgba(49,46,129,.18); margin-bottom:1.15rem; }
    .hero-eyebrow { display:inline-flex; align-items:center; gap:.45rem; padding:.4rem .75rem; border:1px solid rgba(255,255,255,.24); border-radius:999px; background:rgba(255,255,255,.09); color:#dbeafe; font-size:.82rem; letter-spacing:.04em; }
    .hero-title { font-size:clamp(2.25rem, 5.3vw, 4.25rem); font-weight:850; line-height:1.04; letter-spacing:-.045em; margin:.95rem 0 .75rem; }
    .hero-title span { background:linear-gradient(90deg,#a5f3fc,#c4b5fd,#fbcfe8); -webkit-background-clip:text; background-clip:text; color:transparent; }
    .hero-copy { max-width:690px; color:#e0e7ff; font-size:1.08rem; line-height:1.75; margin-bottom:1.35rem; }
    .hero-pills { display:flex; flex-wrap:wrap; gap:.55rem; margin-top:1.2rem; }
    .hero-pill { border:1px solid rgba(255,255,255,.18); background:rgba(255,255,255,.08); border-radius:12px; padding:.55rem .75rem; color:#f8fafc; font-size:.86rem; }
    .section-kicker { color:#4f46e5; text-transform:uppercase; font-size:.76rem; font-weight:800; letter-spacing:.12em; margin-bottom:.25rem; }
    .section-heading { color:#172554; font-size:1.65rem; font-weight:800; letter-spacing:-.025em; margin:.1rem 0 .35rem; }
    .section-subtitle { color:#64748b; margin-bottom:1rem; }
    .feature-card { height:100%; min-height:145px; padding:1.05rem 1.1rem; border:1px solid #e2e8f0; border-radius:18px; background:rgba(255,255,255,.88); box-shadow:0 8px 24px rgba(15,23,42,.035); }
    .feature-icon { font-size:1.45rem; margin-bottom:.45rem; }
    .feature-title { color:#172554; font-weight:750; font-size:1rem; margin-bottom:.28rem; }
    .feature-copy { color:#64748b; font-size:.88rem; line-height:1.55; }
    .job-card { padding: 1rem; border: 1px solid #dbe3ef; border-radius: 14px; background:white; margin-bottom: .7rem; }
    .muted { color:#62748a; }
    div[data-testid="stFileUploader"] { border:1px dashed #a5b4fc; border-radius:18px; padding:.6rem; background:#ffffffb8; }
    div.stButton > button[kind="primary"], div[data-testid="stDownloadButton"] button { border-radius:12px; font-weight:700; }
    @media (max-width: 640px) { .hero-wrap {border-radius:20px;} .hero-copy {font-size:1rem;} }
    /* Hide Streamlit's app toolbar/footer badges where the current version permits. */
    [data-testid="stToolbar"], [data-testid="stStatusWidget"], #MainMenu, footer { visibility: hidden !important; height: 0 !important; }
</style>
""", unsafe_allow_html=True)
st.markdown("""<style>
.brand-mark{font-size:1.15rem;font-weight:850;color:#172554;padding:.45rem 0;white-space:nowrap}.brand-mark span{color:#6366f1}.nav-divider{height:1px;background:#e2e8f0;margin:.25rem 0 1.4rem}.home-panel,.about-panel{border:1px solid #e2e8f0;border-radius:20px;padding:1.3rem 1.4rem;background:linear-gradient(145deg,#fff,#f5f7ff);box-shadow:0 10px 30px rgba(15,23,42,.04);height:100%}.home-panel p,.about-panel p{color:#475569;padding:.45rem 0;border-bottom:1px solid #edf0f7}.home-panel p:last-child,.about-panel p:last-child{border-bottom:0}.footer-note{text-align:center;color:#94a3b8;font-size:.8rem;margin-top:2.2rem;padding:1rem 0}.stButton button{transition:transform .15s ease,box-shadow .15s ease}.stButton button:hover{transform:translateY(-1px);box-shadow:0 5px 15px #4f46e51c}
</style>""", unsafe_allow_html=True)

def extract_text(uploaded_file):
    """Extract text from PDF or DOCX without saving the uploaded file to disk."""
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()
    if name.endswith(".pdf"):
        doc = fitz.open(stream=data, filetype="pdf")
        return "\n".join(page.get_text("text") for page in doc).strip()
    if name.endswith(".docx"):
        import io
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs).strip()
    raise ValueError("Please upload a PDF or DOCX file.")

def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()

ROLE_SKILLS = {
    "Python Developer": ["python", "django", "flask", "fastapi", "sql", "git", "api", "testing"],
    "Data Analyst": ["sql", "python", "excel", "power bi", "tableau", "statistics", "pandas", "visualization"],
    "Web Developer": ["html", "css", "javascript", "react", "python", "api", "git", "responsive"],
    "Machine Learning Engineer": ["python", "machine learning", "scikit-learn", "pandas", "numpy", "statistics", "deep learning", "model evaluation"],
    "Software Tester": ["testing", "test cases", "selenium", "python", "bug tracking", "api testing", "sql", "automation"],
    "Cloud Support Associate": ["linux", "networking", "cloud", "aws", "azure", "python", "security", "troubleshooting"],
    "Android Developer": ["kotlin", "java", "android", "api", "sqlite", "git", "ui", "testing"],
    "Cybersecurity Analyst": ["networking", "linux", "security", "siem", "python", "incident response", "risk", "logs"],
}

def find_skills(text):
    low = text.lower()
    all_skills = sorted({s for skills in ROLE_SKILLS.values() for s in skills}, key=len, reverse=True)
    found = []
    for skill in all_skills:
        pattern = r"(?<![a-z0-9+#])" + re.escape(skill.lower()) + r"(?![a-z0-9+#])"
        if re.search(pattern, low):
            found.append(skill)
    return sorted(set(found))

def is_likely_resume(text):
    """Heuristic check to reject ordinary documents before resume analysis."""
    normalized = re.sub(r"\s+", " ", text or "").strip().lower()
    if len(normalized.split()) < 35:
        return False, "This document is too short to identify as a resume. Please upload a complete resume."

    # Look for several resume-specific signals rather than relying on a single keyword.
    section_terms = [
        "education", "work experience", "professional experience", "employment history",
        "technical skills", "skills", "projects", "internship", "career objective",
        "professional summary", "certifications", "qualification", "experience",
    ]
    section_hits = sum(1 for term in section_terms if re.search(r"\b" + re.escape(term) + r"\b", normalized))
    skill_hits = len(find_skills(normalized))
    contact_hits = int(bool(re.search(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", normalized)))
    contact_hits += int(bool(re.search(r"(?:\+?\d[\d ()-]{7,}\d)", normalized)))

    # Accept a mix of resume sections, recognizable skills, and/or contact details.
    signals = (section_hits >= 2) or (section_hits >= 1 and skill_hits >= 2) or (skill_hits >= 3 and contact_hits >= 1)
    if not signals:
        return False, "This file doesn't appear to be a resume. Please upload a resume containing details such as education, skills, projects, or work experience."
    return True, "Resume document detected."

def career_recommendations(resume_text):
    skills = set(find_skills(resume_text))
    scored = []
    for role, required in ROLE_SKILLS.items():
        matched = sorted(skills.intersection(required))
        missing = sorted(set(required) - skills)
        score = round(100 * len(matched) / max(1, len(required)))
        scored.append({
            "role": role, "score": score, "matched": matched, "missing": missing,
            "required_count": len(required)
        })
    return sorted(scored, key=lambda x: x["score"], reverse=True), sorted(skills)

def match_score(resume_text, job_text):
    """Combine semantic-ish TF-IDF similarity with visible skill overlap."""
    resume_text, job_text = clean_text(resume_text), clean_text(job_text)
    if not resume_text or not job_text:
        return 0, [], []
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=8000)
        matrix = vec.fit_transform([resume_text, job_text])
        similarity = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
    except ValueError:
        similarity = 0.0
    resume_skills, job_skills = set(find_skills(resume_text)), set(find_skills(job_text))
    matched = sorted(resume_skills.intersection(job_skills))
    missing = sorted(job_skills - resume_skills)
    skill_score = len(matched) / max(1, len(job_skills))
    combined = round(100 * (0.55 * similarity + 0.45 * skill_score))
    return min(100, combined), matched, missing

COUNTRY_CODES = {
    "India": "in", "United Kingdom": "gb", "United States": "us",
    "Australia": "au", "Canada": "ca"
}

def search_live_jobs(query, location_text, country_name, limit=10):
    app_id = os.getenv("ADZUNA_APP_ID")
    app_key = os.getenv("ADZUNA_APP_KEY")
    if not app_id or not app_key:
        return None, "Live job search is not configured yet. Add your Adzuna API credentials to the .env file."
    country_code = COUNTRY_CODES[country_name]
    url = f"https://api.adzuna.com/v1/api/jobs/{country_code}/search/1"
    params = {
        "app_id": app_id, "app_key": app_key, "results_per_page": limit,
        "what": query, "content-type": "application/json"
    }
    if location_text.strip():
        params["where"] = location_text.strip()
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
        jobs = []
        for item in payload.get("results", []):
            jobs.append({
                "title": item.get("title") or "Untitled role",
                "company": (item.get("company") or {}).get("display_name", "Company not listed"),
                "location": (item.get("location") or {}).get("display_name", "Location not listed"),
                "description": item.get("description", ""),
                "url": item.get("redirect_url", ""),
                "created": item.get("created", ""),
                "salary_min": item.get("salary_min"),
                "salary_max": item.get("salary_max"),
            })
        return jobs, None
    except requests.RequestException as exc:
        return None, f"Could not retrieve live jobs: {exc}"
    except (ValueError, TypeError):
        return None, "The job service returned an unexpected response."

def show_job(job, resume_text, idx):
    description = job.get("description", "")
    score, matched, missing = match_score(resume_text, job.get("title", "") + " " + description)
    with st.container(border=True):
        st.subheader(job.get("title", "Job"))
        st.write(f"**Company:** {job.get('company', 'Not listed')}  ·  **Location:** {job.get('location', 'Not listed')}")
        if job.get("created"):
            st.caption(f"Listing date supplied by source: {job['created']}")
        c1, c2 = st.columns([1, 2])
        c1.metric("Estimated resume match", f"{score}%")
        with c2:
            st.write("**Skills found in both**")
            st.write(", ".join(matched) if matched else "No known skill overlap detected; review the job description.")
            st.write("**Potential skill gaps / not found in resume**")
            st.write(", ".join(missing) if missing else "No gaps detected among the skills this prototype recognizes.")
        if job.get("url"):
            st.link_button("Open original job listing", job["url"])
        with st.expander("View job description"):
            st.write(description or "No description was provided by the listing source.")
        st.caption("Match score is an estimate based on text similarity and recognized skills, not a hiring probability. Listings can expire; confirm status on the original page.")

def resume_quality_checks(text):
    """Simple transparent checks; not a formal ATS certification."""
    low = text.lower()
    checks = [
        ("Contact details", bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)) or bool(re.search(r"(?:\+?\d[\d\s().-]{8,}\d)", text)), "Include a professional email and a reachable phone number."),
        ("Skills section clues", any(k in low for k in ["skills", "technical skills", "technologies"]), "Add a clearly labelled Skills section with relevant skills you can demonstrate."),
        ("Project or experience evidence", any(k in low for k in ["project", "experience", "internship", "work history"]), "Add projects, internship, or experience entries with your actual contribution."),
        ("Education section", any(k in low for k in ["education", "degree", "diploma", "university", "college"]), "Make your education and expected graduation details easy to find."),
        ("Action/result language", any(k in low for k in ["built", "developed", "created", "implemented", "improved", "automated", "designed"]), "Describe what you did with strong verbs; add truthful measurable results where possible."),
        ("Reasonable length", 250 <= len(text.split()) <= 1400, "For many entry-level resumes, aim for concise, relevant content; adjust to your experience and local norms."),
    ]
    return checks

def build_learning_plan(missing_skills):
    if not missing_skills:
        return ["Review target job descriptions and confirm your resume gives evidence for the skills you already list.", "Build or polish one project that demonstrates your strongest skills.", "Practice explaining project decisions and trade-offs."]
    return [f"Week {i+1}: Learn the basics of **{skill}** and complete one small hands-on exercise." for i, skill in enumerate(missing_skills[:4])] + ["Final step: update your resume only after you can honestly demonstrate the skill."]

def make_report(resume_name, skills, recommendations, checks):
    lines = ["CAREERLENS AI — RESUME REVIEW", f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", f"Resume file: {resume_name}", "", "DETECTED SKILLS", ", ".join(skills) if skills else "No known skills detected", "", "ROLE SUGGESTIONS (skill coverage estimate)"]
    for rec in recommendations[:6]:
        lines += [f"- {rec['role']}: {rec['score']}%", f"  Skills found: {', '.join(rec['matched']) or 'None detected'}", f"  Skills to investigate: {', '.join(rec['missing']) or 'None in this taxonomy'}"]
    lines += ["", "RESUME QUALITY CHECKS"]
    for label, ok, advice in checks:
        lines.append(f"[{'PASS' if ok else 'REVIEW'}] {label}: {advice}")
    lines += ["", "IMPORTANT NOTE", "Scores and checks are heuristic guidance, not an ATS guarantee or a prediction of hiring success. Verify all facts and tailor your resume honestly to each role."]
    return "\n".join(lines)


# ---------- Navigation ----------
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "resume_name" not in st.session_state:
    st.session_state.resume_name = ""
if "live_jobs" not in st.session_state:
    st.session_state.live_jobs = None

nav_items = ["Home", "Resume Analyzer", "Find Jobs", "About"]
nav_cols = st.columns([1.05, 1.55, 1.15, 0.8, 1.3])
with nav_cols[0]:
    st.markdown('<div class="brand-mark">🧭 CareerLens <span>AI</span></div>', unsafe_allow_html=True)
for i, item in enumerate(nav_items):
    with nav_cols[i+1]:
        if st.button(item, key=f"nav_{item}", type="primary" if st.session_state.page == item else "secondary", use_container_width=True):
            st.session_state.page = item
            st.rerun()
st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Job search settings")
    country = st.selectbox("Country", ["India", "United Kingdom", "United States", "Australia", "Canada"], index=0)
    location = st.text_input("City / location (optional)", placeholder="e.g., Hyderabad")
    results_limit = st.slider("Maximum jobs to show", 5, 20, 10)
    st.caption("Live listings require valid Adzuna API credentials in your .env file.")
    st.divider()
    st.caption("Your resume is analyzed in this app session. Avoid uploading documents containing information you do not want to process.")

def render_hero():
    st.markdown("""<div class="hero-wrap">
      <div class="hero-eyebrow">✦ YOUR NEXT CHAPTER STARTS HERE</div>
      <div class="hero-title">AI Resume Analyzer<br><span>Personalize your career with AI.</span></div>
      <div class="hero-copy">Turn your resume into a career game plan. Discover roles that fit your skills, explore job opportunities, spot skill gaps, and take your next step with confidence.</div>
      <div class="hero-pills"><div class="hero-pill">✧ Resume insights</div><div class="hero-pill">◎ Career recommendations</div><div class="hero-pill">⌕ Job matching</div><div class="hero-pill">↗ Skill roadmap</div></div>
    </div>""", unsafe_allow_html=True)

def render_feature_cards():
    st.markdown('<div class="section-kicker">ONE SMARTER CAREER MOVE AT A TIME</div><div class="section-heading">Your career toolkit, all in one place</div><div class="section-subtitle">Simple insights to help you understand your strengths and plan what comes next.</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3, gap="medium")
    cards = [("🧠", "Understand your resume", "Identify recognized skills and review useful resume-quality checks."), ("🎯", "Find your direction", "Explore career roles, compare job requirements, and see potential skill gaps."), ("🚀", "Build your next step", "Get a starter learning roadmap, interview prompts, and downloadable reports.")]
    for col, (icon, title, copy) in zip((f1,f2,f3), cards):
        with col:
            st.markdown(f'<div class="feature-card"><div class="feature-icon">{icon}</div><div class="feature-title">{title}</div><div class="feature-copy">{copy}</div></div>', unsafe_allow_html=True)

def resume_upload():
    st.markdown('<div class="section-kicker">START HERE</div><div class="section-heading">Upload your resume</div><div class="section-subtitle">Choose a PDF or DOCX file to unlock career insights, job matching, and your skill roadmap.</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose your resume file", type=["pdf", "docx"], help="Supported formats: PDF and DOCX.")
    if uploaded:
        try:
            text = extract_text(uploaded)
        except Exception as exc:
            st.error(f"Could not read this resume: {exc}")
            return
        if not text:
            st.error("No selectable text was found. If this is a scanned PDF, OCR support is needed.")
            return
        is_resume, validation_message = is_likely_resume(text)
        if not is_resume:
            # Clear any previously uploaded resume so invalid documents cannot reuse old results.
            st.session_state.resume_text = ""
            st.session_state.resume_name = ""
            st.error(validation_message)
            return
        st.session_state.resume_text = text
        st.session_state.resume_name = uploaded.name
        st.success("Resume text extracted successfully. Go to Resume Analyzer or Find Jobs to continue.")
    if st.session_state.resume_text:
        st.info(f"Resume ready: **{st.session_state.resume_name}** · {len(st.session_state.resume_text.split())} words")
        if st.button("Analyze my resume", type="primary", key="go_analyze"):
            st.session_state.page = "Resume Analyzer"
            st.rerun()

if st.session_state.page == "Home":
    render_hero()
    render_feature_cards()
    st.markdown('<div style="height:.7rem"></div>', unsafe_allow_html=True)
    left, right = st.columns([1.1, .9], gap="large")
    with left:
        resume_upload()
    with right:
        st.markdown('<div class="home-panel"><div class="section-kicker">HOW IT WORKS</div><div class="section-heading">A clearer path forward</div><p>01 · Upload your resume</p><p>02 · Discover roles that fit your current skills</p><p>03 · Compare your resume with a job description</p><p>04 · Explore real listings and plan what to learn next</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="footer-note">CareerLens AI provides guidance to support your decisions. It does not guarantee interviews or employment.</div>', unsafe_allow_html=True)

elif st.session_state.page == "Resume Analyzer":
    render_hero()
    resume_upload()
    resume_text = st.session_state.resume_text
    if resume_text:
        st.divider()
        skills = find_skills(resume_text)
        recommendations, _ = career_recommendations(resume_text)
        checks = resume_quality_checks(resume_text)
        passed = sum(1 for _, ok, _ in checks if ok)
        st.markdown('<div class="section-kicker">YOUR RESUME REPORT</div><div class="section-heading">Resume insights</div>', unsafe_allow_html=True)
        q1, q2, q3 = st.columns(3)
        q1.metric("Words in resume", len(resume_text.split()))
        q2.metric("Recognized skills", len(skills))
        q3.metric("Checklist", f"{passed}/{len(checks)}")
        with st.expander("Preview extracted text"):
            st.text_area("Extracted resume text", resume_text, height=180)
        st.subheader("Detected skills")
        st.write(" · ".join(f"`{s}`" for s in skills) if skills else "No skills from the built-in skill list were detected.")
        st.subheader("Resume quality checklist")
        for label, ok, advice in checks:
            st.markdown(f"{'✅' if ok else '🔎'} **{label}** — {'Detected' if ok else 'Review suggested'}")
            if not ok: st.caption(advice)
        st.subheader("Career recommendations")
        st.caption("Role coverage is a simple skills-based estimate, not a hiring prediction.")
        cols = st.columns(2)
        for i, rec in enumerate(recommendations[:6]):
            with cols[i % 2]:
                with st.container(border=True):
                    st.subheader(rec["role"])
                    st.metric("Skill coverage estimate", f"{rec['score']}%")
                    st.write("**Skills found:** " + (", ".join(rec["matched"]) if rec["matched"] else "None detected"))
                    st.write("**Skills to investigate:** " + (", ".join(rec["missing"][:5]) if rec["missing"] else "No listed gaps"))
                    if st.button(f"Find jobs: {rec['role']}", key=f"role_{i}"):
                        st.session_state["job_query"] = rec["role"]
                        st.session_state.page = "Find Jobs"
                        st.rerun()
        best_role = recommendations[0] if recommendations else None
        if best_role:
            with st.expander(f"Starter learning roadmap for {best_role['role']}", expanded=True):
                for item in build_learning_plan(best_role["missing"]): st.markdown(f"- {item}")
                st.markdown("**Portfolio idea:** Build a small project related to this role, document it clearly, and be ready to explain your own contribution.")
        report = make_report(st.session_state.resume_name, skills, recommendations, checks)
        st.download_button("Download resume analysis report (.txt)", data=report, file_name="careerlens_resume_report.txt", mime="text/plain")
    else:
        st.info("Upload your resume above to see detected skills, quality checks, career recommendations, and a learning roadmap.")

elif st.session_state.page == "Find Jobs":
    st.markdown('<div class="section-kicker">YOUR NEXT OPPORTUNITY</div><div class="section-heading">Find jobs & compare requirements</div><div class="section-subtitle">Search vacancies from the configured provider or compare your resume with a job description.</div>', unsafe_allow_html=True)
    resume_text = st.session_state.resume_text
    if not resume_text:
        st.warning("For personalized resume matching, upload your resume on the Home page first. You can still search jobs without it.")
    query = st.text_input("Job title or keywords", value=st.session_state.get("job_query", "Python Developer"), key="job_search_input")
    if st.button("Search live jobs", type="primary"):
        with st.spinner("Searching the job listings provider..."):
            jobs, error = search_live_jobs(query, location, country, results_limit)
        if error:
            st.warning(error)
            st.info("Add ADZUNA_APP_ID and ADZUNA_APP_KEY to your local .env file. Do not share or commit those credentials.")
        elif not jobs:
            st.session_state.live_jobs = []
            st.info("No results were returned. Try another title or location.")
        else:
            st.session_state.live_jobs = jobs
            st.session_state.live_jobs_query = query
            st.session_state.live_jobs_checked = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if st.session_state.live_jobs is not None:
        st.caption(f"Results for “{st.session_state.get('live_jobs_query', '')}” · fetched {st.session_state.get('live_jobs_checked', '')}")
        for idx, job in enumerate(st.session_state.live_jobs):
            if resume_text:
                show_job(job, resume_text, idx)
            else:
                with st.container(border=True):
                    st.subheader(job.get("title", "Job"))
                    st.write(f"**Company:** {job.get('company', 'Not listed')} · **Location:** {job.get('location', 'Not listed')}")
                    st.write(job.get("description", "")[:700])
                    if job.get("url"): st.link_button("Open original job listing", job["url"])
    st.divider()
    st.subheader("Match your resume with a job description")
    job_title = st.text_input("Target job title", placeholder="e.g., Junior Data Analyst", key="match_job_title")
    job_description = st.text_area("Paste the job description", height=180, placeholder="Responsibilities, required skills, qualifications...", key="match_job_description")
    if st.button("Analyze job match"):
        if not resume_text:
            st.error("Please upload a resume on the Home page first.")
        elif not job_description.strip():
            st.error("Please paste a job description first.")
        else:
            score, matched, missing = match_score(resume_text, job_title + " " + job_description)
            st.metric("Estimated resume-to-job match", f"{score}%")
            a, b = st.columns(2)
            with a:
                st.markdown("#### Matching skills")
                st.write(", ".join(matched) if matched else "No recognized matching skills found.")
            with b:
                st.markdown("#### Potential gaps")
                st.write(", ".join(missing) if missing else "No gaps found among recognized skills.")
            st.markdown("#### Suggested next steps")
            st.write("Review the job requirements, strengthen relevant skills with practical projects, and update your resume only with truthful evidence.")
            match_report = {"generated_at": datetime.now().isoformat(timespec="minutes"), "target_job_title": job_title or "Not specified", "estimated_match_score_percent": score, "matched_skills": matched, "potential_gaps_not_found_in_resume": missing, "note": "Heuristic estimate, not an ATS score or hiring probability."}
            st.download_button("Download job-match report (.json)", data=json.dumps(match_report, indent=2), file_name="careerlens_job_match.json", mime="application/json")
            st.markdown("#### Interview practice")
            for i, question in enumerate([f"Explain a project relevant to {job_title or 'this role'} and your contribution.", "Describe a technical problem you solved and how you tested your solution.", "Which required skill are you strongest in? Give a real example.", "Which skill will you improve next, and what is your plan?"], 1): st.markdown(f"{i}. {question}")

elif st.session_state.page == "About":
    st.markdown('<div class="section-kicker">ABOUT THE PROJECT</div><div class="section-heading">CareerLens AI</div><div class="section-subtitle">A Python-based career support prototype designed to help users review resumes and explore next steps.</div>', unsafe_allow_html=True)
    st.markdown("""<div class="about-panel"><h3>What CareerLens AI can do</h3><p>📄 Extract text from PDF and DOCX resumes</p><p>🧠 Identify skills from a small built-in skill taxonomy</p><p>🎯 Suggest role families based on recognized skills</p><p>🧰 Run transparent resume quality checks</p><p>🔎 Search job listings through Adzuna when credentials and service coverage are available</p><p>📊 Compare resume text with a job description and show potential skill gaps</p><p>🗺️ Create a starter learning roadmap and interview practice questions</p></div>""", unsafe_allow_html=True)
    st.markdown("#### Technology stack")
    st.write("Python · Streamlit · PyMuPDF · python-docx · scikit-learn TF-IDF/cosine similarity · Adzuna job-search API")
    st.markdown("#### Important limitations")
    st.write("This prototype uses transparent rules and text similarity, not a trained large language model. Scores are estimates and can miss skills that are not explicitly written in a resume. Job listings may expire or may not be available for every region. Always verify details on the original listing and keep all resume claims truthful.")
    if st.button("Start with my resume", type="primary"):
        st.session_state.page = "Home"
        st.rerun()

st.markdown('<div class="footer-note">© CareerLens AI · Built with Python and Streamlit · Career guidance only, not a hiring guarantee.</div>', unsafe_allow_html=True)
