import hashlib
import json
import re
import time
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher

import pandas as pd
import requests
import streamlit as st
from docx import Document
from pypdf import PdfReader

st.set_page_config(page_title="VerifyAI - Candidate Command Center", layout="wide")


# =====================================================================
# CUSTOM ENTERPRISE CSS (DARK GLASSMORPHISM, STATUS PILLS & BADGES)
# =====================================================================
st.markdown("""
<style>
    /* Global App Background */
    .stApp {
        background-color: #0D1117;
        color: #E6EDF3;
    }
    
    /* Dark Glassmorphism Card Containers */
    div[data-testid="stVerticalBlock"] > div {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 12px;
    }
    
    /* Verified Human Status Pill */
    .status-verified {
        background-color: #0E3A2F;
        color: #3FB950;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 12px;
        display: inline-block;
    }
    
    /* Effort Flagged Status Pill */
    .status-flagged {
        background-color: #3D2D00;
        color: #D29922;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 12px;
        display: inline-block;
    }
    
    /* Top Floating Hybrid AI Pipeline Status Badge */
    .pipeline-status {
        background-color: #161B22;
        border: 1px solid #30363D;
        padding: 10px 16px;
        border-radius: 8px;
        font-size: 13px;
        color: #E6EDF3;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# CONFIG & SECRETS
# =====================================================================
def get_secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


OLLAMA_URL = get_secret("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = get_secret("OLLAMA_MODEL", "llama3.2")
SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_KEY = get_secret("SUPABASE_KEY")
EMPLOYER_PASSWORD = get_secret("EMPLOYER_PASSWORD", "admin")
APP_SALT = get_secret("APP_SALT", "change-this-salt")

REVIEW_THRESHOLD = 70
MAX_RESUME_CHARS = 7000

WEIGHTS = {"answers": 0.55, "evidence": 0.20, "projects": 0.15, "resume": 0.10}
HUMAN_WEIGHT = 0.40

CONSENT_TEXT = (
    "I agree that VerifyAI stores my profile and answers, analyses them with an AI model, "
    "and shows them to hiring teams. I can request a human review or delete my data at any time."
)


# =====================================================================
# AI INTERACTION ENGINE (DYNAMIC GROQ DISCOVERY & FAST OLLAMA FALLBACK)
# =====================================================================
def parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def ask_ai(system_prompt, user_input, as_json=False):
    groq_key = get_secret("GROQ_API_KEY")
    
    # 1. PRIMARY: Groq Cloud API with dynamic runtime model discovery
    if groq_key:
        try:
            from groq import Groq
            groq_client = Groq(api_key=groq_key)
            
            model_objects = groq_client.models.list().data
            available_models = [m.id for m in model_objects]
            
            priority_list = [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "llama3-70b-8192",
                "llama3-8b-8192",
                "mixtral-8x7b-32768"
            ]
            
            models_to_try = [m for m in priority_list if m in available_models]
            if not models_to_try and available_models:
                models_to_try = [m for m in available_models if "whisper" not in m and "vision" not in m]
            
            for m_name in models_to_try:
                try:
                    kwargs = {
                        "model": m_name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_input}
                        ],
                        "temperature": 0.2
                    }
                    if as_json:
                        kwargs["response_format"] = {"type": "json_object"}
                    
                    completion = groq_client.chat.completions.create(**kwargs)
                    text = completion.choices[0].message.content
                    
                    if as_json:
                        data = parse_json(text)
                        if data is not None:
                            return data
                    else:
                        return text
                except Exception:
                    continue
        except Exception as groq_err:
            st.error(f"Groq API Error: {groq_err}")

    st.warning("Falling back to local Ollama processing...")
    
    # 2. FALLBACK: Fast Local Ollama Execution
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 2048},
    }
    if as_json:
        payload["format"] = "json"

    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=30)
        r.raise_for_status()
        text = r.json()["message"]["content"]
    except Exception as ollama_err:
        st.error(f"Both Groq Cloud and Local Ollama failed: {ollama_err}")
        return None

    if as_json:
        data = parse_json(text)
        if data is None:
            st.error("The AI returned an unreadable response. Please try again.")
        return data
    return text


FAIRNESS = (
    "FAIRNESS RULES: ignore name, gender, age, nationality, religion, marital status, photo, location, "
    "college or university prestige, and employment gaps. Judge only demonstrated skills, projects and the "
    "quality of answers. Never penalise a non-traditional background or a career break."
)

ANALYZE_SYSTEM = (
    "You are a career analyst. Read the resume and return ONLY valid JSON with these keys: "
    "summary (string, 2 sentences, do NOT include the person's name), skills (list of strings), "
    "qualifications (list of degree or certificate names only, WITHOUT institution names), "
    "projects (list of objects with title and description), experience_years (number), "
    "primary_domain (string), "
    "suggested_roles (list of exactly 5 objects with role and reason - roles this person can apply for now), "
    "questions (list of exactly 5 strings). "
    "The questions must test what the candidate really knows about their own domain and projects. "
    "Never invent facts that are not in the resume. " + FAIRNESS
)

REVIEW_SYSTEM = (
    "You are an expert resume reviewer. Review the resume for the candidate's primary domain. "
    "Return ONLY valid JSON with keys: target_profile (string), "
    "resume_strength_score (integer 0-100), "
    "project_evidence_score (integer 0-100), "
    "keyword_stuffing_risk ('Low', 'Medium' or 'High'), strong_points (list, max 3), weak_points (list, max 3), "
    "improvement_tips (list, max 4). " + FAIRNESS
)

EVAL_SYSTEM = (
    "You are a strict but fair hiring assessor. You receive a candidate profile and their answers. "
    "Return ONLY valid JSON with keys: "
    "scores (list of integers 0-100, one per answer, in order), "
    "category (best-fit job category), "
    "level ('Beginner', 'Intermediate' or 'Advanced'), "
    "strengths (list of strings), gaps (list of strings), "
    "verified_claims (list of skills genuinely demonstrated), "
    "unverified_claims (list of skills NOT demonstrated), "
    "interview_probes (list of exactly 3 live verification questions), "
    "reasoning (2-3 sentences explaining the judgement). " + FAIRNESS
)

ROADMAP_SYSTEM = (
    "You are a senior career mentor. Write a complete, practical roadmap in Markdown for moving "
    "from the candidate's current profile to the target role. Include: Gap Analysis, Timeline, "
    "Courses, Projects, Interview Questions, Target Companies, and 30-Day Plan."
)

MATCH_SYSTEM = (
    "You are a recruiter's assistant. Compare the candidate profile with the job requirements. "
    "Return ONLY valid JSON with keys: match_score (integer 0-100), "
    "resume_fit ('Strong', 'Average' or 'Weak'), "
    "qualified (boolean), reason (2 sentences), missing_skills (list of strings). " + FAIRNESS
)


# =====================================================================
# DATABASE INTERACTION (SUPABASE)
# =====================================================================
@st.cache_resource
def get_supabase():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        from supabase import create_client
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        return None


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def save_candidate(email, **fields):
    sb = get_supabase()
    if sb is None:
        return False
    try:
        fields["email"] = email
        fields["updated_at"] = now_iso()
        sb.table("candidates").upsert(fields, on_conflict="email").execute()
        return True
    except Exception as e:
        st.warning(f"Could not save to database: {e}")
        return False


def get_candidate(email):
    sb = get_supabase()
    if sb is None:
        return None
    try:
        res = sb.table("candidates").select("*").eq("email", email).execute()
        return res.data[0] if res.data else None
    except Exception as e:
        st.warning(f"Could not load from database: {e}")
        return None


def list_candidates():
    sb = get_supabase()
    if sb is None:
        return []
    try:
        return sb.table("candidates").select("*").execute().data or []
    except Exception as e:
        st.warning(f"Could not load candidates: {e}")
        return []


def delete_candidate(email):
    sb = get_supabase()
    if sb is None:
        return False
    try:
        sb.table("candidates").delete().eq("email", email).execute()
        return True
    except Exception as e:
        st.warning(f"Could not delete candidate: {e}")
        return False


def log_event(actor, action, target_email="", details=""):
    sb = get_supabase()
    if sb is None:
        return
    try:
        sb.table("audit_log").insert({
            "actor": actor,
            "action": action,
            "target": cid(target_email) if target_email else "",
            "details": details[:500],
        }).execute()
    except Exception:
        pass


# =====================================================================
# UTILITIES & CALCULATIONS
# =====================================================================
def cid(email):
    return "C-" + hashlib.sha256((email + APP_SALT).encode()).hexdigest()[:6].upper()


def hash_pin(email, pin):
    return hashlib.sha256(f"{email}:{pin}:{APP_SALT}".encode()).hexdigest()


def extract_text(file):
    name = file.name.lower()
    if name.endswith(".pdf"):
        return "\n".join((p.extract_text() or "") for p in PdfReader(file).pages)
    if name.endswith(".docx"):
        return "\n".join(p.text for p in Document(file).paragraphs)
    return file.getvalue().decode("utf-8", errors="ignore")


SENSITIVE_LINE = re.compile(
    r"(date of birth|\bdob\b|\bgender\b|\bsex\b|marital|nationality|religion|\bage\s*:)", re.I
)


def redact_pii(text):
    text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[email]", text)
    text = re.sub(r"https?://\S+|www\.\S+", "[link]", text)
    text = re.sub(r"\+?\d[\d\s\-()]{8,}\d", "[phone]", text)
    return "\n".join(l for l in text.splitlines() if not SENSITIVE_LINE.search(l))


def as_list(x):
    return x if isinstance(x, list) else []


def to_int(x, default=0):
    try:
        return max(0, min(100, int(x)))
    except Exception:
        return default


def role_name(r):
    return r.get("role", "") if isinstance(r, dict) else str(r)


def profile_brief(profile):
    return json.dumps({k: profile.get(k) for k in (
        "summary", "skills", "qualifications", "projects", "experience_years", "primary_domain")},
        ensure_ascii=False)


def resume_label(score):
    return "Strong" if score >= 75 else "Average" if score >= 50 else "Weak"


def final_score(c):
    base = c.get("score") or 0
    hr = c.get("human_rating")
    return round((1 - HUMAN_WEIGHT) * base + HUMAN_WEIGHT * hr * 20) if hr else base


def effort_check(questions, answers, seconds):
    flags, points = [], 0
    n = len(answers)
    filled = [a for a in answers if a.strip()]
    empty = n - len(filled)
    if empty:
        flags.append(f"{empty} of {n} questions left blank")
        points += 2 if empty >= n / 2 else 1
    if filled:
        avg = sum(len(a.split()) for a in filled) / len(filled)
        if avg < 15:
            flags.append(f"Very short answers (average {avg:.0f} words)")
            points += 2
        elif avg < 30:
            flags.append(f"Brief answers (average {avg:.0f} words)")
            points += 1
        if any(SequenceMatcher(None, filled[i].lower(), filled[j].lower()).ratio() > 0.8
               for i in range(len(filled)) for j in range(i + 1, len(filled))):
            flags.append("Near-identical answers to different questions")
            points += 2
        if any(a.strip() and SequenceMatcher(None, q.lower(), a.lower()).ratio() > 0.7
               for q, a in zip(questions, answers)):
            flags.append("Answer repeats the question")
            points += 2
        for a in filled:
            words = a.lower().split()
            if (re.search(r"(.)\1{4,}", a) or max(len(w) for w in words) > 30
                    or (len(words) >= 8 and len(set(words)) / len(words) < 0.35)):
                flags.append("Gibberish or heavily repeated text")
                points += 3
                break
    if seconds is not None and seconds < 5 * max(n, 1) and filled:
        flags.append(f"Submitted fast ({int(seconds)}s)")
    level = "Not serious" if points >= 3 else "Low effort" if points >= 1 else "Serious"
    return {"level": level, "flags": flags, "seconds": int(seconds or 0)}


def compute_scores(answer_scores, ev, review):
    answers = round(sum(answer_scores) / max(len(answer_scores), 1))
    ver, unv = len(as_list(ev.get("verified_claims"))), len(as_list(ev.get("unverified_claims")))
    evidence = round(100 * ver / (ver + unv)) if (ver + unv) else 50
    projects = to_int((review or {}).get("project_evidence_score"), 50)
    resume = to_int((review or {}).get("resume_strength_score"), 50)
    total = round(
        WEIGHTS["answers"] * answers + WEIGHTS["evidence"] * evidence
        + WEIGHTS["projects"] * projects + WEIGHTS["resume"] * resume
    )
    return {"answers": answers, "evidence": evidence, "projects": projects, "resume": resume,
            "total": total, "weights": WEIGHTS}, total


# =====================================================================
# CANDIDATE PORTAL
# =====================================================================
def delete_section(cand):
    with st.expander("Delete my data"):
        st.write("This permanently removes your profile, answers and roadmap.")
        sure = st.checkbox("Yes, delete everything I submitted")
        if st.button("Delete my data") and sure:
            log_event("candidate", "data_deleted", cand["email"])
            if delete_candidate(cand["email"]):
                st.session_state.pop("cand", None)
                st.success("Your data has been deleted.")
                st.rerun()


def candidate_portal():
    st.title("🎯 Candidate Career Coach")
    st.write("Upload your resume. The AI reads it, tests what you really know, suggests roles you can "
             "apply for, and builds a roadmap if you want to transition to another role.")
    if get_supabase() is None:
        st.error("Database is not configured. Add SUPABASE_URL and SUPABASE_KEY to .streamlit/secrets.toml")
        return

    with st.expander("How you are scored (transparency)"):
        st.markdown(
            f"- **{int(WEIGHTS['answers'] * 100)}%** quality of your answers to the domain questions\n"
            f"- **{int(WEIGHTS['evidence'] * 100)}%** how well your answers prove the skills on your resume\n"
            f"- **{int(WEIGHTS['projects'] * 100)}%** how concrete your project experience is\n"
            f"- **{int(WEIGHTS['resume'] * 100)}%** resume clarity and domain strength\n\n"
            "The AI is explicitly instructed to ignore your name, gender, age, college, location, and career gaps. "
            "**The AI never rejects candidates.** It provides recommendations; a human makes all decisions."
        )

    c1, c2, c3 = st.columns(3)
    name = c1.text_input("Your name")
    email = c2.text_input("Your email").strip().lower()
    pin = c3.text_input("Create / enter your PIN (min 4 characters)", type="password")
    uploaded = st.file_uploader("Upload your resume (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"])
    consent = st.checkbox(CONSENT_TEXT)

    b1, b2, _ = st.columns([1, 1, 4])
    analyze = b1.button("Analyze my resume", type="primary")
    load_saved = b2.button("Load my saved results")

    if load_saved:
        row = get_candidate(email) if email else None
        if not row or not row.get("profile"):
            st.info("No saved results found for this email.")
        elif row.get("pin_hash") and row["pin_hash"] != hash_pin(email, pin):
            st.error("Invalid PIN.")
        else:
            st.session_state.cand = row
            st.rerun()

    if analyze:
        existing = get_candidate(email) if email else None
        if not email or "@" not in email:
            st.error("Please enter a valid email address.")
        elif len(pin) < 4:
            st.error("Please choose a PIN with at least 4 characters.")
        elif not consent:
            st.error("Please tick the consent box to proceed.")
        elif uploaded is None:
            st.error("Please upload your resume.")
        elif existing and existing.get("pin_hash") and existing["pin_hash"] != hash_pin(email, pin):
            st.error("This email is already registered. Enter the PIN you created previously.")
        else:
            text = redact_pii(extract_text(uploaded).strip())
            if len(text) < 100:
                st.error("Could not read text from this file. Please upload a clear text PDF or DOCX.")
            else:
                body = f"RESUME:\n{text[:MAX_RESUME_CHARS]}"
                with st.spinner("Analyzing resume..."):
                    profile = ask_ai(ANALYZE_SYSTEM, body, as_json=True)
                    review = ask_ai(REVIEW_SYSTEM, body, as_json=True) if profile else None
                if profile and review:
                    profile["questions"] = [str(q) for q in as_list(profile.get("questions"))][:5]
                    profile.pop("name", None)
                    if not profile["questions"]:
                        st.error("The AI did not generate assessment questions. Please try again.")
                    else:
                        review["resume_label"] = resume_label(to_int(review.get("resume_strength_score"), 50))
                        save_candidate(
                            email, name=name, pin_hash=hash_pin(email, pin), consent_at=now_iso(),
                            resume_text=None,
                            profile=profile, resume_review=review, qa=None, evaluation=None,
                            effort=None, breakdown=None, category=None, level=None, score=None,
                            status="Not assessed", target_role=None, roadmap=None, model_used=OLLAMA_MODEL,
                        )
                        log_event("candidate", "resume_analyzed", email)
                        st.session_state.cand = get_candidate(email)
                        st.session_state.pop("started_at", None)
                        st.rerun()

    cand = st.session_state.get("cand")
    if not cand:
        return
    email = cand["email"]
    profile = cand["profile"] or {}
    review = cand.get("resume_review") or {}

    st.divider()
    st.subheader(f"Profile: {cand.get('name') or 'You'}")
    st.write(profile.get("summary", ""))
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("**Skills**")
        st.write(", ".join(str(s) for s in as_list(profile.get("skills"))) or "-")
        st.markdown("**Qualifications**")
        for q in as_list(profile.get("qualifications")):
            st.write(f"- {q}")
    with p2:
        st.markdown("**Projects**")
        for p in as_list(profile.get("projects")):
            if isinstance(p, dict):
                st.write(f"- **{p.get('title', '')}**: {p.get('description', '')}")
            else:
                st.write(f"- {p}")

    if review:
        st.markdown(f"#### Your resume evaluation for *{review.get('target_profile', 'your domain')}*: "
                    f"**{review.get('resume_label', '-')}** ({review.get('resume_strength_score', '-')}/100)")
        r1, r2, r3 = st.columns(3)
        r1.markdown("**Strengths**")
        for s in as_list(review.get("strong_points")):
            r1.write(f"- {s}")
        r2.markdown("**Areas for Improvement**")
        for s in as_list(review.get("weak_points")):
            r2.write(f"- {s}")
        r3.markdown("**Optimization Tips**")
        for s in as_list(review.get("improvement_tips")):
            r3.write(f"- {s}")

    st.divider()
    questions = as_list(profile.get("questions"))

    if not cand.get("evaluation"):
        st.subheader("Step 2: Practical Knowledge Assessment")
        st.caption("Answer in your own words with concrete details.")
        st.session_state.setdefault("started_at", time.time())
        with st.form("qa_form"):
            answers = [st.text_area(f"Q{i + 1}. {q}", key=f"ans_{i}", height=130)
                       for i, q in enumerate(questions)]
            submitted = st.form_submit_button("Submit answers")
        if submitted:
            if not any(a.strip() for a in answers):
                st.error("Please answer at least one question.")
            else:
                seconds = time.time() - st.session_state.get("started_at", time.time())
                eff = effort_check(questions, answers, seconds)
                qa_text = "\n\n".join(f"Q{i + 1}: {q}\nAnswer: {a or '(no answer)'}"
                                      for i, (q, a) in enumerate(zip(questions, answers)))
                with st.spinner("Evaluating response quality..."):
                    ev = ask_ai(EVAL_SYSTEM, f"PROFILE:\n{profile_brief(profile)}\n\n{qa_text}", as_json=True)
                if ev:
                    scores = [to_int(s) for s in as_list(ev.get("scores"))]
                    scores = (scores + [0] * len(questions))[:len(questions)]
                    breakdown, total = compute_scores(scores, ev, review)
                    if eff["level"] != "Serious":
                        status = "Needs another attempt"
                    elif total >= REVIEW_THRESHOLD:
                        status = "Qualified"
                    else:
                        status = "Needs Improvement"
                    qa = [{"question": q, "answer": a, "score": s}
                          for q, a, s in zip(questions, answers, scores)]
                    save_candidate(
                        email, qa=qa, evaluation=ev, effort=eff, breakdown=breakdown,
                        category=ev.get("category"), level=ev.get("level"),
                        score=total, status=status, model_used=OLLAMA_MODEL,
                    )
                    log_event("candidate", "answers_submitted", email, f"effort={eff['level']} score={total}")
                    st.session_state.pop("started_at", None)
                    st.session_state.cand = get_candidate(email)
                    st.rerun()
        delete_section(cand)
        return

    ev = cand["evaluation"]
    bd = cand.get("breakdown") or {}
    eff = cand.get("effort") or {"level": "Serious", "flags": []}
    st.subheader("Assessment Result")
    m1, m2, m3 = st.columns(3)
    m1.metric("Overall Score", f"{cand.get('score')}/100")
    m2.metric("Assessed Level", ev.get("level", "-"))
    m3.metric("AI Recommendation", cand.get("status", "-"))

    if eff["level"] != "Serious":
        st.warning(
            "Your answers were flagged for potential quality or effort issues (" + "; ".join(eff.get("flags", [])) + "). "
            "You can retake the assessment with more detailed responses."
        )
    if bd:
        st.markdown("**Score Calculation Breakdown**")
        st.dataframe(pd.DataFrame([
            {"Component": "Answer quality", "Your score": bd.get("answers"), "Weight": f"{int(WEIGHTS['answers'] * 100)}%"},
            {"Component": "Skills proven by answers", "Your score": bd.get("evidence"), "Weight": f"{int(WEIGHTS['evidence'] * 100)}%"},
            {"Component": "Project evidence", "Your score": bd.get("projects"), "Weight": f"{int(WEIGHTS['projects'] * 100)}%"},
            {"Component": "Resume strength", "Your score": bd.get("resume"), "Weight": f"{int(WEIGHTS['resume'] * 100)}%"},
        ]), hide_index=True)

    st.write(f"**Best-fit Category:** {ev.get('category', '-')}")
    st.write(ev.get("reasoning", ""))
    s1, s2 = st.columns(2)
    with s1:
        st.markdown("**Demonstrated Strengths**")
        for s in as_list(ev.get("strengths")):
            st.write(f"- {s}")
    with s2:
        st.markdown("**Identified Gaps**")
        for g in as_list(ev.get("gaps")):
            st.write(f"- {g}")

    if cand.get("employer_message"):
        st.info(f"**Employer Feedback:** {cand['employer_message']}")

    a1, a2 = st.columns(2)
    if a1.button("Retake questions"):
        save_candidate(email, evaluation=None, qa=None, effort=None, breakdown=None,
                       score=None, status="Not assessed", category=None, level=None)
        for i in range(len(questions)):
            st.session_state.pop(f"ans_{i}", None)
        st.session_state.pop("started_at", None)
        st.session_state.cand = get_candidate(email)
        st.rerun()

    with a2.expander("Request Human Review"):
        reason = st.text_area("Explain what you believe was overlooked", key="appeal_text")
        if st.button("Submit Review Request"):
            if not reason.strip():
                st.error("Please provide a brief explanation.")
            else:
                save_candidate(email, appeal_status="Requested", appeal_reason=reason)
                log_event("candidate", "appeal_requested", email)
                st.session_state.cand = get_candidate(email)
                st.rerun()
        if cand.get("appeal_status"):
            st.write(f"Appeal status: **{cand['appeal_status']}**")

    st.divider()
    st.subheader("Step 3: Recommended Roles")
    roles = as_list(profile.get("suggested_roles"))
    for r in roles:
        if isinstance(r, dict):
            st.write(f"- **{r.get('role', '')}**: {r.get('reason', '')}")
        else:
            st.write(f"- {r}")

    st.divider()
    st.subheader("Step 4: Career Development Roadmap")
    other = "Other Role..."
    options = [role_name(r) for r in roles if role_name(r)] + [other]
    choice = st.selectbox("Select Target Role", options)
    target = st.text_input("Enter custom target role") if choice == other else choice

    if st.button("Generate Roadmap", type="primary"):
        if not target.strip():
            st.error("Please enter a target role.")
        else:
            user_msg = (f"TARGET ROLE: {target}\n\nCURRENT PROFILE:\n{profile_brief(profile)}\n\n"
                        f"CURRENT LEVEL: {ev.get('level')}\nGAPS: {ev.get('gaps')}")
            with st.spinner("Generating personalized roadmap..."):
                roadmap = ask_ai(ROADMAP_SYSTEM, user_msg)
            if roadmap:
                save_candidate(email, target_role=target, roadmap=roadmap)
                st.session_state.cand = get_candidate(email)
                st.rerun()

    if cand.get("roadmap"):
        st.markdown(f"## Roadmap to: {cand.get('target_role')}")
        st.markdown(cand["roadmap"])
        st.download_button("Download Roadmap", cand["roadmap"], file_name="career_roadmap.md")

    delete_section(cand)


# =====================================================================
# EMPLOYER PORTAL (COMMAND CENTER INTERFACE WITH COMPLETE DETAILS)
# =====================================================================
def who(c, blind):
    if blind and c.get("decision") != "Shortlisted":
        return cid(c["email"])
    return f"{c.get('name') or '-'} <{c['email']}> [{cid(c['email'])}]"


def candidate_compact(c):
    p, ev = c.get("profile") or {}, c.get("evaluation") or {}
    qa = [{"q": x.get("question"), "a": (x.get("answer") or "")[:500], "score": x.get("score")}
          for x in as_list(c.get("qa"))]
    return json.dumps({
        "skills": p.get("skills"), "qualifications": p.get("qualifications"),
        "projects": p.get("projects"), "experience_years": p.get("experience_years"),
        "assessment_score": c.get("score"), "strengths": ev.get("strengths"),
        "gaps": ev.get("gaps"), "answers": qa,
    }, ensure_ascii=False)[:5000]


def employer_portal():
    # Top Bar: Platform Name + Dynamic Floating Status Badge
    top_col1, top_col2 = st.columns([3, 1])
    with top_col1:
        st.title("Candidate Command Center")
    with top_col2:
        st.markdown("""
            <div class="pipeline-status">
                <b>Hybrid AI Pipeline Status</b><br>
                🟢 <b>Groq API</b> (Primary - Active/Sub-second)<br>
                🟢 <b>Ollama</b> (Fallback - Standby)
            </div>
        """, unsafe_allow_html=True)

    if get_supabase() is None:
        st.error("Database is not configured. Add SUPABASE_URL and SUPABASE_KEY to .streamlit/secrets.toml")
        return
    if EMPLOYER_PASSWORD:
        if st.sidebar.text_input("Employer Password", type="password") != EMPLOYER_PASSWORD:
            st.info("Please enter the employer password in the sidebar.")
            return

    blind = st.sidebar.toggle("Blind Review Mode (Hide identities)", value=True)
    all_c = [c for c in list_candidates() if c.get("profile")]
    assessed = [c for c in all_c if c.get("evaluation")]
    serious = [c for c in assessed if (c.get("effort") or {}).get("level", "Serious") == "Serious"]
    flagged = [c for c in assessed if c not in serious]
    appeals = [c for c in all_c if c.get("appeal_status") == "Requested"]

    if not all_c:
        st.info("No candidates registered in database.")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Candidates", len(all_c))
    m2.metric("Assessed (Serious)", len(serious))
    m3.metric("Needs Retake", len(flagged))
    m4.metric("Pending Appeals", len(appeals))

    tab_cmd, tab_job, tab_appeal, tab_gov = st.tabs([
        "📊 Command Center & Full Inspector", "🎯 Job Matcher",
        f"📥 Appeals ({len(appeals)})", "🛡️ Governance & Audit",
    ])

    # ---------- TAB 1: COMMAND CENTER & DETAILED INSPECTOR ----------
    with tab_cmd:
        left_col, right_col = st.columns([1.4, 1.3])

        # --- LEFT COLUMN: Candidate Evaluation Queue ---
        with left_col:
            st.subheader("Candidate Evaluation Queue")
            for c in all_c:
                c1, c2, c3, c4 = st.columns([1.2, 2, 2, 2])
                c1.write(f"**{cid(c['email'])}**")
                c2.write(c.get("category") or "Applicant")
                
                score_str = f"{c.get('score')}/100" if c.get('score') is not None else "Pending"
                c3.write(f"🎯 **{score_str}**")
                
                eff_level = (c.get("effort") or {}).get("level", "Serious")
                badge_text = "Verified Human" if eff_level == "Serious" else "Effort Flagged"
                badge_style = "status-verified" if eff_level == "Serious" else "status-flagged"
                c4.markdown(f'<span class="{badge_style}">{badge_text}</span>', unsafe_allow_html=True)
                st.divider()

        # --- RIGHT COLUMN: Candidate Detail Inspector ---
        with right_col:
            st.markdown("### Candidate Inspector `[BLIND PROFILE]`")
            cand_options = {f"{who(c, blind)} | Score: {c.get('score') if c.get('score') is not None else 'Pending'}": c for c in all_c}
            selected_label = st.selectbox("Select Candidate to Inspect", list(cand_options.keys()))
            c = cand_options[selected_label]

            p = c.get("profile") or {}
            ev = c.get("evaluation") or {}
            bd = c.get("breakdown") or {}
            eff = c.get("effort") or {}

            score_val = f"{c.get('score')}/100" if c.get("score") is not None else "Pending Assessment"
            st.markdown(f"## Composite Score: <span style='color:#3FB950;'>{score_val}</span>", unsafe_allow_html=True)

            # 2x2 Dark Metric Gauge Grid
            g1, g2 = st.columns(2)
            g1.metric("Answer Quality", f"{bd.get('answers', 0)}/100")
            g2.metric("Skill Claim Evidence", f"{bd.get('evidence', 0)}/100")

            g3, g4 = st.columns(2)
            g3.metric("Project Depth", f"{bd.get('projects', 0)}/100")
            g4.metric("Resume Alignment", f"{bd.get('resume', 0)}/100")

            st.divider()
            
            # --- CANDIDATE RESUME & PROJECTS SUMMARY ---
            st.markdown("#### Candidate Profile & Resume Summary")
            st.write(p.get("summary", "No summary available."))
            st.markdown("**Skills:** " + (", ".join(str(s) for s in as_list(p.get("skills"))) or "-"))
            st.markdown("**Qualifications:** " + (", ".join(str(s) for s in as_list(p.get("qualifications"))) or "-"))
            
            with st.expander("📁 View Candidate Projects"):
                for pr in as_list(p.get("projects")):
                    if isinstance(pr, dict):
                        st.write(f"- **{pr.get('title', '')}**: {pr.get('description', '')}")
                    else:
                        st.write(f"- {pr}")

            # --- SUBMITTED ANSWERS & QUESTION SCORES ---
            if ev:
                st.divider()
                st.markdown("#### Proof-of-Execution Assessment & Answers")
                st.write("**AI Analysis:** " + ev.get("reasoning", ""))
                
                s1, s2 = st.columns(2)
                with s1:
                    st.markdown("**✅ Claims Proven**")
                    for s in as_list(ev.get("verified_claims")):
                        st.write(f"- {s}")
                with s2:
                    st.markdown("**❓ Unverified Claims**")
                    for s in as_list(ev.get("unverified_claims")):
                        st.write(f"- {s}")

                with st.expander("📝 View Submitted Candidate Answers & Ratings", expanded=True):
                    qa_items = as_list(c.get("qa"))
                    if qa_items:
                        for i, item in enumerate(qa_items, 1):
                            st.markdown(f"**Q{i}: {item.get('question')}** *(Score: {item.get('score')}/100)*")
                            st.info(item.get("answer") or "(No answer provided)")
                    else:
                        st.write("No submitted answers found for this candidate yet.")

                st.markdown("**Suggested Live Verification Probes:**")
                for probe in as_list(ev.get("interview_probes")):
                    st.write(f"- {probe}")

            st.divider()
            
            # --- AUDIT LOG & INTEGRITY ---
            st.markdown("#### Integrity & Effort Audit Log")
            st.write(f"⏱️ **Response Time:** {eff.get('seconds', 0)}s")
            st.write(f"🛡️ **Seriousness Status:** {eff.get('level', 'Serious')}")
            if eff.get("flags"):
                st.warning("Flags: " + "; ".join(eff["flags"]))

            st.divider()
            
            # --- HUMAN IN THE LOOP CONTROLS ---
            st.markdown("#### Human-in-the-Loop Controls")
            key = c["email"]
            rating = st.slider("Human Soft-skills Rating (1-5 Stars)", 0, 5, int(c.get("human_rating") or 0), key=f"hr_{key}")
            notes = st.text_area("Reviewer Justification Notes", value=c.get("human_notes") or "", key=f"hn_{key}")
            msg = st.text_area("Feedback Message to Candidate", value=c.get("employer_message") or "", key=f"em_{key}")
            
            options = ["New", "On hold", "Shortlisted", "Rejected"]
            cur = c.get("decision") if c.get("decision") in options else "New"
            decision = st.selectbox("Status Decision", options, index=options.index(cur), key=f"dec_{key}")

            if st.button("Save Evaluation & Decision", key=f"save_{key}", type="primary"):
                if decision == "Rejected" and not notes.strip():
                    st.error("Mandatory: Notes required when rejecting a candidate.")
                else:
                    if save_candidate(key, decision=decision, human_notes=notes, employer_message=msg, human_rating=rating or None):
                        log_event("employer", f"decision_{decision}", key, notes[:200])
                        st.success("Candidate record updated.")
                        st.rerun()

            if (c.get("decision") or "New") == "Shortlisted":
                st.success(f"🔓 Shortlisted Contact Revealed: {c.get('name') or '-'} | {c['email']}")

    # ---------- TAB 2: JOB MATCHING ----------
    with tab_job:
        job = st.text_area("Paste Job Requirements (Skills, experience, duties)",
                            placeholder="e.g. Senior Python Developer - FastAPI, PostgreSQL, AWS...")
        if st.button("Evaluate Candidate Match"):
            if not job.strip():
                st.error("Please enter job requirements.")
            elif not serious:
                st.error("No assessed candidates available.")
            else:
                results, bar = [], st.progress(0.0, text="Evaluating profiles...")
                for i, c in enumerate(serious):
                    res = ask_ai(MATCH_SYSTEM, f"JOB REQUIREMENT:\n{job}\n\nCANDIDATE:\n{candidate_compact(c)}", as_json=True)
                    if res:
                        results.append({
                            "Candidate": who(c, blind),
                            "Match Score": to_int(res.get("match_score")),
                            "Resume Fit": res.get("resume_fit", "-"),
                            "Qualified": "Yes" if res.get("qualified") else "No",
                            "Reasoning": res.get("reason", ""),
                            "Missing Skills": ", ".join(str(s) for s in as_list(res.get("missing_skills"))),
                        })
                    bar.progress((i + 1) / len(serious), text=f"Processed {i + 1} of {len(serious)}")
                bar.empty()
                results.sort(key=lambda r: r["Match Score"], reverse=True)
                st.session_state.ranking = results
        if st.session_state.get("ranking"):
            st.dataframe(pd.DataFrame(st.session_state.ranking), hide_index=True)

    # ---------- TAB 3: APPEALS ----------
    with tab_appeal:
        if not appeals:
            st.write("No active review appeals.")
        for c in appeals:
            with st.expander(f"{cid(c['email'])} - {c.get('category') or '-'} - Score {c.get('score')}"):
                st.write("**Candidate Appeal Argument:** " + (c.get("appeal_reason") or ""))
                reply = st.text_area("Employer Response Message", key=f"ap_{c['email']}",
                                     value=c.get("employer_message") or "")
                if st.button("Resolve Appeal", key=f"apb_{c['email']}"):
                    save_candidate(c["email"], appeal_status="Resolved", employer_message=reply)
                    log_event("employer", "appeal_resolved", c["email"], reply[:200])
                    st.rerun()

    # ---------- TAB 4: GOVERNANCE & AUDIT ----------
    with tab_gov:
        st.markdown(
            "### Compliance & Governance Active Safeguards\n"
            "- **Blind Review:** Identity details remain redacted until shortlist confirmation.\n"
            "- **Data Minimisation:** Contact info, demographics, and university prestige stripped prior to LLM analysis.\n"
            f"- **Transparent Formula:** Assessment weights ({int(WEIGHTS['answers']*100)}% Answers, "
            f"{int(WEIGHTS['evidence']*100)}% Skill Proof, {int(WEIGHTS['projects']*100)}% Projects, "
            f"{int(WEIGHTS['resume']*100)}% Resume).\n"
            "- **Human-in-the-Loop:** System cannot execute automated rejections; human justification required.\n"
            "- **Auditability:** Log trail recorded in database for review.\n"
            "- **Hybrid LLM Execution:** High-speed cloud processing with fallback to local processing."
        )
        st.markdown("### System Audit Logs")
        sb = get_supabase()
        try:
            logs = sb.table("audit_log").select("*").order("created_at", desc=True).limit(100).execute().data
            st.dataframe(pd.DataFrame(logs), hide_index=True) if logs else st.write("No logged events.")
        except Exception as e:
            st.warning(f"Audit log query failed: {e}")


# =====================================================================
# MAIN ROUTER
# =====================================================================
portal = st.sidebar.radio("Select Portal", ["1. Candidate", "2. Employer"])
if portal == "1. Candidate":
    candidate_portal()
else:
    employer_portal()