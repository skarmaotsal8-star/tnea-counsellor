from flask import Flask, render_template, request, jsonify
import requests, os, sys

app = Flask(__name__)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"

# ── System prompts ────────────────────────────────────────────────────────────

CHAT_SYSTEM_PROMPT = """You are TNEA Counsellor AI — a friendly, knowledgeable assistant
specialising in Tamil Nadu Engineering Admissions (TNEA).
Help students with cutoffs, branch selection, college comparisons, counselling rounds,
documents, fees, and Anna University affiliated colleges.
Be warm, encouraging and precise. Use bullet points for lists.
"""

COLLEGE_SYSTEM_PROMPT = """You are an expert on Indian engineering colleges, especially those affiliated to Anna University in Tamil Nadu.
When asked about a college, respond ONLY with a valid JSON object (no markdown, no extra text) in exactly this structure:

{
  "full_name": "Full official college name",
  "short_name": "Common short name or abbreviation",
  "location": "City, District, Tamil Nadu",
  "established": "Year (e.g. 1978)",
  "type": "Government / Government-Aided / Private / Deemed",
  "affiliation": "Anna University / Autonomous / Deemed University",
  "naac_grade": "A++ / A+ / A / B++ / B+ / B / Not Accredited",
  "nirf_rank": "National rank number or 'Not Ranked'",
  "overview": "3-4 sentence overview of the college — history, reputation, key achievements.",
  "total_seats": "Approximate total UG seats (number as string)",
  "campus_area": "e.g. 165 acres",
  "hostel": true or false,
  "website": "https://... (official website URL)",
  "branches": [
    {
      "name": "Branch full name",
      "code": "Branch code e.g. CS, EC",
      "seats": 60,
      "oc_cutoff": "Typical OC cutoff range e.g. 190-196",
      "bc_cutoff": "Typical BC cutoff range",
      "mbc_cutoff": "Typical MBC cutoff range",
      "sc_cutoff": "Typical SC cutoff range"
    }
  ],
  "fees": {
    "tuition_per_year": "Amount in INR e.g. ₹45,000 (Govt) / ₹1,20,000 (Private)",
    "hostel_per_year": "Amount or 'Not Available'",
    "total_4_years": "Approximate total cost"
  },
  "placements": {
    "avg_package": "e.g. ₹4.5 LPA",
    "highest_package": "e.g. ₹42 LPA",
    "placement_percent": "e.g. 85%",
    "top_recruiters": ["Company1", "Company2", "Company3", "Company4", "Company5"],
    "notable_alumni": "1-2 sentence mention of notable alumni if known, else empty string"
  },
  "facilities": ["Library", "Sports Complex", "Gymnasium", "Canteen", "Wi-Fi Campus", "Labs", "Auditorium"],
  "rankings": {
    "nirf": "Rank or 'Not Ranked'",
    "outlook": "Rank or 'Not Ranked'",
    "week": "Rank or 'Not Ranked'"
  },
  "pros": ["Pro point 1", "Pro point 2", "Pro point 3"],
  "cons": ["Con point 1", "Con point 2"],
  "nearby_colleges": ["College Name 1", "College Name 2", "College Name 3"]
}

Always fill all fields with realistic, best-effort data for Tamil Nadu engineering colleges. If exact data is unknown, provide a realistic estimate based on similar colleges. Never leave fields empty.
"""

OR_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost",
    "X-Title": "TNEA Counsellor"
}

# ── ML Backend ────────────────────────────────────────────────────────────────

try:
    sys.path.insert(0, os.path.dirname(__file__))
    from model_backend import recommend_colleges, df_final
    ML_AVAILABLE = df_final is not None and not df_final.empty
    COMMUNITIES  = sorted(df_final["Community"].unique().tolist()) if ML_AVAILABLE else ["OC","BC","BCM","MBC","SC","SCA","ST"]
    BRANCHES     = sorted(df_final["Branch Name"].unique().tolist()) if ML_AVAILABLE else []
    COLLEGE_LIST = sorted(df_final["College Name"].unique().tolist()) if ML_AVAILABLE else []
    print(f"[INFO] ML ready — {len(df_final)} records" if ML_AVAILABLE else "[WARN] No data found")
except Exception as e:
    ML_AVAILABLE = False
    COMMUNITIES  = ["OC","BC","BCM","MBC","SC","SCA","ST"]
    BRANCHES     = []
    COLLEGE_LIST = []
    print(f"[WARN] model_backend error: {e}")

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predictor")
def predictor():
    return render_template("predictor.html",
                           communities=COMMUNITIES,
                           branches=BRANCHES,
                           ml_available=ML_AVAILABLE)

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

@app.route("/college-search")
def college_search():
    query = request.args.get("q", "").strip()
    return render_template("college_search.html", query=query)

@app.route("/predict", methods=["POST"])
def predict():
    if not ML_AVAILABLE:
        return jsonify({"error": "Dataset not loaded. Add tnea*.csv files to dataset/ folder."})
    data = request.json
    dream, ambitious, safe, error = recommend_colleges(
        float(data.get("cutoff", 0)),
        data.get("community", "OC"),
        data.get("branches", [])
    )
    if error:
        return jsonify({"error": error})
    def to_list(df):
        return df.to_dict(orient="records") if df is not None and not df.empty else []
    return jsonify({"dream": to_list(dream), "ambitious": to_list(ambitious), "safe": to_list(safe)})

@app.route("/chat", methods=["POST"])
def chat():
    history  = request.json.get("history", [])
    question = request.json.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Please type a question."})
    messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}] + history[-10:] + \
               [{"role": "user", "content": question}]
    try:
        resp   = requests.post(OPENROUTER_URL, headers=OR_HEADERS,
                               json={"model": "openai/gpt-4o-mini", "messages": messages, "max_tokens": 600},
                               timeout=30)
        answer = resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        answer = f"⚠️ AI service error: {e}"
    return jsonify({"answer": answer})

@app.route("/api/college-details", methods=["POST"])
def college_details():
    """Fetch rich college info via AI — like Shiksha.com"""
    college_name = request.json.get("college", "").strip()
    if not college_name:
        return jsonify({"error": "College name required."})

    messages = [
        {"role": "system", "content": COLLEGE_SYSTEM_PROMPT},
        {"role": "user",   "content": f"Give me complete details about: {college_name}"}
    ]
    try:
        resp = requests.post(OPENROUTER_URL, headers=OR_HEADERS,
                             json={"model": "openai/gpt-4o-mini", "messages": messages, "max_tokens": 2000},
                             timeout=45)
        raw = resp.json()["choices"][0]["message"]["content"]
        # Strip any accidental markdown fences
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        import json
        data = json.loads(raw.strip())
        return jsonify({"success": True, "data": data})
    except Exception as e:
        return jsonify({"error": f"AI service error: {e}"})

@app.route("/api/college-suggestions", methods=["GET"])
def college_suggestions():
    """Return autocomplete suggestions from the dataset + a fixed popular list."""
    q = request.args.get("q", "").lower().strip()
    # Combine dataset colleges with popular ones
    popular = [
        "Anna University, Chennai",
        "PSG College of Technology, Coimbatore",
        "Coimbatore Institute of Technology",
        "Sri Venkateswara College of Engineering",
        "SSN College of Engineering",
        "Thiagarajar College of Engineering",
        "Government College of Technology, Coimbatore",
        "NIT Trichy",
        "CEG Anna University",
        "Kongu Engineering College",
        "RMK Engineering College",
        "Saveetha Engineering College",
        "Velammal Engineering College",
        "Rajalakshmi Engineering College",
        "SRM Institute of Science and Technology",
        "Vellore Institute of Technology",
        "Karpagam College of Engineering",
        "Sri Krishna College of Engineering",
        "Kumaraguru College of Technology",
        "Amrita School of Engineering",
    ]
    all_colleges = list(set(COLLEGE_LIST + popular))
    if q:
        results = [c for c in all_colleges if q in c.lower()][:10]
    else:
        results = popular[:10]
    return jsonify({"suggestions": results})

if __name__ == "__main__":
    app.run(debug=True)
