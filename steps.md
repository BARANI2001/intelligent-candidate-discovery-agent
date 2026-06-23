# Steps to Run – Intelligent Candidate Discovery Agent

## 1. Prerequisites

- Python **3.11+**
- **uv** package manager → https://docs.astral.sh/uv/getting-started/installation/
- **Node.js 18+** with npm

---

## 2. Clone & Configure

```bash
git clone <repo-url>
cd intelligent-candidate-discovery-hackathon/intelligent-candidate-discovery-agent
make install
```

If you have environment variables, set them up now (e.g. `cp .env.example .env`). 

---

## 3. Backend Setup

```bash
make backend
```

Backend runs at: **http://localhost:8000**

> Swagger UI available at: http://localhost:8000/docs

---

## 4. Frontend Setup

```bash
make ui
```

Frontend runs at: **http://localhost:5173**

---

## 5. Using the App

1. Open **http://localhost:5173** in your browser
2. Click **Upload Job Description** and select a `.docx` file
3. Click **Upload Candidates** and select the `candidates.json` file
4. Click **Evaluate Candidates**
5. Watch the processing complete and view the structured texts and valid candidates displayed on the screen.

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `uv: command not found` | Install uv: `curl -Ls https://astral.sh/uv/install.sh \| sh` (Mac/Linux/WSL) or `irm https://astral.sh/uv/install.ps1 \| iex` (Windows) |
| CORS error in browser | Ensure backend is running on port 8000 |
| `Validation Error` on backend | Ensure your `candidates.json` strictly matches the required schema fields. |

---

## 7. Output Format

The backend endpoint returns a structured response:

```json
{
  "job_description": {
    "raw_text": "Extracted text...",
    "paragraph_count": 10
  },
  "candidates": [
    {
      "candidate_id": "CAND_0001234",
      "profile": { ... },
      "career_history": [ ... ],
      "redrob_signals": { ... }
    }
  ]
}
```
