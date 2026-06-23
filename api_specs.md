# API Specifications

## 1. /evaluate-candidates - POST

Accepts a Job Description file (`.docx`) and a JSON string of candidate details. It validates both inputs, extracts text from the Job Description, and parses the candidates into a validated list of objects using Pydantic models.

**Request Body (`multipart/form-data`):**
- `jd_file`: (File) The Job Description document (must be a `.docx` file).
- `candidates_json`: (String) A JSON string containing an array of candidate objects following the Candidate schema.

**Response:**
```json
{
  "job_description": {
    "raw_text": "Extracted text from the docx file...",
    "paragraph_count": 42
  },
  "candidates": [
    {
      "candidate_id": "CAND_0001234",
      "profile": {
        "anonymized_name": "Candidate A",
        "headline": "Software Engineer",
        "summary": "Experienced developer...",
        "location": "New York, NY",
        "country": "USA",
        "years_of_experience": 5.5,
        "current_title": "Senior Engineer",
        "current_company": "Tech Corp",
        "current_company_size": "51-200",
        "current_industry": "Technology"
      },
      "career_history": [ ... ],
      "education": [ ... ],
      "skills": [ ... ],
      "certifications": [ ... ],
      "languages": [ ... ],
      "redrob_signals": { ... }
    }
  ]
}
```
