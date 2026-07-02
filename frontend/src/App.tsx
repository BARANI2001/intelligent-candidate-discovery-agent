import { useState, useRef } from 'react';
import axios from 'axios';
import { UploadCloud, FileText, Users, Loader2 } from 'lucide-react';
import './App.css';

interface JobDescription {
  raw_text: string;
  paragraph_count: number;
}

interface Profile {
  anonymized_name: string;
  headline: string;
  summary: string;
  current_title: string;
  current_company: string;
}

interface Candidate {
  candidate_id: string;
  profile: Profile;
}

interface RankedCandidate {
  candidate_id: string;
  rank: number;
  final_score: number;
  base_score: number;
  relevance_score: number;
  behavioral_score: number;
  applied_multipliers: Record<string, any>;
  justification: string;
  relevance_justification?: string;
  behavioral_summary?: string;
}

interface RankingResponse {
  job_description_summary: string;
  total_candidates: number;
  results: RankedCandidate[];
}

interface JobStatusResponse {
  job_id: string;
  status: string;
  error?: string;
  results?: RankingResponse;
}

function App() {
  const [jdFile, setJdFile] = useState<File | null>(null);
  const [candidatesFile, setCandidatesFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [pollingStatus, setPollingStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RankingResponse | null>(null);
  const [candidatesMap, setCandidatesMap] = useState<Record<string, Candidate>>({});

  const jdInputRef = useRef<HTMLInputElement>(null);
  const candidatesInputRef = useRef<HTMLInputElement>(null);

  const handleJdFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setJdFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleCandidatesFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setCandidatesFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleSubmit = async () => {
    if (!jdFile || !candidatesFile) {
      setError("Please upload both a Job Description (.docx) and Candidates Data (.json or .jsonl).");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Read the candidates file as text and parse mapping for UI display
      const candidatesText = await candidatesFile.text();
      let parsedArray: Candidate[] = [];
      try {
        parsedArray = JSON.parse(candidatesText);
      } catch (e) {
        // Fallback to JSONL parsing (one JSON object per line)
        parsedArray = candidatesText
          .split('\n')
          .map(line => line.trim())
          .filter(line => line.length > 0)
          .map(line => JSON.parse(line));
      }

      const map: Record<string, Candidate> = {};
      if (Array.isArray(parsedArray)) {
        parsedArray.forEach(c => {
          if (c && c.candidate_id) {
            map[c.candidate_id] = c;
          }
        });
      }
      setCandidatesMap(map);

      const formData = new FormData();
      formData.append('jd_file', jdFile);
      formData.append('candidates_json', candidatesText);

      const response = await axios.post<JobStatusResponse>('http://localhost:8000/rank-candidates', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.status === 'completed' && response.data.results) {
        setResult(response.data.results);
        setLoading(false);
      } else {
        const currentJobId = response.data.job_id;
        setPollingStatus("Evaluating candidates... Polling server every 5 seconds");
        
        const intervalId = setInterval(async () => {
          try {
            const statusRes = await axios.get<JobStatusResponse>(`http://localhost:8000/ranking-status/${currentJobId}`);
            if (statusRes.data.status === 'completed' && statusRes.data.results) {
              clearInterval(intervalId);
              setResult(statusRes.data.results);
              setLoading(false);
              setPollingStatus(null);
            } else if (statusRes.data.status === 'failed') {
              clearInterval(intervalId);
              setError(statusRes.data.error || "Background ranking job failed.");
              setLoading(false);
              setPollingStatus(null);
            }
          } catch (pollErr: any) {
            console.error("Polling error:", pollErr);
          }
        }, 5000);
      }
    } catch (err: any) {
      console.error(err);
      if (err.response && err.response.data && err.response.data.detail) {
        // Handle FastAPI validation errors
        const detail = err.response.data.detail;
        if (Array.isArray(detail)) {
            setError(`Validation Error: ${detail[0].msg} at ${detail[0].loc?.join('.')}`);
        } else {
            setError(detail);
        }
      } else {
        setError(err.message || "An unexpected error occurred.");
      }
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>Intelligent Candidate Discovery</h1>
        <p>Upload a job description and candidate profiles to generate deterministic, rule-based AI rankings.</p>
      </header>

      {error && <div className="error-message">{error}</div>}

      <div className="upload-section">
        {/* JD Upload Zone */}
        <div 
          className={`file-drop-zone ${jdFile ? 'active' : ''}`}
          onClick={() => jdInputRef.current?.click()}
        >
          <FileText className="icon" size={48} />
          <h3>{jdFile ? jdFile.name : 'Upload Job Description'}</h3>
          <p>{jdFile ? 'Click to change' : 'Accepts .docx only'}</p>
          <input 
            type="file" 
            ref={jdInputRef} 
            onChange={handleJdFileChange} 
            accept=".docx" 
            className="file-input" 
          />
        </div>

        {/* Candidates Upload Zone */}
        <div 
          className={`file-drop-zone ${candidatesFile ? 'active' : ''}`}
          onClick={() => candidatesInputRef.current?.click()}
        >
          <Users className="icon" size={48} />
          <h3>{candidatesFile ? candidatesFile.name : 'Upload Candidates'}</h3>
          <p>{candidatesFile ? 'Click to change' : 'Accepts .json or .jsonl'}</p>
          <input 
            type="file" 
            ref={candidatesInputRef} 
            onChange={handleCandidatesFileChange} 
            accept=".json,.jsonl" 
            className="file-input" 
          />
        </div>
      </div>

      <button 
        className="btn-submit" 
        onClick={handleSubmit} 
        disabled={loading || Boolean(pollingStatus) || !jdFile || !candidatesFile}
      >
        {loading || pollingStatus ? (
          <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
            <Loader2 className="spinner" size={24} /> {pollingStatus || "Running Relevance & Behavioral Pipeline..."}
          </span>
        ) : (
          <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
            <UploadCloud size={24} /> Rank Candidates
          </span>
        )}
      </button>

      {/* Results Display */}
      {result && (
        <div className="results-section">
          <div className="result-card jd-summary-card">
            <h2>Job Description Summary ({result.total_candidates} Candidates Evaluated)</h2>
            <div className="raw-text-box">
              {result.job_description_summary}
            </div>
          </div>

          <div className="rankings-header">
            <h2>Ranked Candidates Leaderboard</h2>
          </div>

          <div className="ranked-candidate-list">
            {result.results.map((item) => {
              const cand = candidatesMap[item.candidate_id];
              return (
                <div key={item.candidate_id} className="ranked-card">
                  <div className="ranked-card-header">
                    <div className="rank-badge">#{item.rank}</div>
                    <div className="candidate-info">
                      <h3 className="candidate-name">
                        {cand?.profile?.anonymized_name || item.candidate_id}
                      </h3>
                      <div style={{ fontSize: '0.9rem', color: '#2563eb', fontWeight: 700, marginBottom: '0.35rem' }}>
                        Candidate ID: {item.candidate_id}
                      </div>
                      <p className="candidate-title">
                        <strong>{cand?.profile?.headline || 'Candidate Profile'}</strong>
                        {cand?.profile?.current_title ? ` — ${cand.profile.current_title} at ${cand.profile.current_company}` : ''}
                      </p>
                    </div>
                    <div className="score-badge">
                      <span className="score-value">{item.final_score.toFixed(1)}</span>
                      <span className="score-label">Final Score</span>
                    </div>
                  </div>

                  <div className="score-breakdown">
                    <div className="score-pill">Relevance: <strong>{item.relevance_score.toFixed(1)}</strong></div>
                    <div className="score-pill">Behavioral: <strong>{item.behavioral_score.toFixed(1)}</strong></div>
                    <div className="score-pill">Base: <strong>{item.base_score.toFixed(1)}</strong></div>
                  </div>

                  {item.applied_multipliers && Object.keys(item.applied_multipliers).length > 0 && (
                    <div className="multipliers-section">
                      <span className="multipliers-label">Signals Applied:</span>
                      <div className="multipliers-tags">
                        {Object.entries(item.applied_multipliers).map(([key, val]) => (
                          <span key={key} className="multiplier-tag">{key}: {String(val)}</span>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="justification-box">
                    <strong>Deterministic Justification:</strong>
                    <p>{item.justification}</p>
                    {item.relevance_justification && (
                      <p className="sub-justification"><em>Relevance Details:</em> {item.relevance_justification}</p>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
