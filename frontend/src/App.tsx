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

interface EvaluationResponse {
  job_description: JobDescription;
  candidates: Candidate[];
}

function App() {
  const [jdFile, setJdFile] = useState<File | null>(null);
  const [candidatesFile, setCandidatesFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<EvaluationResponse | null>(null);

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
      setError("Please upload both a Job Description (.docx or .pdf) and Candidates Data (.json).");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Read the candidates JSON file as text
      const candidatesText = await candidatesFile.text();

      const formData = new FormData();
      formData.append('jd_file', jdFile);
      formData.append('candidates_json', candidatesText);

      const response = await axios.post<EvaluationResponse>('http://localhost:8000/evaluate-candidates', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResult(response.data);
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
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>Intelligent Candidate Discovery</h1>
        <p>Upload a job description and candidate profiles to evaluate.</p>
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
          <p>{jdFile ? 'Click to change' : 'Accepts .docx (or .pdf if supported by backend)'}</p>
          <input 
            type="file" 
            ref={jdInputRef} 
            onChange={handleJdFileChange} 
            accept=".docx,.pdf" 
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
          <p>{candidatesFile ? 'Click to change' : 'Accepts .json array of candidates'}</p>
          <input 
            type="file" 
            ref={candidatesInputRef} 
            onChange={handleCandidatesFileChange} 
            accept=".json" 
            className="file-input" 
          />
        </div>
      </div>

      <button 
        className="btn-submit" 
        onClick={handleSubmit} 
        disabled={loading || !jdFile || !candidatesFile}
      >
        {loading ? (
          <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
            <Loader2 className="spinner" size={24} /> Processing...
          </span>
        ) : (
          <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
            <UploadCloud size={24} /> Evaluate Candidates
          </span>
        )}
      </button>

      {/* Results Display */}
      {result && (
        <div className="results-section">
          <div className="result-card">
            <h2>Job Description Text ({result.job_description.paragraph_count} paragraphs)</h2>
            <div className="raw-text-box">
              {result.job_description.raw_text}
            </div>
          </div>

          <div className="result-card">
            <h2>Parsed Candidates ({result.candidates.length})</h2>
            <div className="candidate-list">
              {result.candidates.map((candidate) => (
                <div key={candidate.candidate_id} className="candidate-card">
                  <h3 className="candidate-name">{candidate.profile.anonymized_name}</h3>
                  <p><strong>{candidate.profile.headline}</strong></p>
                  <p>{candidate.profile.current_title} at {candidate.profile.current_company}</p>
                  <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
                    {candidate.profile.summary.substring(0, 100)}...
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
