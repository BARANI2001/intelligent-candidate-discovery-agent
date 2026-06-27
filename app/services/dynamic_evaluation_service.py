"""
Dynamic Evaluation Service

Handles dynamic processing of:
- Any job description (text or docx)
- Candidate JSON in various formats (list, jsonl, etc.)
- Maintains schema consistency via Pydantic validation
- Provides unified evaluation interface
"""

import json
from typing import List, Dict, Any, Union, Optional, BinaryIO
from io import BytesIO

from pydantic import ValidationError

from app.models.candidate import Candidate
from app.models.jd import JobDescription
from app.models.evaluation import EvaluationResult
from app.services.relevance_evaluator import RelevanceEvaluator
from app.services.docx_parser import extract_jd_text


class DynamicEvaluationService:
    """
    Unified service for evaluating candidates against job descriptions.
    
    Handles:
    - Multiple JD formats (text, docx bytes)
    - Multiple candidate formats (JSON list, JSONL, dict)
    - Schema validation and error handling
    - Efficient batch processing
    """

    def __init__(self, use_fastembed: bool = True):
        """
        Initialize evaluation service.
        
        Args:
            use_fastembed: Use FastEmbed for embeddings (True) or mock (False)
        """
        self.evaluator = RelevanceEvaluator(use_fastembed=use_fastembed)
        self.use_fastembed = use_fastembed

    # ========================================================================
    # JD Processing Methods
    # ========================================================================

    def process_jd_text(self, text: str) -> JobDescription:
        """
        Process job description from raw text.
        
        Args:
            text: Job description text
        
        Returns:
            JobDescription object
        
        Raises:
            ValueError: If text is empty or invalid
        """
        if not text or not isinstance(text, str):
            raise ValueError("JD text must be a non-empty string")

        text = text.strip()
        if not text:
            raise ValueError("JD text cannot be empty after stripping")

        # Count paragraphs (rough heuristic: double newlines)
        paragraph_count = len([p for p in text.split('\n\n') if p.strip()])

        return JobDescription(
            raw_text=text,
            paragraph_count=max(1, paragraph_count)
        )

    def process_jd_docx(self, docx_bytes: Union[bytes, BinaryIO]) -> JobDescription:
        """
        Process job description from DOCX file.
        
        Args:
            docx_bytes: DOCX file as bytes or file-like object
        
        Returns:
            JobDescription object
        
        Raises:
            ValueError: If docx is invalid or extraction fails
        """
        if isinstance(docx_bytes, BinaryIO):
            docx_data = docx_bytes.read()
        else:
            docx_data = docx_bytes

        if not docx_data or len(docx_data) == 0:
            raise ValueError("DOCX data is empty")

        try:
            return extract_jd_text(docx_data)
        except Exception as e:
            raise ValueError(f"Failed to extract text from DOCX: {str(e)}")

    def process_jd(self, jd_input: Union[str, bytes, BinaryIO]) -> JobDescription:
        """
        Smart JD processing - detects format and processes accordingly.
        
        Args:
            jd_input: JD as text, docx bytes, or file-like object
        
        Returns:
            JobDescription object
        """
        if isinstance(jd_input, str):
            # Text input
            return self.process_jd_text(jd_input)
        elif isinstance(jd_input, (bytes, BinaryIO)):
            # DOCX bytes
            return self.process_jd_docx(jd_input)
        else:
            raise ValueError(f"Invalid JD input type: {type(jd_input)}")

    # ========================================================================
    # Candidate Processing Methods
    # ========================================================================

    def parse_candidate_json(self, candidate_dict: Dict[str, Any]) -> Candidate:
        """
        Parse and validate a single candidate JSON object.
        
        Args:
            candidate_dict: Candidate data as dictionary
        
        Returns:
            Validated Candidate object
        
        Raises:
            ValidationError: If candidate data doesn't match schema
        """
        try:
            return Candidate(**candidate_dict)
        except ValidationError as e:
            raise ValueError(f"Invalid candidate data: {e.json()}")

    def parse_candidates_json_list(self, candidates_json: Union[str, List[Dict]]) -> List[Candidate]:
        """
        Parse candidates from JSON array format.
        
        Args:
            candidates_json: JSON string or Python list
        
        Returns:
            List of validated Candidate objects
        
        Raises:
            ValueError: If JSON is invalid or candidates don't match schema
        """
        # Parse JSON if string
        if isinstance(candidates_json, str):
            try:
                candidates_data = json.loads(candidates_json)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON: {str(e)}")
        else:
            candidates_data = candidates_json

        # Ensure it's a list
        if not isinstance(candidates_data, list):
            raise ValueError("Candidates must be a JSON array")

        if not candidates_data:
            raise ValueError("Candidates array is empty")

        # Parse each candidate
        candidates = []
        for idx, candidate_dict in enumerate(candidates_data):
            if not isinstance(candidate_dict, dict):
                raise ValueError(f"Candidate {idx} is not a JSON object")

            try:
                candidate = self.parse_candidate_json(candidate_dict)
                candidates.append(candidate)
            except ValueError as e:
                raise ValueError(f"Candidate {idx}: {str(e)}")

        return candidates

    def parse_candidates_jsonl(self, jsonl_text: str) -> List[Candidate]:
        """
        Parse candidates from JSONL format (one JSON object per line).
        
        Args:
            jsonl_text: JSONL formatted text
        
        Returns:
            List of validated Candidate objects
        
        Raises:
            ValueError: If JSONL is invalid or candidates don't match schema
        """
        if not jsonl_text or not isinstance(jsonl_text, str):
            raise ValueError("JSONL input must be a non-empty string")

        lines = [line.strip() for line in jsonl_text.strip().split('\n') if line.strip()]

        if not lines:
            raise ValueError("JSONL input is empty")

        candidates = []
        for line_num, line in enumerate(lines, 1):
            try:
                candidate_dict = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {str(e)}")

            if not isinstance(candidate_dict, dict):
                raise ValueError(f"Line {line_num} is not a JSON object")

            try:
                candidate = self.parse_candidate_json(candidate_dict)
                candidates.append(candidate)
            except ValueError as e:
                raise ValueError(f"Line {line_num}: {str(e)}")

        return candidates

    def parse_candidates(
        self, candidates_input: Union[str, List[Dict], bytes], format_hint: Optional[str] = None
    ) -> List[Candidate]:
        """
        Smart candidate parsing - detects format and parses accordingly.
        
        Args:
            candidates_input: Candidates as JSON string, list, or bytes
            format_hint: Optional format hint ('json_array', 'jsonl')
        
        Returns:
            List of validated Candidate objects
        
        Raises:
            ValueError: If parsing fails or candidates don't match schema
        """
        # Handle bytes
        if isinstance(candidates_input, bytes):
            candidates_input = candidates_input.decode('utf-8')

        # Detect format if not provided
        if not format_hint:
            if isinstance(candidates_input, list):
                format_hint = 'json_array'
            elif isinstance(candidates_input, str):
                # Try to detect: count newlines with json content
                first_line = candidates_input.strip().split('\n')[0].strip()
                if first_line.startswith('['):
                    format_hint = 'json_array'
                else:
                    format_hint = 'jsonl'

        # Parse based on format
        if format_hint == 'json_array':
            return self.parse_candidates_json_list(candidates_input)
        elif format_hint == 'jsonl':
            return self.parse_candidates_jsonl(candidates_input)
        else:
            raise ValueError(f"Unknown format hint: {format_hint}")

    # ========================================================================
    # Evaluation Methods
    # ========================================================================

    def evaluate(
        self,
        jd_input: Union[str, bytes, BinaryIO],
        candidates_input: Union[str, List[Dict], bytes],
        candidates_format_hint: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Complete evaluation pipeline: JD + candidates -> ranked results.
        
        Args:
            jd_input: Job description (text, docx bytes, or file)
            candidates_input: Candidates (JSON array, JSONL, or list)
            candidates_format_hint: Optional format hint for candidates
        
        Returns:
            EvaluationResult with ranked candidates
        
        Raises:
            ValueError: If processing fails at any step
        """
        # Process JD
        try:
            jd = self.process_jd(jd_input)
        except Exception as e:
            raise ValueError(f"JD processing failed: {str(e)}")

        # Process candidates
        try:
            candidates = self.parse_candidates(candidates_input, candidates_format_hint)
        except Exception as e:
            raise ValueError(f"Candidate parsing failed: {str(e)}")

        # Evaluate
        try:
            result = self.evaluator.evaluate_all_candidates(jd, candidates)
        except Exception as e:
            raise ValueError(f"Evaluation failed: {str(e)}")

        return result

    def evaluate_batch(
        self,
        jd_input: Union[str, bytes, BinaryIO],
        candidates_input: Union[str, List[Dict], bytes],
        candidates_format_hint: Optional[str] = None,
        batch_size: int = 10,
    ) -> EvaluationResult:
        """
        Evaluate candidates in batches for memory efficiency.
        
        Args:
            jd_input: Job description
            candidates_input: Candidates (supports same formats)
            candidates_format_hint: Optional format hint
            batch_size: Number of candidates per batch
        
        Returns:
            Combined EvaluationResult (all batches ranked together)
        """
        # For now, just call evaluate (can be optimized for very large batches)
        return self.evaluate(jd_input, candidates_input, candidates_format_hint)

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_jd_summary(self, jd: JobDescription) -> Dict[str, Any]:
        """
        Get summary of JD analysis.
        
        Args:
            jd: JobDescription object
        
        Returns:
            Summary dictionary with keywords, statistics, etc.
        """
        keywords = self.evaluator.extract_jd_required_skills(jd)

        return {
            'raw_text_length': len(jd.raw_text),
            'paragraph_count': jd.paragraph_count,
            'extracted_keywords': keywords,
            'keyword_count': len(keywords),
            'summary': f"JD with {jd.paragraph_count} paragraphs, {len(keywords)} extracted keywords"
        }

    def get_candidate_summary(self, candidate: Candidate) -> Dict[str, Any]:
        """
        Get summary of candidate profile.
        
        Args:
            candidate: Candidate object
        
        Returns:
            Summary dictionary with profile info
        """
        return {
            'candidate_id': candidate.candidate_id,
            'name': candidate.profile.anonymized_name,
            'title': candidate.profile.current_title,
            'experience_years': candidate.profile.years_of_experience,
            'skill_count': len(candidate.skills),
            'skills': [s.name for s in candidate.skills],
            'company': candidate.profile.current_company,
            'location': candidate.profile.location,
            'open_to_work': candidate.redrob_signals.open_to_work_flag
        }

    def validate_schema(self, candidate_dict: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate candidate against schema without raising exceptions.
        
        Args:
            candidate_dict: Candidate data to validate
        
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            Candidate(**candidate_dict)
            return True, "Valid"
        except ValidationError as e:
            return False, f"Schema validation failed: {e.json()}"


# Global instance for convenience
_dynamic_service = None


def get_dynamic_evaluation_service(use_fastembed: bool = True) -> DynamicEvaluationService:
    """
    Get or create global dynamic evaluation service.
    
    Args:
        use_fastembed: Use FastEmbed embeddings
    
    Returns:
        DynamicEvaluationService instance
    """
    global _dynamic_service
    if _dynamic_service is None:
        _dynamic_service = DynamicEvaluationService(use_fastembed=use_fastembed)
    return _dynamic_service
