from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field

EntityValue = Union[str, List[str]]



class ExtractedRule(BaseModel):
    condition: str
    action: str
    exception: Optional[str] = None
    evidence: str
    page_number: int = Field(..., ge=1)
    confidence: float = Field(..., ge=0.0, le=1.0)


class ChunkExtractionOutput(BaseModel):
    extracted_entities: Dict[str, EntityValue]
    extracted_rules: List[ExtractedRule]
    ambiguities: List[str]


class VerifiedRule(BaseModel):
    condition: str
    action: str
    exception: Optional[str] = None
    evidence: str
    page_number: int = Field(..., ge=1)
    supported: bool
    support_reason: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class VerificationOutput(BaseModel):
    verified_rules: List[VerifiedRule]
    ambiguities: List[str]


class FinalDecisionOutput(BaseModel):
    recommendation: str
    reasoning_summary: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    manual_review_needed: bool


class CriticOutput(BaseModel):
    passed: bool
    feedback: str


class OneShotAnalysisOutput(BaseModel):
    document_type: str
    extracted_entities: Dict[str, EntityValue]
    extracted_rules: List[ExtractedRule]
    verified_rules: List[VerifiedRule]
    ambiguities: List[str]
    final_decision: FinalDecisionOutput
