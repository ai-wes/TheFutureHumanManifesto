
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from enum import Enum

class DataSource(str, Enum):
    ARXIV = "arxiv"
    NEWS = "news"
    PATENTS = "patents"
    GDELT = "gdelt"
    SOCIAL = "social"
    MARKETS = "markets"

class EventType(str, Enum):
    BREAKTHROUGH = "breakthrough"
    SETBACK = "setback"
    REGULATION = "regulation"
    FUNDING = "funding"
    COLLABORATION = "collaboration"

class Domain(str, Enum):
    AGI = "artificial_general_intelligence"
    LONGEVITY = "biotechnology_longevity"
    BCI = "brain_computer_interfaces"
    NANOTECH = "nanotechnology"
    QUANTUM = "quantum_computing"
    SPACE = "space_colonization"
    GENETICS = "genetic_engineering"

class RawDataItem(BaseModel):
    """Raw data item from any source"""
    id: str
    source: DataSource
    title: str
    content: str
    url: Optional[str] = None
    author: Optional[str] = None
    published_date: Optional[datetime] = None
    extracted_date: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ProcessedEntity(BaseModel):
    """Processed entity for knowledge graph"""
    name: str
    entity_type: str
    domain: Domain
    confidence: float = Field(ge=0.0, le=1.0)
    properties: Dict[str, Any] = Field(default_factory=dict)
    source_ids: List[str] = Field(default_factory=list)

class ProcessedRelationship(BaseModel):
    """Processed relationship for knowledge graph"""
    source_entity: str
    target_entity: str
    relationship_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    properties: Dict[str, Any] = Field(default_factory=dict)
    source_ids: List[str] = Field(default_factory=list)

class ScenarioGenome(BaseModel):
    """Represents a scenario as an evolvable genome"""
    id: str
    technological_factors: List[str]
    social_factors: List[str]
    economic_factors: List[str]
    timeline: str
    key_events: List[str]
    domains: List[Domain]
    probability_weights: Dict[str, float] = Field(default_factory=dict)
    fitness_score: Optional[float] = None
    generation: int = 0
    parent_ids: List[str] = Field(default_factory=list)

class ForecastResult(BaseModel):
    """Result from probabilistic forecasting"""
    domain: Domain
    milestone: str
    probability: float = Field(ge=0.0, le=1.0)
    confidence_interval_lower: float
    confidence_interval_upper: float
    uncertainty: float
    forecast_date: datetime = Field(default_factory=datetime.now)
    model_used: str
    time_horizon: int  # days
    factors: List[str] = Field(default_factory=list)

class GeneratedScenario(BaseModel):
    """Final generated scenario with probability assessment"""
    id: str
    title: str
    narrative: str
    probability: float = Field(ge=0.0, le=1.0)
    confidence_interval: tuple[float, float]
    timeline: Dict[str, List[str]]  # year -> events
    domains_involved: List[Domain]
    key_factors: List[str]
    risks: List[str]
    opportunities: List[str]
    contradictions: List[str] = Field(default_factory=list)
    consistency_score: float = Field(ge=0.0, le=1.0, default=1.0)
    generated_date: datetime = Field(default_factory=datetime.now)

class BookChapter(BaseModel):
    """Represents a book chapter with scenarios"""
    chapter_id: str
    title: str
    theme: Domain
    scenarios: List[GeneratedScenario]
    introduction: str
    analysis: str
    conclusion: str
    created_date: datetime = Field(default_factory=datetime.now)

    @validator('scenarios')
    def validate_scenarios(cls, v):
        if len(v) < 3 or len(v) > 7:
            raise ValueError('Each chapter should have 3-7 scenarios')
        return v
