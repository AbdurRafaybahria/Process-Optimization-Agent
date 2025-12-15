# 📋 Complete Workflow & Rule Implementation Guide

## 🔄 Complete Workflow (Request → Response)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        API REQUEST FLOW                                      │
│                                                                              │
│  POST /cms/optimize/{process_id}/json                                        │
│                        │                                                     │
│                        ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ 1. API/main.py (Entry Point)                                            │ │
│  │    - CMSClient.get_process_with_relations(process_id)                   │ │
│  │    - CMSClient.get_jobs_for_process() → Fetch real skills               │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                        │                                                     │
│                        ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ 2. multi_job_resolver.py (Multi-Job Resolution)                         │ │
│  │    - Resolve 1:N job-task relationships → 1:1                           │ │
│  │    - Skill matching using CMS skills                                     │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                        │                                                     │
│                        ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ 3. cms_transformer.py (Data Transformation)                             │ │
│  │    - Transform CMS format → Agent format                                 │ │
│  │    - Validate process data                                               │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                        │                                                     │
│                        ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ 4. intelligent_optimizer.py (Main Orchestrator)                         │ │
│  │    - Calls ProcessIntelligence for type detection                        │ │
│  │    - Selects appropriate optimizer strategy                              │ │
│  │    - Coordinates optimization                                            │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                        │                                                     │
│                        ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ 5. process_intelligence.py (Process Type Detection) ← SLM INTEGRATION   │ │
│  │    - Keyword matching (75+ healthcare, 100+ manufacturing terms)        │ │
│  │    - NLP semantic analysis                                               │ │
│  │    - Pattern analysis                                                    │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                        │                                                     │
│                        ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ 6. optimizers.py (Core Scheduling Engine)                               │ │
│  │    - ProcessOptimizer.optimize()                                         │ │
│  │    - Dependency detection, Resource matching, Scheduling                 │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                        │                                                     │
│                        ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ 7. analyzers.py + nlp_dependency_analyzer.py ← SLM INTEGRATION          │ │
│  │    - DependencyDetector (spaCy + Sentence Transformers)                 │ │
│  │    - DeadlockDetector                                                    │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                        │                                                     │
│                        ▼                                                     │
│                  JSON RESPONSE                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 File-by-File Rule Implementation

### 1. API/main.py (Entry Point)

| Rule | Implementation |
|------|----------------|
| Cost Calculation | Lines 586-604: `cost = duration × hourly_rate` |
| Current State | Lines 586-595: `SUM of all task durations` |
| Optimized State | Lines 597-604: `MAX end_hour (parallel)` |

---

### 2. multi_job_resolver.py (Job Resolution)

| Rule | Lines | Implementation |
|------|-------|----------------|
| Resource Matching | 416-460 | `_get_job_capabilities()` - Gets skills from CMS |
| Skill Matching | 458-540 | `_calculate_skill_match()` - Fuzzy semantic matching |
| Best Fit (≥90%) | 340-350 | `if best_match.match_percentage >= self.best_fit_threshold` |

---

### 3. process_intelligence.py (Process Type Detection) ⭐ SLM Integration Point

| Rule | Lines | Implementation |
|------|-------|----------------|
| Keyword Matching | 50-100 | `patterns[ProcessType.HEALTHCARE]['keywords']` - 75+ healthcare terms |
| Manufacturing Terms | 150-250 | `patterns[ProcessType.MANUFACTURING]['keywords']` - 100+ terms |
| "Patient" = 99% Healthcare | ~400 | Special confidence boost for "patient" keyword |
| Confidence Scoring | ~350-400 | Weighted combination of keyword + NLP + pattern scores |

---

### 4. optimizers.py (Core Scheduling)

| Rule | Lines | Implementation |
|------|-------|----------------|
| **Dependency Rules** | 70-90 | `_detect_and_apply_dependencies()` |
| Cyclic Detection → Reject | 105-125 | `deadlock_detector.detect_deadlocks()` - Rejects cyclic deps |
| **Resource Matching** | 230-280 | `_find_best_resource_simple()` |
| `has_any_skill()` | 236-240 | `if not resource.has_all_skills(required_skills)` |
| Resource ONE task at a time | 170-175 | `resource_next_available` tracking |
| Resource Availability | 170 | `resource_next_available = {r.id: 0.0 for r in process.resources}` |
| **Scheduling Priority** | 220-225 | `_calculate_task_priority()` |
| Dependencies first | 178-185 | Tasks with deps added to ready queue after deps complete |
| Longer duration priority | Implicit | Priority queue ordering |
| **Cost Calculation** | 157-160 | `cost = duration_hours * best_resource.hourly_rate` |

---

### 5. analyzers.py (Dependency Detection) ⭐ SLM Integration Point

| Rule | Lines | Implementation |
|------|-------|----------------|
| NLP Analysis | 60-75 | `NLPDependencyAnalyzer` initialization |
| spaCy Integration | 72 | `self.nlp = spacy.load('en_core_web_sm')` |
| Keyword Detection | 80-95 | `dependency_keywords` dictionary |
| Similarity Analysis | 17 | `TfidfVectorizer` + `cosine_similarity` |
| Validate Dependencies | 133-145 | `validate_dependencies()` |
| Remove Circular | 145 | `_remove_circular_dependencies()` |

---

### 6. nlp_dependency_analyzer.py ⭐ SLM Integration Point (Primary)

| Rule | Lines | Implementation |
|------|-------|----------------|
| **spaCy Model** | 124 | `self.nlp_model = spacy.load("en_core_web_sm")` |
| **Sentence Transformers** | 132 | `self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')` |
| Sequential Keywords | 70-78 | `SEQUENTIAL_KEYWORDS` list |
| Action Verb Analysis | 80-95 | `EARLY_ACTIONS`, `LATE_ACTIONS`, `MIDDLE_ACTIONS` |
| Domain-specific Rules | 98-110 | `INSURANCE_RULES` dictionary |
| Task Analysis | 140-180 | `analyze_task()` - Extracts verbs, objects, entities |
| Relationship Detection | 250+ | `determine_relationship()` - Confidence scoring |

---

### 7. models.py (Data Models)

| Rule | Lines | Implementation |
|------|-------|----------------|
| Task Dependencies | ~50 | `task.dependencies: Set[str]` |
| `task.can_start()` | ~80 | `def can_start(completed: Set[str])` |
| Resource Skills | ~120 | `resource.skills: List[Skill]` |
| `has_any_skill()` | ~140 | Skill matching method |

---

## 🤖 SLM (Small Language Model) Integration Points

### 1. spaCy (`en_core_web_sm`)

**Location**: 
- `nlp_dependency_analyzer.py` Line 124
- `analyzers.py` Line 72

**Used For**:
- Part-of-speech tagging (verb extraction)
- Named entity recognition
- Dependency parsing for task relationships
- Extracting action verbs from task descriptions

```python
self.nlp_model = spacy.load("en_core_web_sm")
doc = self.nlp_model(text)
for token in doc:
    if token.pos_ == "VERB":
        return token.lemma_
```

---

### 2. Sentence Transformers (`all-MiniLM-L6-v2`)

**Location**: `nlp_dependency_analyzer.py` Line 132

**Used For**:
- Semantic similarity between task descriptions
- Detecting implicit dependencies through meaning
- Finding related tasks for parallelization

```python
self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
# Computes embeddings for semantic comparison
```

---

### 3. TF-IDF + Cosine Similarity

**Location**: `analyzers.py` Line 17

**Used For**:
- Task description similarity analysis
- Detecting parallel execution opportunities

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
self.vectorizer = TfidfVectorizer(stop_words='english')
```

---

## 📊 Core Optimization Rules (Universal)

### 1. Dependency Rules
- ✅ Task must wait for ALL dependencies to complete before starting
- ✅ Dependencies detected via: NLP analysis, keywords
- ✅ Cyclic dependencies = deadlock → rejected

### 2. Resource Matching Rules
- ✅ Task requires specific skills → only resources with those skills can execute it
- ✅ Changed from `has_all_skills()` to `has_any_skill()` (partial match allowed)
- ✅ Resource can only work on ONE task at a time
- ✅ Resource availability tracked (earliest free time)

### 3. Scheduling Priority Rules
- ✅ Tasks with dependencies scheduled first (critical path)
- ✅ Longer duration tasks prioritized (reduce overall time)
- ✅ Tasks with fewer resource options prioritized (avoid bottlenecks)

### 4. Cost & Time Calculation
- ✅ Cost = Duration × Resource hourly rate
- ✅ Current state = SUM of all task durations (sequential)
- ✅ Optimized state = MAX end time (parallel execution)

---

## 🔍 Detection Rules

Process type detected by:
1. **Keyword matching** (75+ healthcare, 100+ manufacturing terms)
2. **NLP semantic analysis** (Sentence Transformers model)
3. **Pattern analysis** (sequential flow, parallelism, approval gates)
4. **Confidence scoring** (weighted combination of above)

**Special rule**: "patient" keyword = 99% healthcare confidence

---

## 📊 Summary Table

| Component | File | Key Rules Implemented |
|-----------|------|----------------------|
| **Entry Point** | `API/main.py` | Cost calculation, State comparison |
| **Job Resolution** | `multi_job_resolver.py` | Skill matching, 90% threshold |
| **Type Detection** | `process_intelligence.py` | Keyword matching, Confidence scoring |
| **Core Scheduling** | `optimizers.py` | Dependencies, Resource matching, Priority |
| **NLP Analysis** | `analyzers.py` | spaCy, TF-IDF, Dependency detection |
| **Advanced NLP** | `nlp_dependency_analyzer.py` | Sentence Transformers, Semantic analysis |
| **Data Models** | `models.py` | Task, Resource, Skill structures |
| **CMS Integration** | `cms_client.py` | API communication, Authentication |
| **Transformation** | `cms_transformer.py` | CMS → Agent format conversion |

---

## 📂 File Locations

```
process_optimization_agent/
├── Optimization/
│   ├── analyzers.py              # Dependency & Deadlock detection
│   ├── cms_client.py             # CMS API client
│   ├── cms_transformer.py        # Data transformation
│   ├── intelligent_optimizer.py  # Main orchestrator
│   ├── models.py                 # Data models (Task, Resource, Skill)
│   ├── multi_job_resolver.py     # Multi-job resolution
│   ├── nlp_dependency_analyzer.py # Advanced NLP analysis
│   ├── optimizers.py             # Core scheduling engine
│   ├── process_intelligence.py   # Process type detection
│   └── task_classifier.py        # Task classification
├── scenarios/
│   ├── healthcare/               # Healthcare-specific optimizers
│   ├── manufacturing/            # Manufacturing-specific optimizers
│   ├── insurance/                # Insurance-specific optimizers
│   └── banking/                  # Banking-specific optimizers
└── API/
    └── main.py                   # FastAPI entry point
```
