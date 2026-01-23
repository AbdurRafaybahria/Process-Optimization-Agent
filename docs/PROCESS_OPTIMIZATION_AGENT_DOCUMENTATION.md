# Process Optimization Agent - Complete Documentation

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [File Structure](#file-structure)
5. [Data Models](#data-models)
6. [Analyzers](#analyzers)
7. [Optimizers](#optimizers)
8. [NLP & AI Components](#nlp--ai-components)
9. [CMS Integration](#cms-integration)
10. [Visualization](#visualization)
11. [API Layer](#api-layer)
12. [Domain-Specific Scenarios](#domain-specific-scenarios)
13. [Scripts & Tests](#scripts--tests)
14. [Configuration & Dependencies](#configuration--dependencies)
15. [Workflow](#workflow)

---

## Overview

The **Process Optimization Agent** is an intelligent AI-powered system designed to optimize business processes through:

- **Automatic Process Type Detection**: Identifies healthcare, manufacturing, banking, insurance, and academic processes
- **Dependency Analysis**: Uses NLP and rule-based methods to detect task dependencies
- **Multi-Algorithm Optimization**: Employs Greedy, Q-Learning (RL), and Genetic algorithms
- **CMS Integration**: Seamlessly integrates with enterprise Content Management Systems
- **Cost & Time Optimization**: Reduces process duration and labor costs
- **What-If Analysis**: Simulates different scenarios for decision support

### Key Features

| Feature | Description |
|---------|-------------|
| **SLM-Powered Analysis** | Uses `all-MiniLM-L6-v2` Sentence Transformer for semantic understanding |
| **Multi-Job Resolution** | Resolves tasks with multiple job assignments to 1:1 relationships |
| **Deadlock Detection** | Identifies and resolves circular dependencies and resource conflicts |
| **Parallel Execution** | Maximizes parallelization opportunities to reduce total time |
| **Dual Optimization** | Optimizes both user experience and administrative efficiency |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API LAYER (FastAPI)                            │
│                               main.py                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CMS INTEGRATION LAYER                               │
│                   CMSClient ← → CMSDataTransformer                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PROCESS INTELLIGENCE LAYER                             │
│          ProcessIntelligence → TaskClassifier → ProcessType Detection       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ANALYZER LAYER                                     │
│    DependencyDetector ← NLPDependencyAnalyzer ← DeadlockDetector           │
│                        ↓                                                    │
│                  WhatIfAnalyzer                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OPTIMIZER LAYER                                     │
│                    IntelligentOptimizer (Master)                            │
│                             ↓                                               │
│    ┌─────────────────┬─────────────────┬──────────────────┐                │
│    │ ProcessOptimizer│ RLBasedOptimizer│ GeneticOptimizer │                │
│    │    (Greedy)     │  (Q-Learning)   │   (Evolution)    │                │
│    └─────────────────┴─────────────────┴──────────────────┘                │
│                             ↓                                               │
│              Domain-Specific Optimizers                                     │
│    (Healthcare, Insurance, Manufacturing, Banking)                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      VISUALIZATION LAYER                                    │
│                          Visualizer                                         │
│       (Summary Pages, Gantt Charts, Resource Allocation)                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### Component Interaction Flow

```
1. CMS Request → API (main.py)
2. API → CMSClient (fetch data)
3. CMSClient → CMSDataTransformer (transform format)
4. Transformer → ProcessIntelligence (detect type)
5. ProcessIntelligence → DependencyDetector (analyze dependencies)
6. DependencyDetector → NLPDependencyAnalyzer (semantic analysis)
7. Analyzer → IntelligentOptimizer (select optimizer)
8. Optimizer → Schedule (generate optimized schedule)
9. Schedule → Visualizer (create visualizations)
10. API Response → CMS (return results)
```

---

## File Structure

```
Main Process Optimization Agent/
├── API/
│   ├── __init__.py
│   ├── main.py                          # FastAPI server & endpoints
│   └── README.md
│
├── process_optimization_agent/
│   ├── __init__.py                      # Package exports
│   │
│   ├── Optimization/
│   │   ├── __init__.py
│   │   ├── models.py                    # Data models (Task, Resource, Process, Schedule)
│   │   ├── analyzers.py                 # DependencyDetector, DeadlockDetector, WhatIfAnalyzer
│   │   ├── optimizers.py                # ProcessOptimizer, RLBasedOptimizer, GeneticOptimizer
│   │   ├── nlp_dependency_analyzer.py   # SLM-powered semantic dependency analysis
│   │   ├── intelligent_optimizer.py     # Master optimizer orchestrating all strategies
│   │   ├── cms_client.py                # CMS API client with cookie authentication
│   │   ├── cms_transformer.py           # CMS data format transformer
│   │   ├── multi_job_resolver.py        # Resolves multi-job task assignments
│   │   ├── process_intelligence.py      # Process type detection
│   │   ├── task_classifier.py           # Task involvement classification
│   │   └── user_journey_optimizer.py    # User-centric optimization
│   │
│   ├── scenarios/
│   │   ├── healthcare/                  # Healthcare-specific optimizers
│   │   ├── insurance/                   # Insurance workflow optimizers
│   │   ├── manufacturing/               # Production line optimizers
│   │   └── banking/                     # Loan approval optimizers
│   │
│   └── visualization/
│       ├── __init__.py
│       └── visualizer.py                # Matplotlib/Seaborn visualizations
│
├── scripts/
│   ├── run_optimization.py              # CLI wrapper for optimization
│   └── run_rl_optimizer.py              # Main RL optimization script
│
├── tests/
│   ├── test_api_endpoints.py            # API endpoint tests
│   └── test_process_detection.py        # Process detection tests
│
├── examples/                            # Sample process JSON files
├── outputs/                             # Generated outputs
├── visualization_outputs/               # Visualization JSON outputs
├── docs/                                # Documentation
├── tools/                               # Utility scripts
│
├── requirements.txt                     # Python dependencies
└── Procfile                             # Railway deployment config
```

---

## Data Models

### File: `process_optimization_agent/Optimization/models.py`

Contains all core data structures used throughout the system.

### 1. Enums

```python
class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"

class SkillLevel(Enum):
    """Skill proficiency levels (1-4)"""
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4

class UserInvolvement(Enum):
    """Level of user/patient involvement in a task"""
    DIRECT = "direct"           # User actively participates
    PASSIVE = "passive"         # User present but not active
    ADMINISTRATIVE = "admin"    # User not involved at all
```

### 2. Skill

```python
@dataclass
class Skill:
    """Represents a skill with proficiency level"""
    name: str                   # Skill name (e.g., "Python", "Patient Care")
    level: SkillLevel           # Proficiency level (1-4)
    
    # Methods:
    # - to_dict(): JSON serialization
```

### 3. Task

```python
@dataclass
class Task:
    """Represents a work task with requirements and constraints"""
    id: str                                    # Unique identifier
    name: str                                  # Task name
    description: str                           # Task description
    duration_hours: float                      # Estimated duration
    required_skills: List[Skill]               # Skills needed
    dependencies: Set[str]                     # Task IDs this depends on
    order: Optional[int]                       # Explicit execution order
    deadline: Optional[float]                  # Hour when task should complete
    status: TaskStatus                         # Current status
    assigned_resource: Optional[str]           # Assigned resource ID
    start_hour: Optional[float]                # Scheduled start
    end_hour: Optional[float]                  # Scheduled end
    user_involvement: UserInvolvement          # User involvement level
    metadata: Dict[str, Any]                   # Additional data
    
    # Methods:
    # - can_start(completed_tasks): Check if dependencies satisfied
    # - to_dict(): JSON serialization
```

### 4. Resource

```python
@dataclass
class Resource:
    """Represents a person/resource with skills and availability"""
    id: str                                    # Unique identifier
    name: str                                  # Resource name
    skills: List[Skill]                        # Available skills
    hourly_rate: float                         # Cost per hour (default: $50)
    max_hours_per_day: float                   # Daily limit (default: 8h)
    total_available_hours: float               # Project total (default: 160h)
    current_workload: float                    # Hours assigned
    metadata: Dict[str, Any]                   # Additional data
    
    # Methods:
    # - has_skill(required_skill): Check skill match
    # - has_all_skills(required_skills): Check all skills
    # - is_available(start_hour, duration): Check availability
    # - get_skill_score(required_skills): Calculate match score (0-1)
```

### 5. Process

```python
@dataclass
class Process:
    """Represents a complete process to optimize"""
    id: str                                    # Unique identifier
    name: str                                  # Process name
    description: str                           # Process description
    company: str                               # Company name
    start_date: datetime                       # Project start date
    project_duration_hours: float              # Total project hours
    tasks: List[Task]                          # All tasks
    resources: List[Resource]                  # All resources
    metadata: Dict[str, Any]                   # Additional data
    
    # Methods:
    # - get_task_by_id(task_id): Find task
    # - get_resource_by_id(resource_id): Find resource
    # - calculate_critical_path(): Find critical path
```

### 6. ScheduleEntry

```python
@dataclass
class ScheduleEntry:
    """Single assignment in a schedule"""
    task_id: str                               # Task being scheduled
    resource_id: str                           # Assigned resource
    start_time: datetime                       # Start datetime
    end_time: datetime                         # End datetime
    start_hour: float                          # Start hour (0-based)
    end_hour: float                            # End hour (0-based)
    cost: float                                # Entry cost
```

### 7. Schedule

```python
@dataclass
class Schedule:
    """Complete schedule for a process"""
    process_id: str                            # Associated process
    entries: List[ScheduleEntry]               # All schedule entries
    total_duration_hours: float                # Total duration
    total_cost: float                          # Total cost
    metrics: Dict[str, Any]                    # Performance metrics
    deadlocks_detected: List[str]              # Detected deadlocks
    optimization_metrics: Dict[str, Any]       # Optimization stats
    
    # Methods:
    # - add_entry(entry): Add schedule entry
    # - calculate_metrics(process): Compute metrics
    # - to_dict(): JSON serialization
```

---

## Analyzers

### File: `process_optimization_agent/Optimization/analyzers.py`

Contains analysis components for dependency detection, deadlock resolution, and what-if analysis.

### 1. DependencyDetector

**Purpose**: Detects dependencies between tasks using multiple methods.

```python
class DependencyDetector:
    """Detects dependencies between tasks using NLP and rule-based methods"""
    
    def __init__(self, use_nlp=True, similarity_threshold=0.7, process_type="unknown"):
        # Initializes:
        # - spaCy NLP model (en_core_web_sm)
        # - TF-IDF Vectorizer
        # - NLP Dependency Analyzer (SLM)
        # - Keyword patterns for detection
```

**Detection Methods**:

| Method | Description | Confidence |
|--------|-------------|------------|
| **Rule-Based** | Keyword matching (after, before, requires) | 85% |
| **spaCy NLP** | Part-of-speech and dependency parsing | 80% |
| **TF-IDF Similarity** | Cosine similarity of task descriptions | 70% |
| **Pattern Matching** | Workflow patterns (design→develop→test) | 85% |
| **Resource-Based** | Tasks sharing same resource | 60% |
| **Domain-Specific** | Industry-specific rules | 90% |

**Key Methods**:

```python
def detect_dependencies(self, tasks, other_tasks=None, resources=None):
    """Main entry point - detects all dependencies"""
    # Returns: Dict[task_id, Set[dependency_ids]]

def detect_sequential_dependencies(self, tasks):
    """Find tasks that must run sequentially"""
    
def detect_parallel_opportunities(self, tasks):
    """Find tasks that can run in parallel"""

def validate_dependencies(self, tasks, dependencies):
    """Remove invalid and circular dependencies"""
```

**Dependency Keywords**:

```python
dependency_keywords = {
    'after': ['after', 'following', 'once', 'when', 'until', 'subsequent to'],
    'before': ['before', 'prior to', 'in preparation for', 'preceding'],
    'requires': ['requires', 'needs', 'depends on', 'relies on', 'using'],
    'parallel': ['meanwhile', 'concurrently', 'in parallel', 'simultaneously']
}
```

### 2. DeadlockDetector

**Purpose**: Identifies and resolves deadlock situations in process workflows.

```python
class DeadlockDetector:
    """Detects and resolves deadlocks in schedules"""
```

**Deadlock Types**:

| Type | Description | Resolution |
|------|-------------|------------|
| **Resource Contention** | Multiple tasks competing for same resource | Re-schedule |
| **Circular Dependencies** | A→B→C→A dependency chain | Break weakest link |
| **Capacity Exceeded** | Resource overloaded | Redistribute tasks |

**Algorithm**: Uses **Kahn's Algorithm** for topological sorting to detect cycles.

```python
def detect_deadlocks(self, process, schedule):
    """Detect all types of deadlocks"""
    # Returns: List of deadlock descriptions

def resolve_dependency_deadlocks(self, process, log_callback=None):
    """Automatically resolve circular dependencies"""
    # Removes edges to break cycles

def _remove_circular_dependencies(self, dependencies):
    """Kahn's algorithm implementation"""
    # Uses in-degree calculation and BFS
```

### 3. WhatIfAnalyzer

**Purpose**: Simulates different scenarios to predict optimization outcomes.

```python
class WhatIfAnalyzer:
    """Performs what-if analysis for process optimization"""
    
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.scenarios = []
```

**Scenario Parameters**:

```python
scenario_params = {
    'time_weight': 0.5,           # Weight for time optimization (0-1)
    'cost_weight': 0.5,           # Weight for cost optimization (0-1)
    'load_balancing_factor': 1.0, # Resource load balancing
    'resource_constraints': {},    # Resource-specific limits
    'deadline_constraints': {}     # Task deadline overrides
}
```

**Key Methods**:

```python
def analyze_scenario(self, process, scenario_name, params):
    """Run optimization with specific parameters"""
    # Returns: Schedule with scenario metrics

def compare_scenarios(self, process, scenarios):
    """Compare multiple what-if scenarios"""
    # Returns: Comparison report

def suggest_optimal_scenario(self, process):
    """Find best parameters automatically"""
```

---

## Optimizers

### File: `process_optimization_agent/Optimization/optimizers.py`

Contains three optimization algorithms: Greedy, Q-Learning, and Genetic.

### 1. BaseOptimizer (Abstract)

```python
class BaseOptimizer(ABC):
    """Abstract base class for all optimizers"""
    
    def __init__(self):
        self.dependency_detector = DependencyDetector()
        self.deadlock_detector = DeadlockDetector()
    
    @abstractmethod
    def optimize(self, process: Process) -> Schedule:
        """Optimize the process and return a schedule"""
        pass
```

### 2. ProcessOptimizer (Greedy)

**Purpose**: Fast, rule-based optimization using priority queue.

**Strategies**:

| Strategy | Focus | Priority Function |
|----------|-------|-------------------|
| `time` | Minimize duration | Earliest available resource |
| `cost` | Minimize cost | Cheapest qualified resource |
| `balanced` | Both time and cost | Weighted combination |

**Algorithm**:

```
1. Initialize priority queue with tasks having no dependencies
2. While queue not empty:
   a. Pop highest priority task
   b. Find best resource (based on strategy)
   c. Schedule task at earliest available time
   d. Update resource availability
   e. Add newly ready tasks to queue
3. Handle deadlocks if detected
4. Calculate final metrics
```

**Priority Calculation**:

```python
def _calculate_task_priority(self, task, process):
    """Lower number = scheduled earlier"""
    order_val = getattr(task, 'order', None)
    return float(order_val) if order_val is not None else float('inf')
```

**Resource Scoring**:

```python
def _calculate_resource_score_simple(self, task, resource, available_hour, workload):
    score = 0.0
    
    if self.strategy == "cost":
        score = 1000.0 / (resource.hourly_rate + 1.0)
    elif self.strategy == "time":
        score = 1000.0 / (available_hour + 1.0)
    else:  # balanced
        cost_score = 500.0 / (resource.hourly_rate + 1.0)
        time_score = 500.0 / (available_hour + 1.0)
        score = cost_score + time_score
    
    # Skill match bonus
    skill_score = resource.get_skill_score(task.required_skills)
    score += skill_score * 100.0
    
    # Workload penalty
    workload_ratio = workload[resource.id] / resource.total_available_hours
    score -= workload_ratio * 50.0
    
    return score
```

### 3. RLBasedOptimizer (Q-Learning)

**Purpose**: Learns optimal scheduling through reinforcement learning.

**Hyperparameters**:

| Parameter | Value | Description |
|-----------|-------|-------------|
| α (alpha) | 0.1 | Learning rate |
| ε (epsilon) | 0.3 | Exploration rate |
| γ (gamma) | 0.9 | Discount factor |
| Episodes | 100 | Training iterations |

**State Representation**:

```python
def _get_state_simple(self, process, completed_tasks, resource_next_available, resource_workload):
    state = []
    
    # Task features: [completed, ready_to_start] for each task
    for task in process.tasks:
        state.append(1.0 if task.id in completed_tasks else 0.0)
        state.append(1.0 if task.can_start(completed_tasks) else 0.0)
    
    # Resource features: [normalized_availability, normalized_workload]
    for resource in process.resources:
        next_hour = resource_next_available.get(resource.id, 0.0)
        state.append(min(1.0, next_hour / process.project_duration_hours))
        workload = resource_workload.get(resource.id, 0.0)
        state.append(min(1.0, workload / resource.total_available_hours))
    
    return np.array(state)
```

**Action Space**:

```python
Action = (task_id, resource_id)
# All valid task-resource pairs where:
# - Task dependencies are satisfied
# - Resource has required skills
# - Resource has available capacity
```

**Reward Function**:

```python
def _calculate_reward(self, task, resource, start_time, process):
    reward = 0.0
    
    # Duration reward (shorter is better)
    duration_reward = 1.0 / max(0.001, task.duration_hours)
    
    # Resource utilization (prefer less-loaded)
    utilization_reward = 1.0 - self._get_resource_load(resource.id)
    
    # Skill match bonus
    skill_match_reward = sum(
        res_skill.level.value / 5.0
        for req_skill in task.required_skills
        for res_skill in resource.skills
        if res_skill.name == req_skill.name
    )
    
    # Parallel execution bonus
    parallel_bonus = 0.3 if can_parallelize else 0.0
    
    # Critical path bonus
    critical_bonus = 0.4 if task.is_critical else 0.0
    
    # Cost penalty
    cost_penalty = (resource.hourly_rate * task.duration_hours) / 2000.0
    
    # Combine with weights
    reward = (
        time_weight * duration_reward +
        0.2 * utilization_reward +
        0.15 * skill_match_reward +
        0.25 * parallel_bonus +
        0.1 * critical_bonus
    ) - cost_weight * cost_penalty
    
    return max(0.000001, reward)
```

**Q-Learning Update**:

```
Q(s, a) ← Q(s, a) + α × [R + γ × max(Q(s', a')) - Q(s, a)]
```

### 4. GeneticOptimizer (Evolution)

**Purpose**: Evolutionary optimization for global optima search.

**Parameters**:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Population | 50 | Number of solutions |
| Generations | 100 | Evolution iterations |
| Crossover Rate | 80% | Probability of crossover |
| Mutation Rate | 10% | Probability of mutation |
| Elite Count | 5 | Best solutions preserved |

**Chromosome Encoding**:

```python
chromosome = [
    (task_0_resource_idx, task_0_position),
    (task_1_resource_idx, task_1_position),
    ...
]
# Each gene: (resource index, scheduling priority)
```

**Fitness Function**:

```python
def _calculate_fitness(self, chromosome, process):
    schedule = self._decode_chromosome(chromosome, process)
    
    # Multi-objective fitness
    time_score = 1000.0 / (schedule.total_duration_hours + 1)
    cost_score = 1000.0 / (schedule.total_cost + 1)
    utilization_score = average_resource_utilization * 100
    
    fitness = (
        time_weight * time_score +
        cost_weight * cost_score +
        0.2 * utilization_score
    )
    
    # Penalty for constraint violations
    if has_deadlocks:
        fitness *= 0.5
    if missed_deadlines:
        fitness *= 0.7
    
    return fitness
```

**Genetic Operators**:

```python
# Selection: Tournament selection (k=3)
def _tournament_selection(self, population, fitness_scores, k=3):
    candidates = random.sample(range(len(population)), k)
    winner = max(candidates, key=lambda i: fitness_scores[i])
    return population[winner]

# Crossover: Two-point crossover
def _crossover(self, parent1, parent2):
    if random.random() > self.crossover_rate:
        return parent1.copy(), parent2.copy()
    
    point1, point2 = sorted(random.sample(range(len(parent1)), 2))
    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    return child1, child2

# Mutation: Random resource reassignment
def _mutate(self, chromosome):
    for i in range(len(chromosome)):
        if random.random() < self.mutation_rate:
            chromosome[i] = (
                random.randint(0, num_resources - 1),
                random.random()
            )
    return chromosome
```

---

## NLP & AI Components

### File: `process_optimization_agent/Optimization/nlp_dependency_analyzer.py`

**Purpose**: Advanced semantic analysis using Small Language Models (SLMs).

### SLM Model

```python
# Model: all-MiniLM-L6-v2
# Type: Sentence Transformer
# Dimensions: 384
# Layers: 6 transformer layers
# Speed: ~14,000 sentences/sec

from sentence_transformers import SentenceTransformer
self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
```

### TaskAnalysis

```python
@dataclass
class TaskAnalysis:
    """Analysis results for a single task"""
    task_id: str
    task_name: str
    action_verb: Optional[str]         # Primary action (generate, review, etc.)
    objects: List[str]                 # Key objects (invoice, report, etc.)
    entities: List[str]                # Named entities
    has_temporal_keywords: bool        # Contains time-related words
    temporal_keywords: List[str]       # Specific time keywords found
    requires_input_from: List[str]     # Input requirements
    produces_output: List[str]         # Output indicators
```

### TaskRelationship

```python
@dataclass
class TaskRelationship:
    """Relationship between two tasks"""
    task1_id: str
    task2_id: str
    dependency_type: DependencyType    # SEQUENTIAL, PARALLEL, CONDITIONAL
    confidence: float                  # 0.0 - 1.0
    reasons: List[str]                 # Explanation of detection
    can_parallelize: bool              # Safe to run in parallel
```

### Semantic Similarity Calculation

```python
def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
    """Calculate semantic similarity using sentence embeddings"""
    embeddings = self.sentence_model.encode([text1, text2])
    
    # Cosine similarity
    similarity = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )
    return float(similarity)
```

### Sequential Keywords

```python
SEQUENTIAL_KEYWORDS = [
    'after', 'once', 'before', 'then', 'following', 'upon',
    'subsequent', 'next', 'when', 'upon completion', 'after receiving',
    'based on', 'using', 'with', 'from', 'requires', 'needs',
    'pending', 'awaiting', 'depends on', 'following receipt'
]
```

### Action Classification

```python
EARLY_ACTIONS = ['generate', 'create', 'initiate', 'start', 'calculate', 'prepare', 'collect']
MIDDLE_ACTIONS = ['review', 'verify', 'validate', 'process', 'analyze', 'assess', 'evaluate']
LATE_ACTIONS = ['send', 'deliver', 'distribute', 'notify', 'close', 'finalize', 'approve']
```

---

## CMS Integration

### File: `process_optimization_agent/Optimization/cms_client.py`

**Purpose**: Communicates with enterprise CMS systems using HttpOnly cookie authentication.

### Authentication

```python
class CMSClient:
    def __init__(self, base_url=None, bearer_token=None, use_cookies=True):
        self.base_url = base_url or os.getenv("REACT_APP_BASE_URL")
        self.session = requests.Session()
        
        if use_cookies:
            self._authenticate_with_cookies()
    
    def _authenticate_with_cookies(self):
        """Authenticate and receive HttpOnly cookie"""
        auth_url = f"{self.base_url}/auth/login"
        auth_data = {
            "email": os.getenv("CMS_AUTH_EMAIL"),
            "password": os.getenv("CMS_AUTH_PASSWORD")
        }
        response = self.session.post(auth_url, json=auth_data)
        # Cookie automatically stored in session
```

### API Methods

```python
def get_process_with_relations(self, process_id: int) -> Dict[str, Any]:
    """Fetch process with all relations"""
    url = f"{self.base_url}/process/{process_id}/with-relations"
    return self.session.get(url).json()

def get_all_processes(self) -> List[Dict[str, Any]]:
    """Fetch all processes"""
    return self.session.get(f"{self.base_url}/process").json()

def verify_session(self) -> Dict[str, Any]:
    """Verify session is valid"""
    return self.session.get(f"{self.base_url}/auth/me").json()
```

### File: `process_optimization_agent/Optimization/cms_transformer.py`

**Purpose**: Transforms CMS data format to agent-compatible format.

### Data Format Conversion

**CMS Format**:
```json
{
  "process_id": 1,
  "process_name": "Patient Registration",
  "company": {"name": "Hospital"},
  "process_task": [
    {
      "order": 1,
      "task": {
        "task_id": 1,
        "task_name": "Check-in",
        "task_capacity_minutes": 15,
        "jobTasks": [
          {"job": {"name": "Receptionist", "hourlyRate": 25}}
        ]
      }
    }
  ]
}
```

**Agent Format**:
```json
{
  "id": "1",
  "name": "Patient Registration",
  "company": "Hospital",
  "tasks": [
    {
      "id": "1",
      "name": "Check-in",
      "duration_hours": 0.25,
      "required_skills": [{"name": "Receptionist", "level": 3}],
      "order": 1
    }
  ],
  "resources": [
    {
      "id": "Receptionist",
      "name": "Receptionist",
      "skills": [{"name": "Receptionist", "level": 3}],
      "hourly_rate": 25
    }
  ]
}
```

### Validation

```python
class ProcessValidationError(Exception):
    """Custom exception for validation errors"""
    def __init__(self, message: str, error_code: str = "VALIDATION_ERROR"):
        self.message = message
        self.error_code = error_code
```

---

## Visualization

### File: `process_optimization_agent/visualization/visualizer.py`

**Purpose**: Creates visual representations of optimization results.

### Visualizer Class

```python
class Visualizer:
    """Healthcare-specific visualization for process optimization"""
    
    def __init__(self, style: str = "seaborn-v0_8"):
        plt.style.use(style)
        
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#2ecc71',
            'warning': '#e74c3c',
            'info': '#17A2B8'
        }
```

### Visualization Types

#### 1. Summary Page

```python
def create_summary_page(self, process, schedule, process_type=None, before_metrics=None):
    """
    Creates:
    - Patient/User journey timeline
    - Summary metrics table
    - Before/After comparison (if metrics provided)
    """
```

#### 2. Allocation Page

```python
def create_allocation_page(self, process, schedule, process_type=None):
    """
    Creates:
    - Resource to task assignment timeline (Gantt chart)
    - Time utilization per resource (bar chart)
    - Cost per resource (bar chart)
    - Parallel task execution visualization
    """
```

#### 3. Patient Journey Timeline

```python
def _plot_patient_journey_timeline(self, ax, process, schedule):
    """Line chart showing patient flow through tasks with waiting times"""
    # X-axis: Time (hours)
    # Y-axis: Tasks
    # Markers: Task start/end points
    # Bands: Task duration
    # Gaps: Waiting time (highlighted)
```

#### 4. Resource Timeline

```python
def _plot_resource_task_timeline(self, ax, process, schedule):
    """Gantt chart showing resource assignments"""
    # Y-axis: Resources
    # X-axis: Time (hours)
    # Bars: Tasks assigned to each resource
    # Color-coded by task type
```

### Auto-Detection

```python
def _detect_process_type(self, process):
    """Auto-detect process type for visualization routing"""
    healthcare_keywords = ['patient', 'medical', 'doctor', 'nurse', ...]
    
    for keyword in healthcare_keywords:
        if keyword in process.name.lower():
            return "Healthcare"
    
    return "Manufacturing"  # Default
```

---

## API Layer

### File: `API/main.py`

**Purpose**: FastAPI server providing REST endpoints for CMS integration.

### Server Configuration

```python
app = FastAPI(title="Process Optimization API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://fyp-cms-frontend.vercel.app",
        "http://localhost:3000",
        "https://crystalsystemcms-production.up.railway.app",
        # ... other origins
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
```

### API Endpoints

#### 1. Optimize Process by ID

```
GET /cms/optimize/{process_id}
```

**Response**:
```json
{
  "status": "success",
  "process_id": 1,
  "process_name": "Patient Registration",
  "schedule": {
    "entries": [...],
    "total_duration_hours": 5.5,
    "total_cost": 275.00
  },
  "metrics": {
    "time_saved_percentage": 25.5,
    "cost_saved_percentage": 15.0
  },
  "improvements": {
    "time_efficiency": {...},
    "cost_efficiency": {...}
  }
}
```

#### 2. What-If Analysis

```
GET /cms/whatif/{process_id}
```

**Response**:
```json
{
  "scenario": {
    "assignments": [
      {
        "task_id": "1",
        "resource_id": "R1",
        "start_hour": 0,
        "end_hour": 0.5
      }
    ],
    "constraints": {
      "time_weight": 0.5,
      "cost_weight": 0.5
    }
  },
  "metrics": {
    "total_duration": 5.5,
    "total_cost": 275.00,
    "efficiency_ratio": 0.85
  }
}
```

### Helper Functions

```python
def detect_data_format(payload: Dict) -> str:
    """Detect CMS vs Agent format"""
    if 'tasks' in payload and 'resources' in payload:
        return 'agent'
    elif 'process_task' in payload:
        return 'cms'
    return 'agent'

def _build_improvements_section(time_saved, current_cost, optimized_cost, ...):
    """Build comprehensive improvements breakdown"""
    # Returns detailed cost/time savings analysis
```

---

## Domain-Specific Scenarios

### Healthcare (`scenarios/healthcare/`)

**Optimizer**: `HealthcareOptimizer`
- Focuses on patient journey optimization
- Minimizes waiting time
- Ensures continuity of care

**Metrics**:
- Patient wait time
- Resource utilization
- Care continuity score

### Insurance (`scenarios/insurance/`)

**Optimizer**: `InsuranceProcessOptimizer`
- Handles claims processing workflows
- Optimizes approval chains
- Manages compliance requirements

**Metrics**:
- Claim processing time
- Approval rate
- Compliance score

### Manufacturing (`scenarios/manufacturing/`)

**Optimizer**: `ManufacturingOptimizer`
- Maximizes parallel production
- Optimizes production line efficiency
- Minimizes idle time

**Metrics**:
- Throughput rate
- Equipment utilization
- Production cycle time

### Banking (`scenarios/banking/`)

**Optimizer**: `BankingProcessOptimizer`
- Handles loan approval workflows
- Manages risk assessment chains
- Optimizes compliance checks

**Metrics**:
- Approval time
- Risk score
- Compliance rate

---

## Scripts & Tests

### `scripts/run_optimization.py`

**Purpose**: CLI wrapper for running optimization.

```bash
python run_optimization.py examples/patient_registration.json
```

### `scripts/run_rl_optimizer.py`

**Purpose**: Main RL optimization script with CLI controls.

```bash
python run_rl_optimizer.py process.json \
  --max-parallel 4 \
  --parallel-policy balanced \
  --dep-detect balanced \
  --dep-threshold 0.75 \
  --review-deps
```

**Arguments**:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `process_json` | str | Required | Path to process JSON |
| `--max-parallel` | int | 4 | Maximum parallel tasks |
| `--parallel-policy` | str | balanced | `strict` or `balanced` |
| `--dep-detect` | str | balanced | `off`, `strict`, `balanced`, `aggressive` |
| `--dep-threshold` | float | 0.75 | Semantic similarity threshold |
| `--review-deps` | flag | False | Interactive dependency review |

### `tests/test_process_detection.py`

**Purpose**: Tests process type detection and optimization.

```bash
python tests/test_process_detection.py examples/outpatient_consultation.json
```

### `tests/test_api_endpoints.py`

**Purpose**: API endpoint testing.

---

## Configuration & Dependencies

### `requirements.txt`

```
# Core
fastapi>=0.100.0
uvicorn>=0.22.0
python-dotenv>=1.0.0
requests>=2.31.0

# NLP & AI
spacy>=3.5.0
sentence-transformers>=2.2.0
scikit-learn>=1.2.0
numpy>=1.24.0
tf-keras>=2.15.0

# Data Processing
pandas>=2.0.0
networkx>=3.1

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
pydantic>=2.0.0
```

### Environment Variables

```bash
# CMS Configuration
REACT_APP_BASE_URL=https://server-digitaltwin-enterprise-production.up.railway.app
CMS_AUTH_EMAIL=your-email@example.com
CMS_AUTH_PASSWORD=your-password

# Optional
DEBUG=false
LOG_LEVEL=INFO
```

### Railway Deployment (`Procfile`)

```
web: uvicorn API.main:app --host 0.0.0.0 --port $PORT
```

---

## Workflow

### Complete Optimization Workflow

```
1. INITIALIZATION
   └── API receives request with process_id
   
2. DATA RETRIEVAL
   ├── CMSClient authenticates with CMS
   ├── Fetches process with relations
   └── CMSDataTransformer converts format
   
3. PROCESS ANALYSIS
   ├── ProcessIntelligence detects type
   ├── TaskClassifier classifies involvement
   └── DependencyDetector finds dependencies
   
4. DEPENDENCY ANALYSIS
   ├── Rule-based detection
   ├── spaCy NLP parsing
   ├── TF-IDF similarity
   ├── SLM semantic analysis
   └── Domain-specific rules
   
5. OPTIMIZATION
   ├── IntelligentOptimizer selects strategy
   ├── Domain-specific optimizer runs
   │   ├── ProcessOptimizer (Greedy)
   │   ├── RLBasedOptimizer (Q-Learning)
   │   └── GeneticOptimizer (Evolution)
   └── Multi-job resolution
   
6. POST-PROCESSING
   ├── DeadlockDetector checks cycles
   ├── WhatIfAnalyzer runs scenarios
   └── Metrics calculation
   
7. VISUALIZATION
   ├── Summary page generation
   ├── Allocation page generation
   └── JSON/PNG output
   
8. RESPONSE
   └── API returns optimized schedule with metrics
```

### Multi-Job Resolution Flow

```
1. Identify tasks with multiple job assignments
2. For each multi-job task:
   a. Extract required skills from task description
   b. Match each job against required skills
   c. Calculate match percentage for each job
   d. If best job ≥ 90% match → Keep only best job
   e. Else → Split task into sub-tasks
   f. Each sub-task gets exactly one job
3. Calculate cost savings from resolution
4. Return modified process with 1:1 relationships
```

---

## Summary

The **Process Optimization Agent** is a comprehensive AI-powered system that:

1. **Analyzes** business processes using NLP and machine learning
2. **Detects** task dependencies through multiple methods
3. **Optimizes** schedules using Greedy, Q-Learning, and Genetic algorithms
4. **Integrates** seamlessly with enterprise CMS systems
5. **Visualizes** results for easy understanding
6. **Adapts** to different domains (healthcare, manufacturing, etc.)

The modular architecture allows for:
- Easy extension with new optimizers
- Addition of domain-specific scenarios
- Integration with different CMS systems
- Customization of analysis methods

---

*Document Version: 1.0*  
*Last Updated: January 2026*
