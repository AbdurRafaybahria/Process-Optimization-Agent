"""
Advanced NLP-based Task Dependency Analyzer
Analyzes task descriptions using NLP to detect implicit dependencies
and determine safe parallelization opportunities.
"""

import re
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
from enum import Enum

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class DependencyType(Enum):
    """Types of dependencies between tasks"""
    SEQUENTIAL = "sequential"  # Must run in order
    PARALLEL = "parallel"      # Can run simultaneously
    CONDITIONAL = "conditional"  # Depends on conditions
    UNCERTAIN = "uncertain"    # Needs human review


class ConfidenceLevel(Enum):
    """Confidence levels for parallelization decisions"""
    HIGH = "high"       # >90% confident
    MEDIUM = "medium"   # 50-90% confident
    LOW = "low"         # <50% confident


@dataclass
class TaskRelationship:
    """Represents the relationship between two tasks"""
    task1_id: str
    task2_id: str
    dependency_type: DependencyType
    confidence: float
    reasons: List[str]
    can_parallelize: bool


@dataclass
class TaskAnalysis:
    """Analysis results for a single task"""
    task_id: str
    task_name: str
    action_verb: Optional[str]
    objects: List[str]
    entities: List[str]
    has_temporal_keywords: bool
    temporal_keywords: List[str]
    requires_input_from: List[str]
    produces_output: List[str]


class NLPDependencyAnalyzer:
    """
    Advanced NLP-based dependency analyzer for process tasks.
    Uses multiple techniques to detect implicit dependencies.
    """
    
    # Sequential indicator keywords
    SEQUENTIAL_KEYWORDS = [
        'after', 'once', 'before', 'then', 'following', 'upon', 
        'subsequent', 'next', 'when', 'upon completion', 'after receiving',
        'based on', 'using', 'with', 'from', 'requires', 'needs',
        'pending', 'awaiting', 'depends on', 'following receipt'
    ]
    
    # Action verbs that typically come early in a process
    EARLY_ACTIONS = [
        'generate', 'create', 'initiate', 'start', 'calculate', 
        'prepare', 'collect', 'gather', 'receive', 'submit'
    ]
    
    # Action verbs that typically come late in a process
    LATE_ACTIONS = [
        'send', 'deliver', 'distribute', 'notify', 'close', 
        'finalize', 'complete', 'approve', 'authorize', 'archive'
    ]
    
    # Action verbs that typically come in middle
    MIDDLE_ACTIONS = [
        'review', 'verify', 'validate', 'process', 'analyze',
        'assess', 'evaluate', 'check', 'inspect', 'examine'
    ]
    
    # Domain-specific rules for insurance
    INSURANCE_RULES = {
        ('calculate', 'generate'): 'sequential',  # Calculate before generating invoice
        ('generate', 'send'): 'sequential',       # Generate before sending
        ('generate', 'verify'): 'sequential',     # Generate before verifying
        ('verify', 'approve'): 'sequential',      # Verify before approving
        ('approve', 'send'): 'sequential',        # Approve before sending
        ('receive', 'process'): 'sequential',     # Receive before processing
        ('process', 'update'): 'parallel',        # Can update while processing
        ('generate', 'update'): 'parallel',       # Can update database while generating
    }
    
    def __init__(self, use_advanced_nlp: bool = True):
        """
        Initialize the NLP dependency analyzer.
        
        Args:
            use_advanced_nlp: Whether to use advanced NLP models (requires additional packages)
        """
        self.use_advanced_nlp = use_advanced_nlp
        self.nlp_model = None
        self.sentence_model = None
        
        # Initialize spaCy if available
        if use_advanced_nlp and SPACY_AVAILABLE:
            try:
                self.nlp_model = spacy.load("en_core_web_sm")
            except OSError:
                print("spaCy model not found. Run: python -m spacy download en_core_web_sm")
                self.nlp_model = None
        
        # Initialize sentence transformer if available
        if use_advanced_nlp and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"Failed to load sentence transformer: {e}")
                self.sentence_model = None
    
    def analyze_task(self, task_id: str, task_name: str, task_description: str) -> TaskAnalysis:
        """
        Analyze a single task to extract key information.
        
        Args:
            task_id: Unique task identifier
            task_name: Task name
            task_description: Task description text
            
        Returns:
            TaskAnalysis object with extracted information
        """
        text = f"{task_name}. {task_description}".lower()
        
        # Extract action verb (first verb in the text)
        action_verb = self._extract_action_verb(text)
        
        # Extract objects and entities
        objects = self._extract_objects(text)
        entities = self._extract_entities(text)
        
        # Check for temporal keywords
        temporal_keywords = [kw for kw in self.SEQUENTIAL_KEYWORDS if kw in text]
        has_temporal = len(temporal_keywords) > 0
        
        # Identify inputs and outputs
        requires_input = self._extract_input_requirements(text)
        produces_output = self._extract_output_indicators(text)
        
        return TaskAnalysis(
            task_id=task_id,
            task_name=task_name,
            action_verb=action_verb,
            objects=objects,
            entities=entities,
            has_temporal_keywords=has_temporal,
            temporal_keywords=temporal_keywords,
            requires_input_from=requires_input,
            produces_output=produces_output
        )
    
    def _extract_action_verb(self, text: str) -> Optional[str]:
        """Extract the primary action verb from text"""
        if self.nlp_model:
            doc = self.nlp_model(text)
            for token in doc:
                if token.pos_ == "VERB":
                    return token.lemma_
        else:
            # Fallback: simple pattern matching
            for verb in self.EARLY_ACTIONS + self.MIDDLE_ACTIONS + self.LATE_ACTIONS:
                if verb in text:
                    return verb
        return None
    
    def _extract_objects(self, text: str) -> List[str]:
        """Extract key objects from text"""
        objects = []
        common_objects = [
            'invoice', 'report', 'payment', 'claim', 'policy', 'premium',
            'bill', 'statement', 'document', 'form', 'application',
            'account', 'customer', 'client', 'data', 'record', 'database'
        ]
        
        for obj in common_objects:
            if obj in text:
                objects.append(obj)
        
        return objects
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text"""
        if self.nlp_model:
            doc = self.nlp_model(text)
            return [ent.text for ent in doc.ents]
        return []
    
    def _extract_input_requirements(self, text: str) -> List[str]:
        """Identify what inputs this task requires"""
        requirements = []
        
        # Look for phrases indicating input needs
        input_patterns = [
            r'based on (\w+)',
            r'using (\w+)',
            r'from (\w+)',
            r'requires (\w+)',
            r'needs (\w+)',
            r'with (\w+)',
        ]
        
        for pattern in input_patterns:
            matches = re.findall(pattern, text)
            requirements.extend(matches)
        
        return requirements
    
    def _extract_output_indicators(self, text: str) -> List[str]:
        """Identify what outputs this task produces"""
        outputs = []
        
        # Look for phrases indicating outputs
        output_patterns = [
            r'generate (\w+)',
            r'create (\w+)',
            r'produce (\w+)',
            r'prepare (\w+)',
        ]
        
        for pattern in output_patterns:
            matches = re.findall(pattern, text)
            outputs.extend(matches)
        
        return outputs
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        
        Returns:
            Similarity score between 0 and 1
        """
        if self.sentence_model:
            try:
                embeddings = self.sentence_model.encode([text1, text2])
                similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
                return float(similarity)
            except Exception:
                pass
        
        # Fallback: simple word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        return overlap / total if total > 0 else 0.0
    
    def analyze_task_relationship(
        self, 
        analysis1: TaskAnalysis, 
        analysis2: TaskAnalysis,
        description1: str = "",
        description2: str = ""
    ) -> TaskRelationship:
        """
        Analyze the relationship between two tasks.
        
        Args:
            analysis1: Analysis of first task
            analysis2: Analysis of second task
            description1: Full description of first task
            description2: Full description of second task
            
        Returns:
            TaskRelationship object describing how tasks relate
        """
        reasons = []
        confidence = 0.5  # Start with medium confidence
        can_parallelize = True
        dependency_type = DependencyType.PARALLEL
        
        # 1. Check for temporal keywords
        if analysis1.has_temporal_keywords or analysis2.has_temporal_keywords:
            reasons.append("Contains temporal keywords suggesting sequence")
            confidence -= 0.2
            can_parallelize = False
            dependency_type = DependencyType.SEQUENTIAL
        
        # 2. Check for shared objects
        shared_objects = set(analysis1.objects) & set(analysis2.objects)
        if shared_objects:
            reasons.append(f"Shared objects: {', '.join(shared_objects)}")
            confidence -= 0.15
            # Not necessarily sequential, but increases risk
        
        # 3. Check action verb relationships
        if analysis1.action_verb and analysis2.action_verb:
            verb_pair = (analysis1.action_verb, analysis2.action_verb)
            
            # Check domain-specific rules
            if verb_pair in self.INSURANCE_RULES:
                rule_result = self.INSURANCE_RULES[verb_pair]
                if rule_result == 'sequential':
                    reasons.append(f"Domain rule: {verb_pair[0]} must precede {verb_pair[1]}")
                    confidence = 0.9  # High confidence in domain rules
                    can_parallelize = False
                    dependency_type = DependencyType.SEQUENTIAL
                else:
                    reasons.append(f"Domain rule: {verb_pair[0]} and {verb_pair[1]} can be parallel")
                    confidence = 0.85
            
            # Check action ordering
            if analysis1.action_verb in self.EARLY_ACTIONS and analysis2.action_verb in self.LATE_ACTIONS:
                reasons.append(f"Typical sequence: early action '{analysis1.action_verb}' before late action '{analysis2.action_verb}'")
                confidence = 0.75
                can_parallelize = False
                dependency_type = DependencyType.SEQUENTIAL
            
            if analysis1.action_verb in self.LATE_ACTIONS and analysis2.action_verb in self.EARLY_ACTIONS:
                reasons.append(f"Reverse sequence detected: '{analysis2.action_verb}' should come before '{analysis1.action_verb}'")
                confidence = 0.75
                can_parallelize = False
                dependency_type = DependencyType.SEQUENTIAL
        
        # 4. Check data flow (output -> input)
        for output in analysis1.produces_output:
            if output in analysis2.requires_input_from:
                reasons.append(f"Data flow: Task 1 produces '{output}' needed by Task 2")
                confidence = 0.85
                can_parallelize = False
                dependency_type = DependencyType.SEQUENTIAL
        
        # 5. Calculate semantic similarity
        if description1 and description2:
            similarity = self.calculate_semantic_similarity(description1, description2)
            if similarity > 0.8:
                reasons.append(f"High semantic similarity ({similarity:.2f}) suggests related tasks")
                confidence -= 0.1
                # High similarity suggests they might be sequential
            elif similarity < 0.3:
                reasons.append(f"Low semantic similarity ({similarity:.2f}) suggests independent tasks")
                confidence += 0.1
                # Low similarity suggests they can be parallel
        
        # 6. Check for explicit references
        if analysis2.task_name.lower() in (description1 or "").lower():
            reasons.append("Task 2 explicitly mentioned in Task 1 description")
            confidence = 0.8
            can_parallelize = False
            dependency_type = DependencyType.SEQUENTIAL
        
        # 7. Adjust confidence based on evidence
        if not reasons:
            reasons.append("No dependencies detected - tasks appear independent")
            confidence = 0.85
            can_parallelize = True
        
        # Ensure confidence is between 0 and 1
        confidence = max(0.0, min(1.0, confidence))
        
        return TaskRelationship(
            task1_id=analysis1.task_id,
            task2_id=analysis2.task_id,
            dependency_type=dependency_type,
            confidence=confidence,
            reasons=reasons,
            can_parallelize=can_parallelize
        )
    
    def analyze_all_tasks(
        self, 
        tasks: List[Dict[str, str]]
    ) -> Tuple[List[TaskAnalysis], List[TaskRelationship]]:
        """
        Analyze all tasks and their relationships.
        
        Args:
            tasks: List of task dictionaries with 'id', 'name', and 'description'
            
        Returns:
            Tuple of (task analyses, task relationships)
        """
        # Analyze each task individually
        analyses = []
        for task in tasks:
            analysis = self.analyze_task(
                task_id=task.get('id', ''),
                task_name=task.get('name', ''),
                task_description=task.get('description', '')
            )
            analyses.append(analysis)
        
        # Analyze relationships between all task pairs
        relationships = []
        for i, analysis1 in enumerate(analyses):
            for j, analysis2 in enumerate(analyses):
                if i < j:  # Only analyze each pair once
                    relationship = self.analyze_task_relationship(
                        analysis1, 
                        analysis2,
                        tasks[i].get('description', ''),
                        tasks[j].get('description', '')
                    )
                    relationships.append(relationship)
        
        return analyses, relationships
    
    def get_parallelization_groups(
        self, 
        tasks: List[Dict[str, str]],
        min_confidence: float = 0.7
    ) -> List[List[str]]:
        """
        Determine which tasks can be safely parallelized.
        
        Args:
            tasks: List of task dictionaries
            min_confidence: Minimum confidence threshold for parallelization
            
        Returns:
            List of task groups where tasks within each group can run in parallel
        """
        analyses, relationships = self.analyze_all_tasks(tasks)
        
        # Build a graph of tasks and dependencies
        task_ids = [a.task_id for a in analyses]
        must_be_sequential = set()
        
        for rel in relationships:
            if not rel.can_parallelize and rel.confidence >= min_confidence:
                must_be_sequential.add((rel.task1_id, rel.task2_id))
        
        # Group tasks that can run in parallel
        groups = []
        remaining_tasks = set(task_ids)
        
        while remaining_tasks:
            # Start a new parallel group
            current_group = []
            for task_id in list(remaining_tasks):
                # Check if this task can be added to the current group
                can_add = True
                for existing_task in current_group:
                    if (task_id, existing_task) in must_be_sequential or \
                       (existing_task, task_id) in must_be_sequential:
                        can_add = False
                        break
                
                if can_add:
                    current_group.append(task_id)
                    remaining_tasks.remove(task_id)
            
            if current_group:
                groups.append(current_group)
            else:
                # Shouldn't happen, but prevent infinite loop
                break
        
        return groups
