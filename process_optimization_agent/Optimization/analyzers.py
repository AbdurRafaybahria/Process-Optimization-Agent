"""
Analysis components for the Process Optimization Agent
"""

from typing import Dict, List, Set, Any, Optional, Union
import networkx as nx
from .models import Process, Task, Resource, Skill, SkillLevel, Schedule
from typing import List, Dict, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import os
import warnings
import copy
import re
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Make spacy optional
NLP_AVAILABLE = False
try:
    import spacy
    NLP_AVAILABLE = True
except ImportError:
    warnings.warn(
        "spaCy not found. Install with 'pip install spacy' and download a language model "
        "with 'python -m spacy download en_core_web_sm' for better dependency detection."
    )

from .models import Task, Resource, Process, Schedule, ScheduleEntry
from typing import Tuple

# Import the advanced NLP dependency analyzer
try:
    from .nlp_dependency_analyzer import NLPDependencyAnalyzer, TaskRelationship, DependencyType
    NLP_ANALYZER_AVAILABLE = True
except ImportError:
    NLP_ANALYZER_AVAILABLE = False
    print("Warning: NLP Dependency Analyzer not available. Using basic detection only.")


class DependencyDetector:
    """Detects dependencies between tasks using NLP and rule-based methods"""
    
    def __init__(self, use_nlp: bool = True, similarity_threshold: float = 0.7, process_type: str = "unknown"):
        """Initialize the dependency detector
        
        Args:
            use_nlp: Whether to use NLP for dependency detection
            similarity_threshold: Threshold for considering task descriptions similar (0-1)
            process_type: Type of process (healthcare, manufacturing, etc.) for context-aware detection
        """
        self.use_nlp = use_nlp
        self.similarity_threshold = similarity_threshold
        self.process_type = process_type
        self.nlp = None
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # Initialize advanced NLP analyzer if available
        self.nlp_analyzer = None
        if use_nlp and NLP_ANALYZER_AVAILABLE:
            try:
                self.nlp_analyzer = NLPDependencyAnalyzer(use_advanced_nlp=True)
                print("Advanced NLP Dependency Analyzer initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize NLP Analyzer: {e}")
        
        if use_nlp:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                print("Warning: spaCy model 'en_core_web_sm' not found. Using rule-based detection only.")
                self.use_nlp = False
        
        # Initialize keywords and patterns
        self._init_keywords()
        
        # Cache for similarity scores
        self._similarity_cache = {}
        
    def _init_keywords(self):
        """Initialize keywords for rule-based dependency detection"""
        self.dependency_keywords = {
            'after': ['after', 'following', 'once', 'when', 'until', 'subsequent to'],
            'before': ['before', 'prior to', 'in preparation for', 'to prepare for', 'preceding'],
            'requires': ['requires', 'needs', 'depends on', 'relies on', 'using', 'with', 'must have'],
            'parallel': ['meanwhile', 'concurrently', 'in parallel', 'at the same time', 'simultaneously']
        }
        
        # Common task type patterns with parallel execution potential
        self.task_patterns = {
            'design': {'pattern': r'\b(design|architect|plan|blueprint|mockup|wireframe)\b', 'parallel': True},
            'development': {'pattern': r'\b(develop|code|implement|build|create|program)\b', 'parallel': True},
            'testing': {'pattern': r'\b(test|qa|quality|verify|validate|check)\b', 'parallel': False},
            'deployment': {'pattern': r'\b(deploy|release|launch|publish|go-live|rollout)\b', 'parallel': False},
            'review': {'pattern': r'\b(review|approve|sign-off|audit|inspect|evaluate)\b', 'parallel': True},
            'documentation': {'pattern': r'\b(document|write|manual|guide|spec|tutorial)\b', 'parallel': True},
            'data_processing': {'pattern': r'\b(process|analyze|extract|transform|load|etl|clean|prepare)\b', 'parallel': True},
            'research': {'pattern': r'\b(research|investigate|analyze|evaluate|study|explore)\b', 'parallel': True}
        }
    
    def detect_dependencies(self, tasks: Union[Task, List[Task]], other_tasks: Optional[List[Task]] = None,
                          resources: Optional[List[Resource]] = None) -> Dict[str, Set[str]]:
        """Detect dependencies between tasks using NLP, rule-based, and resource-based approaches
        
        Args:
            tasks: Single task or list of tasks to analyze
            other_tasks: Optional list of other tasks to check against (for batch processing)
            resources: Optional list of resources to consider for resource-based dependencies
            
        Returns:
            Dict mapping task IDs to sets of dependent task IDs
        """
        if isinstance(tasks, Task):
            tasks = [tasks]
            
        if other_tasks is None:
            other_tasks = []
            
        # Initialize dependencies dictionary
        dependencies = defaultdict(set)
        
        # Use advanced NLP analyzer if available
        if self.nlp_analyzer is not None and tasks:
            all_tasks = tasks if not other_tasks else tasks + other_tasks
            nlp_dependencies = self._detect_nlp_dependencies(all_tasks)
            
            # Merge NLP-detected dependencies
            for task_id, deps in nlp_dependencies.items():
                dependencies[task_id].update(deps)
        
        # Process each task with traditional methods
        for task in tasks:
            # Detect dependencies for this task
            task_deps = self._detect_single_task_dependencies(task, other_tasks)
            for dep_id, dep_type in task_deps:
                if dep_id != task.id:  # Prevent self-dependencies
                    dependencies[task.id].add(dep_id)
        
        # Validate and clean up dependencies
        self.validate_dependencies(tasks, dependencies)
        
        # Remove circular dependencies
        dependencies = self._remove_circular_dependencies(dependencies)
        
        # If resources are provided, add resource-based dependencies
        if resources is not None and tasks and other_tasks:
            resource_deps = self._detect_resource_dependencies(tasks + other_tasks, resources)
            for task_id, deps in resource_deps.items():
                dependencies[task_id].update(deps)
        
        return dict(dependencies)
    
    def _detect_single_task_dependencies(self, task: Task, other_tasks: List[Task]) -> List[Tuple[str, str]]:
        """
        Detect dependencies for a single task against other tasks
        Returns list of (task_id, dependency_type) tuples
        """
        dependencies = []
        
        # Check for explicit dependencies
        if hasattr(task, 'depends_on') and task.depends_on:
            for dep_id in task.depends_on:
                if any(t.id == dep_id for t in other_tasks):
                    dependencies.append((dep_id, 'explicit'))
        
        # Use NLP or rule-based to find implicit dependencies in description
        if hasattr(task, 'description') and task.description:
            text = task.description.lower()
            
            if self.nlp is not None:
                # Use spaCy if available
                try:
                    doc = self.nlp(text)
                    for sent in doc.sents:
                        sent_text = sent.text.lower()  # Cache the sentence text
                        for token in sent:
                            try:
                                # Check for dependency indicators
                                if token.text.lower() in ['after', 'before', 'until']:
                                    # Find the object of the preposition
                                    for child in token.children:
                                        if hasattr(child, 'dep_') and child.dep_ in ['pobj', 'pcomp']:
                                            # Get the full subtree text once
                                            subtree_text = getattr(child, 'subtree', '').text.lower()
                                            if not subtree_text:
                                                continue
                                                
                                            # Look for task names in the subtree
                                            for other_task in other_tasks:
                                                if other_task.id != task.id and other_task.name.lower() in subtree_text:
                                                    dep_type = 'after' if token.text.lower() in ['after', 'until'] else 'before'
                                                    dependencies.append((other_task.id, dep_type))
                                
                                # Check for requirement indicators
                                elif hasattr(token, 'lemma_') and token.lemma_ in ['require', 'need', 'depend', 'rely']:
                                    # Look for task names in the sentence
                                    for other_task in other_tasks:
                                        if other_task.id != task.id and other_task.name.lower() in sent_text:
                                            dependencies.append((other_task.id, 'requires'))
                            except Exception as token_error:
                                # Skip this token if there's an error
                                continue
                except Exception as e:
                    warnings.warn(f"Error in NLP processing: {e}. Falling back to rule-based detection.", stacklevel=2)
            
            # Fallback to rule-based detection
            for other_task in other_tasks:
                if other_task.id == task.id:
                    continue
                    
                # Check for task names in description
                if other_task.name.lower() in text:
                    # Check for dependency keywords
                    for dep_type, keywords in self.dependency_keywords.items():
                        for keyword in keywords:
                            if f"{keyword} {other_task.name.lower()}" in text:
                                dependencies.append((other_task.id, dep_type))
                                break
        
        return dependencies
    
    def _detect_nlp_dependencies(self, tasks: List[Task]) -> Dict[str, Set[str]]:
        """
        Use advanced NLP analyzer to detect dependencies between tasks.
        
        Args:
            tasks: List of tasks to analyze
            
        Returns:
            Dict mapping task IDs to sets of dependent task IDs
        """
        if not self.nlp_analyzer or not tasks:
            return {}
        
        dependencies = defaultdict(set)
        
        try:
            # Prepare task data for NLP analyzer
            task_data = []
            for task in tasks:
                task_data.append({
                    'id': task.id,
                    'name': task.name,
                    'description': getattr(task, 'description', '')
                })
            
            # Analyze all tasks and their relationships
            analyses, relationships = self.nlp_analyzer.analyze_all_tasks(task_data)
            
            # Convert relationships to dependencies
            # High confidence sequential relationships become dependencies
            for rel in relationships:
                if not rel.can_parallelize and rel.confidence >= 0.7:
                    # Task 2 depends on Task 1 (Task 1 must complete before Task 2)
                    dependencies[rel.task2_id].add(rel.task1_id)
                    
                    # Log the reasoning for debugging
                    if rel.confidence >= 0.8:
                        print(f"  [NLP] High confidence dependency: {rel.task1_id} → {rel.task2_id}")
                        print(f"        Confidence: {rel.confidence:.2f}, Reasons: {', '.join(rel.reasons[:2])}")
            
        except Exception as e:
            print(f"Warning: Error in advanced NLP dependency detection: {e}")
            import traceback
            traceback.print_exc()
        
        return dict(dependencies)
    
    def _detect_similarity_based_parallelism(self, tasks: List[Task]) -> Dict[str, Set[str]]:
        """Detect tasks that can run in parallel based on description similarity
        
        Args:
            tasks: List of tasks to analyze
            
        Returns:
            Dict mapping task IDs to sets of task IDs that can run in parallel
        """
        parallelism = defaultdict(set)
        
        # Skip if we don't have enough tasks
        if len(tasks) < 2:
            return {}
            
        # Prepare text data for similarity analysis
        task_texts = []
        task_ids = []
        
        for task in tasks:
            if hasattr(task, 'description') and task.description:
                task_texts.append(f"{task.name}. {task.description}")
                task_ids.append(task.id)
        
        if len(task_texts) < 2:
            return {}
            
        try:
            # Calculate TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform(task_texts)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
            
            # Find similar tasks that can run in parallel
            for i, task_id in enumerate(task_ids):
                for j, other_id in enumerate(task_ids):
                    if i != j and similarity_matrix[i][j] > self.similarity_threshold:
                        # Only add if tasks don't have direct dependencies
                        task = next((t for t in tasks if t.id == task_id), None)
                        other_task = next((t for t in tasks if t.id == other_id), None)
                        
                        if task and other_task:
                            # Check if tasks don't depend on each other
                            if (task_id not in other_task.dependencies and 
                                other_id not in task.dependencies):
                                parallelism[task_id].add(other_id)
                                parallelism[other_id].add(task_id)
                                
        except Exception as e:
            print(f"Warning: Error in similarity analysis: {str(e)}")
            
        return dict(parallelism)
    
    def _detect_resource_dependencies(self, tasks: List[Task], resources: List[Resource]) -> Dict[str, Set[str]]:
        """Detect dependencies based on resource sharing and conflicts
        
        Args:
            tasks: List of tasks to analyze
            resources: List of available resources
            
        Returns:
            Dict mapping task IDs to sets of task IDs that share resources
        """
        resource_to_tasks = defaultdict(set)
        
        # Map resources to tasks that require them
        for task in tasks:
            # Find resources that can perform this task
            for resource in resources:
                if resource.has_all_skills(task.required_skills):
                    resource_to_tasks[resource.id].add(task.id)
        
        # Find tasks that share the same resources
        task_to_shared = defaultdict(set)
        for resource_id, task_set in resource_to_tasks.items():
            if len(task_set) > 1:  # Only if multiple tasks share this resource
                for task_id in task_set:
                    task_to_shared[task_id].update(t for t in task_set if t != task_id)
        
        return dict(task_to_shared)
    
    def _detect_pattern_dependencies(self, tasks: List[Task]) -> Dict[str, Set[str]]:
        """Detect dependencies based on common workflow patterns and parallel execution potential"""
        dependencies = defaultdict(set)
        
        # Categorize tasks by type and identify parallel potential
        task_info = {}
        for task in tasks:
            text = f"{task.name} {task.description}".lower()
            task_info[task.id] = {
                'type': None,
                'can_parallelize': False,
                'text': text,
                'keywords': set(),
                'entities': set()
            }
            
            # Extract keywords and entities for better matching
            if self.use_nlp and self.nlp is not None:
                doc = self.nlp(text)
                # Get noun chunks as potential keywords
                task_info[task.id]['keywords'].update(
                    chunk.text.lower() for chunk in doc.noun_chunks
                    if len(chunk.text.split()) <= 3  # Limit to 1-3 word phrases
                )
                # Get named entities
                task_info[task.id]['entities'].update(
                    ent.text.lower() for ent in doc.ents
                )
            
            # Match task patterns
            for task_type, pattern_info in self.task_patterns.items():
                if re.search(pattern_info['pattern'], text, re.IGNORECASE):
                    task_info[task.id]['type'] = task_type
                    task_info[task.id]['can_parallelize'] = pattern_info['parallel']
                    break
        
        # Enhanced workflow rules with parallel execution support
        workflow_rules = [
            # Sequential dependencies (must happen in order)
            {
                'name': 'design_before_development',
                'prereq_types': ['design'],
                'dependent_types': ['development'],
                'is_sequential': True
            },
            {
                'name': 'development_before_testing',
                'prereq_types': ['development'],
                'dependent_types': ['testing', 'review'],
                'is_sequential': True
            },
            {
                'name': 'testing_before_deployment',
                'prereq_types': ['testing', 'review'],
                'dependent_types': ['deployment'],
                'is_sequential': True
            },
            # Parallel execution opportunities
            {
                'name': 'parallel_design_tasks',
                'prereq_types': ['design'],
                'dependent_types': ['design'],
                'is_sequential': False,
                'max_parallel': 3  # Maximum number of parallel design tasks
            },
            {
                'name': 'parallel_development',
                'prereq_types': ['development'],
                'dependent_types': ['development'],
                'is_sequential': False,
                'max_parallel': 4  # Maximum number of parallel development tasks
            }
        ]
        
        # Apply workflow rules
        for rule in workflow_rules:
            prereq_tasks = [
                tid for tid, info in task_info.items() 
                if info['type'] in rule['prereq_types']
            ]
            
            dependent_tasks = [
                tid for tid, info in task_info.items() 
                if info['type'] in rule['dependent_types']
            ]
            
            if rule.get('is_sequential', True):
                # Sequential dependencies
                for prereq in prereq_tasks:
                    for dependent in dependent_tasks:
                        if prereq != dependent:  # No self-dependencies
                            dependencies[dependent].add(prereq)
            else:
                # Parallel execution - limit dependencies to enable parallelization
                max_parallel = rule.get('max_parallel', len(prereq_tasks))
                for i in range(0, len(prereq_tasks), max_parallel):
                    parallel_group = prereq_tasks[i:i + max_parallel]
                    for task in parallel_group[1:]:
                        # Make tasks in the same group dependent on the first task in the group
                        # This creates a chain that enables parallel execution of the group
                        dependencies[task].add(parallel_group[0])
        
        # Add explicit parallel execution hints from task descriptions
        for task in tasks:
            if not hasattr(task, 'description') or not task.description:
                continue
                
            text = task.description.lower()
            for keyword in self.dependency_keywords['parallel']:
                if keyword in text:
                    # Mark this task as parallelizable
                    if task.id in task_info:
                        task_info[task.id]['can_parallelize'] = True
        
        return dict(dependencies)
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity measure"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def validate_dependencies(self, tasks: List[Task], dependencies: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """Validate and clean up detected dependencies"""
        # Create task ID set for validation
        task_ids = {task.id for task in tasks}
        
        # Remove invalid dependencies
        cleaned_deps = {}
        for task_id, deps in dependencies.items():
            if task_id in task_ids:
                valid_deps = {dep for dep in deps if dep in task_ids and dep != task_id}
                if valid_deps:
                    cleaned_deps[task_id] = valid_deps
        
        # Check for circular dependencies
        cleaned_deps = self._remove_circular_dependencies(cleaned_deps)
        
        return cleaned_deps
    
    def _remove_circular_dependencies(self, dependencies: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """
        Remove circular dependencies using Kahn's algorithm for topological sort.
        Returns a new dependency dict with cycles removed.
        """
        if not dependencies:
            return {}
            
        # Create a copy to avoid modifying the original
        deps = {k: set(v) for k, v in dependencies.items()}
        
        # Initialize in-degree and nodes
        in_degree = defaultdict(int)
        nodes = set(deps.keys())
        
        # Calculate in-degree for each node
        for node in nodes:
            in_degree[node] = 0
            
        for node in nodes:
            for dep in deps.get(node, set()):
                in_degree[dep] = in_degree.get(dep, 0) + 1
        
        # Initialize queue with nodes having no incoming edges
        queue = deque([node for node in nodes if in_degree[node] == 0])
        topo_order = []
        
        # Kahn's algorithm
        while queue:
            node = queue.popleft()
            topo_order.append(node)
            
            # Reduce in-degree for all neighbors
            for neighbor in deps.get(node, set()):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # If there are cycles, remove dependencies to break them
        if len(topo_order) != len(nodes):
            # Find nodes in cycles (those not in topo_order)
            cycle_nodes = set(nodes) - set(topo_order)
            
            # For each node in a cycle, remove dependencies that point to nodes in the cycle
            for node in deps:
                deps[node] = {dep for dep in deps[node] if dep not in cycle_nodes}
        
        return deps

    # --- Proposal and review helpers ---
    def propose_dependencies(self, tasks: List[Task], dep_threshold: float = 0.7,
                             mode: str = "balanced") -> List[Dict[str, Any]]:
        """Propose dependency edges with confidence and reason.
        Returns a list of dicts: {from_id, to_id, confidence, reasons}
        where edge is from prerequisite -> dependent.
        """
        proposals: List[Dict[str, Any]] = []
        if not tasks:
            return proposals

        # Build name/desc lookup
        id_to_task = {t.id: t for t in tasks}
        others = tasks

        # 1) Rule/NLP based directional hints (uses existing single-task detection)
        detected = self.detect_dependencies(tasks, other_tasks=others)
        for to_id, dep_ids in detected.items():
            for from_id in dep_ids:
                if from_id == to_id:
                    continue
                proposals.append({
                    'from_id': from_id,
                    'to_id': to_id,
                    'confidence': 0.85,
                    'reasons': ['rule_nlp']
                })

        # 2) Semantic similarity ordering hints (weak, only if balanced/aggressive)
        if mode in ("balanced", "aggressive"):
            try:
                # simple TF-IDF similarity on names+descriptions
                texts = []
                ids = []
                for t in tasks:
                    text = f"{t.name}. {getattr(t, 'description', '')}".strip()
                    texts.append(text)
                    ids.append(t.id)
                if len(texts) >= 2:
                    tfidf = self.vectorizer.fit_transform(texts)
                    sim = cosine_similarity(tfidf, tfidf)
                    for i, a in enumerate(ids):
                        for j, b in enumerate(ids):
                            if i == j:
                                continue
                            score = float(sim[i, j])
                            if score >= dep_threshold:
                                # Heuristic: prefer earlier 'order' as prerequisite if available
                                ta, tb = id_to_task[a], id_to_task[b]
                                order_a = getattr(ta, 'order', None)
                                order_b = getattr(tb, 'order', None)
                                if order_a is not None and order_b is not None and order_a < order_b:
                                    frm, to = a, b
                                else:
                                    # default direction a->b
                                    frm, to = a, b
                                proposals.append({
                                    'from_id': frm,
                                    'to_id': to,
                                    'confidence': min(0.8, max(0.7, score)),
                                    'reasons': ['semantic_similarity']
                                })
            except Exception:
                pass

        # Merge duplicates keeping highest confidence and combined reasons
        merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for p in proposals:
            key = (p['from_id'], p['to_id'])
            if key not in merged or p['confidence'] > merged[key]['confidence']:
                merged[key] = p
            else:
                # combine reasons
                merged[key]['reasons'] = list(set(merged[key]['reasons']) | set(p['reasons']))

        return list(merged.values())

    def apply_dependency_proposals(self, process: Process, proposals: List[Dict[str, Any]],
                                   min_confidence: float = 0.75) -> int:
        """Apply proposals with confidence >= min_confidence to process.tasks.
        Returns the number of edges applied.
        """
        if not proposals:
            return 0
        applied = 0
        tasks_by_id = {t.id: t for t in process.tasks}
        for prop in proposals:
            if prop.get('confidence', 0.0) < min_confidence:
                continue
            frm = prop.get('from_id')
            to = prop.get('to_id')
            if frm not in tasks_by_id or to not in tasks_by_id or frm == to:
                continue
            tgt = tasks_by_id[to]
            deps = set(getattr(tgt, 'dependencies', set()) or set())
            if frm not in deps:
                deps.add(frm)
                # Preserve dependencies as a set for downstream logic relying on set ops
                tgt.dependencies = set(deps)
                applied += 1
        return applied
    
    def detect_sequential_dependencies(self, tasks: List[Task]) -> Dict[str, Set[str]]:
        """
        Enhanced dependency detection for sequential processes (e.g., healthcare)
        Identifies tasks that must follow each other in sequence
        """
        dependencies = {}
        task_map = {task.id: task for task in tasks}
        
        # Sequential keywords indicating order
        sequential_indicators = {
            'after': ['after', 'following', 'post', 'subsequent'],
            'before': ['before', 'prior', 'pre', 'preceding'],
            'then': ['then', 'next', 'followed by'],
            'requires': ['requires', 'needs', 'depends on', 'prerequisite'],
            'completes': ['complete', 'finish', 'done', 'end']
        }
        
        # Healthcare-specific sequential patterns
        if self.process_type == "healthcare":
            healthcare_sequence = [
                'registration', 'triage', 'assessment', 'examination',
                'diagnosis', 'treatment', 'prescription', 'discharge'
            ]
            
            # Create sequential dependencies based on typical flow
            for i in range(len(tasks)):
                for j in range(i + 1, len(tasks)):
                    task1 = tasks[i]
                    task2 = tasks[j]
                    
                    # Check if tasks follow healthcare sequence
                    task1_keywords = task1.name.lower() + ' ' + task1.description.lower()
                    task2_keywords = task2.name.lower() + ' ' + task2.description.lower()
                    
                    for k in range(len(healthcare_sequence) - 1):
                        if healthcare_sequence[k] in task1_keywords and \
                           healthcare_sequence[k + 1] in task2_keywords:
                            if task2.id not in dependencies:
                                dependencies[task2.id] = set()
                            dependencies[task2.id].add(task1.id)
        
        # General sequential pattern detection
        for task in tasks:
            task_text = f"{task.name} {task.description}".lower()
            
            # Look for explicit sequential indicators
            for indicator_type, keywords in sequential_indicators.items():
                for keyword in keywords:
                    if keyword in task_text:
                        # Extract referenced tasks
                        for other_task in tasks:
                            if other_task.id != task.id:
                                other_text = other_task.name.lower()
                                if other_text in task_text:
                                    if indicator_type in ['after', 'requires']:
                                        if task.id not in dependencies:
                                            dependencies[task.id] = set()
                                        dependencies[task.id].add(other_task.id)
                                    elif indicator_type == 'before':
                                        if other_task.id not in dependencies:
                                            dependencies[other_task.id] = set()
                                        dependencies[other_task.id].add(task.id)
        
        return dependencies
    
    def detect_parallel_opportunities(self, tasks: List[Task]) -> List[List[str]]:
        """
        Detect tasks that can be executed in parallel using advanced NLP analysis
        Returns groups of task IDs that can run simultaneously
        """
        parallel_groups = []
        
        # Use advanced NLP analyzer if available
        if self.nlp_analyzer is not None and tasks:
            try:
                # Prepare task data
                task_data = []
                for task in tasks:
                    task_data.append({
                        'id': task.id,
                        'name': task.name,
                        'description': getattr(task, 'description', '')
                    })
                
                # Get parallelization groups from NLP analyzer
                nlp_groups = self.nlp_analyzer.get_parallelization_groups(task_data, min_confidence=0.7)
                
                if nlp_groups:
                    print(f"  [NLP] Detected {len(nlp_groups)} parallelization groups")
                    for i, group in enumerate(nlp_groups):
                        print(f"        Group {i+1}: {len(group)} tasks can run in parallel")
                    return nlp_groups
            except Exception as e:
                print(f"Warning: Error in NLP parallel detection: {e}")
        
        # Fallback to traditional method
        # Find tasks with no dependencies or same dependencies
        dependency_groups = defaultdict(list)
        for task in tasks:
            dep_key = frozenset(task.dependencies) if task.dependencies else frozenset()
            dependency_groups[dep_key].append(task.id)
        
        # Groups with same dependencies can potentially run in parallel
        for dep_set, task_ids in dependency_groups.items():
            if len(task_ids) > 1:
                # Additional checks for parallel compatibility
                if self._can_run_parallel(task_ids, tasks):
                    parallel_groups.append(task_ids)
        
        return parallel_groups
    
    def _can_run_parallel(self, task_ids: List[str], all_tasks: List[Task]) -> bool:
        """
        Check if tasks can actually run in parallel
        Consider resource conflicts and logical constraints
        """
        task_map = {task.id: task for task in all_tasks}
        tasks_to_check = [task_map[tid] for tid in task_ids if tid in task_map]
        
        # Check for resource conflicts
        required_resources = defaultdict(int)
        for task in tasks_to_check:
            for skill in task.required_skills:
                required_resources[skill.name] += 1
        
        # If multiple tasks need the same unique resource, they can't run in parallel
        # This is a simplified check - could be enhanced with actual resource availability
        
        # For healthcare processes, be more restrictive about parallelization
        if self.process_type == "healthcare":
            # In healthcare, tasks involving the patient directly cannot be parallel
            patient_interaction_keywords = [
                'patient', 'examination', 'consultation', 'treatment', 
                'diagnosis', 'assessment', 'procedure'
            ]
            
            patient_tasks = 0
            for task in tasks_to_check:
                task_text = f"{task.name} {task.description}".lower()
                if any(keyword in task_text for keyword in patient_interaction_keywords):
                    patient_tasks += 1
            
            # If multiple tasks involve direct patient interaction, they can't be parallel
            if patient_tasks > 1:
                return False
        
        return True
    
    def detect_critical_sequence(self, tasks: List[Task]) -> List[str]:
        """
        Detect the critical sequence of tasks that form the main flow
        Particularly useful for single-user journey processes
        """
        if not tasks:
            return []
        
        # Build dependency graph
        graph = defaultdict(list)
        reverse_graph = defaultdict(list)
        task_map = {task.id: task for task in tasks}
        
        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id in task_map:
                    graph[dep_id].append(task.id)
                    reverse_graph[task.id].append(dep_id)
        
        # Find tasks with no dependencies (potential start points)
        start_tasks = [task.id for task in tasks if not task.dependencies]
        
        # Find tasks with no dependents (potential end points)
        end_tasks = [
            task.id for task in tasks 
            if task.id not in graph or not graph[task.id]
        ]
        
        # If single start and end, find the path
        if len(start_tasks) == 1 and len(end_tasks) == 1:
            return self._find_longest_path(start_tasks[0], end_tasks[0], graph, task_map)
        
        # Otherwise, find the longest path overall
        longest_path = []
        for start in start_tasks:
            for end in end_tasks:
                path = self._find_longest_path(start, end, graph, task_map)
                if len(path) > len(longest_path):
                    longest_path = path
        
        return longest_path
    
    def _find_longest_path(self, start: str, end: str, 
                           graph: Dict[str, List[str]], 
                           task_map: Dict[str, Task]) -> List[str]:
        """
        Find the longest path between two tasks
        Uses modified BFS to find the path with most tasks
        """
        if start == end:
            return [start]
        
        # BFS with path tracking
        queue = [(start, [start])]
        longest_path = []
        
        while queue:
            current, path = queue.pop(0)
            
            if current == end:
                if len(path) > len(longest_path):
                    longest_path = path
                continue
            
            for next_task in graph.get(current, []):
                if next_task not in path:  # Avoid cycles
                    queue.append((next_task, path + [next_task]))
        
        return longest_path


class DeadlockDetector:
    """
    Detects potential deadlocks in task scheduling and can resolve circular dependencies if enabled.
    """
    
    def resolve_dependency_deadlocks(self, process: Process, log_callback=None) -> bool:
        """
        Detect and break cycles in the process dependency graph by removing minimal dependencies.
        Modifies process.tasks in-place. Returns True if any cycles were found and broken.
        log_callback: optional function to log (task_id, removed_dep_id)
        """
        # Build adjacency list and track in/out degrees
        adj = {}
        in_degree = {}
        out_degree = {}
        
        # Initialize data structures
        for task in process.tasks:
            task_id = task.id
            deps = set(getattr(task, 'dependencies', []))
            adj[task_id] = deps
            in_degree[task_id] = 0
            out_degree[task_id] = 0
        
        # Calculate in/out degrees
        for task_id in adj:
            out_degree[task_id] = len(adj[task_id])
            for dep_id in adj[task_id]:
                if dep_id in in_degree:
                    in_degree[dep_id] += 1
        
        # Find cycles using iterative DFS
        removed = []
        
        def find_cycle():
            visited = set()
            path = []
            cycle = None
            
            def visit(node):
                nonlocal cycle
                if node in visited:
                    if node in path:
                        idx = path.index(node)
                        cycle = path[idx:]
                    return
                
                visited.add(node)
                path.append(node)
                
                for neighbor in adj.get(node, []):
                    if cycle is not None:
                        break
                    visit(neighbor)
                
                path.pop()
            
            for node in adj:
                if cycle is not None:
                    break
                if node not in visited:
                    visit(node)
            
            return cycle
        
        # Keep breaking cycles until none are left
        cycle_found = True
        while cycle_found:
            cycle = find_cycle()
            if not cycle:
                cycle_found = False
                continue
                
            # Find the best dependency to break (one with highest out_degree - in_degree)
            best_score = -1
            best_edge = None
            
            for i in range(len(cycle)):
                src = cycle[i]
                dst = cycle[(i+1) % len(cycle)]
                if dst in adj[src]:
                    score = out_degree[src] - in_degree[dst]
                    if score > best_score:
                        best_score = score
                        best_edge = (src, dst)
            
            if best_edge:
                src, dst = best_edge
                adj[src].remove(dst)
                out_degree[src] -= 1
                in_degree[dst] -= 1
                removed.append((src, dst))
                if log_callback:
                    log_callback(src, dst)
        
        # Apply removals to tasks
        if removed:
            removed_set = set(removed)
            for task in process.tasks:
                if hasattr(task, 'dependencies'):
                    # Work with a set and preserve set type after filtering
                    deps = set(getattr(task, 'dependencies', set()) or set())
                    task.dependencies = {dep for dep in deps if (task.id, dep) not in removed_set}

        return len(removed) > 0

    """Detects potential deadlocks in task scheduling"""
    
    def detect_deadlocks(self, process: Process, schedule: Schedule) -> List[Dict[str, Any]]:
        """
        Detect potential deadlocks in the process
        Returns list of deadlock scenarios with details
        """
        deadlocks = []
        
        # Resource contention deadlocks
        resource_deadlocks = self._detect_resource_deadlocks(process, schedule)
        deadlocks.extend(resource_deadlocks)
        
        # Circular dependency deadlocks
        dependency_deadlocks = self._detect_dependency_deadlocks(process)
        deadlocks.extend(dependency_deadlocks)
        
        # Capacity deadlocks
        capacity_deadlocks = self._detect_capacity_deadlocks(process, schedule)
        deadlocks.extend(capacity_deadlocks)
        
        return deadlocks
    
    def _detect_resource_deadlocks(self, process: Process, schedule: Schedule) -> List[Dict[str, Any]]:
        """Detect deadlocks caused by resource contention"""
        deadlocks = []
        
        # Group tasks by time slots and check for resource conflicts
        time_slots = defaultdict(list)
        for entry in schedule.entries:
            time_key = entry.start_time.strftime("%Y-%m-%d %H")
            time_slots[time_key].append(entry)
        
        for time_key, entries in time_slots.items():
            resource_usage = defaultdict(list)
            for entry in entries:
                resource_usage[entry.resource_id].append(entry.task_id)
            
            # Check for over-allocation
            for resource_id, task_ids in resource_usage.items():
                if len(task_ids) > 1:
                    resource = process.get_resource_by_id(resource_id)
                    if resource:
                        deadlocks.append({
                            'type': 'resource_contention',
                            'resource_id': resource_id,
                            'resource_name': resource.name,
                            'conflicting_tasks': task_ids,
                            'time_slot': time_key,
                            'severity': 'high',
                            'resolution_suggestions': [
                                'reschedule',  # Try moving tasks to different times
                                'reassign'     # Try assigning to different resources
                            ]
                        })
        
        return deadlocks
        
    def resolve_resource_contention(self, process: Process, schedule: Schedule, deadlock: Dict) -> bool:
        """
        Attempt to resolve resource contention by either:
        1. Moving conflicting tasks to different time slots, or
        2. Reassigning tasks to alternative resources
        
        Returns True if resolution was successful, False otherwise
        """
        if deadlock['type'] != 'resource_contention':
            return False
            
        resource_id = deadlock['resource_id']
        task_ids = deadlock['conflicting_tasks']
        resource = process.get_resource_by_id(resource_id)
        
        if not resource:
            return False
            
        print(f"[AUTO-RESOLVE] Resolving resource contention for {resource.name} (ID: {resource_id})")
        
        # Try to find alternative resources for each task
        for task_id in task_ids[1:]:  # Keep first task on this resource
            task = next((t for t in process.tasks if t.id == task_id), None)
            if not task:
                continue
                
            # Find available resources with required skills
            required_skills = task.required_skills
            candidate_resources = [r for r in process.resources 
                                 if r.id != resource_id and 
                                 all(any(rs.name == s.name and rs.level.value >= s.level.value 
                                     for rs in r.skills) for s in required_skills)]
            
            if candidate_resources:
                # Sort by current workload (least loaded first)
                candidate_resources.sort(key=lambda r: len([e for e in schedule.entries if e.resource_id == r.id]))
                new_resource = candidate_resources[0]
                
                # Update task assignment
                task.assigned_resource = new_resource.id
                print(f"[AUTO-RESOLVE] Reassigned task '{task.name}' to {new_resource.name} (ID: {new_resource.id})")
                return True
        
        # If we get here, couldn't reassign - try rescheduling
        print("[AUTO-RESOLVE] Could not reassign tasks, attempting to reschedule...")
        return False  # Let the optimizer handle rescheduling in the next pass
    
    def _detect_dependency_deadlocks(self, process: Process) -> List[Dict[str, Any]]:
        """Detect circular dependency deadlocks"""
        deadlocks = []
        
        # Build dependency graph
        graph = nx.DiGraph()
        for task in process.tasks:
            graph.add_node(task.id)
            for dep in task.dependencies:
                graph.add_edge(dep, task.id)
        
        # Find cycles
        try:
            cycles = list(nx.simple_cycles(graph))
            for cycle in cycles:
                task_names = []
                for task_id in cycle:
                    task = process.get_task_by_id(task_id)
                    task_names.append(task.name if task else task_id)
                
                deadlocks.append({
                    'type': 'circular_dependency',
                    'cycle_tasks': cycle,
                    'cycle_names': task_names,
                    'severity': 'critical'
                })
        except:
            pass
        
        return deadlocks
    
    def _detect_capacity_deadlocks(self, process: Process, schedule: Schedule) -> List[Dict[str, Any]]:
        """Detect deadlocks caused by insufficient capacity"""
        deadlocks = []
        
        # Check if all tasks can be completed with available resources
        unassigned_tasks = []
        for task in process.tasks:
            if not schedule.get_task_schedule(task.id):
                unassigned_tasks.append(task)
        
        for task in unassigned_tasks:
            available_resources = process.get_available_resources(
                task.required_skills, 
                process.start_date, 
                task.duration_hours
            )
            
            if not available_resources:
                deadlocks.append({
                    'type': 'insufficient_capacity',
                    'task_id': task.id,
                    'task_name': task.name,
                    'required_skills': [str(skill) for skill in task.required_skills],
                    'severity': 'high'
                })
        
        return deadlocks


class WhatIfAnalyzer:
    """
    Enhanced What-If Analyzer for RL-based process optimization with automatic dependency detection.
    
    Features:
    - RL-optimizer specific scenario testing
    - Automatic dependency detection and validation
    - Deadlock detection and resolution
    - RL-specific metrics and analysis
    """
    
    def __init__(self, rl_optimizer=None):
        """
        Initialize the analyzer with an RL optimizer.
        
        Args:
            rl_optimizer: An instance of RLBasedOptimizer
        """
        self.rl_optimizer = rl_optimizer
        self.dependency_detector = DependencyDetector()
        self.deadlock_detector = DeadlockDetector()
    
    def analyze_scenarios(self, process: Process, scenarios: Optional[List[Dict[str, Any]]] = None,
                         time_weight: float = 0.6, cost_weight: float = 0.4,
                         auto_detect_dependencies: bool = True,
                         baseline_schedule: Optional[Schedule] = None) -> Dict[str, Any]:
        """
        Analyze multiple what-if scenarios using RL optimization with automatic dependency detection.
        
        Args:
            process: The base process to analyze
            scenarios: Optional list of scenario configurations. If None, uses default RL scenarios.
            time_weight: Weight for time optimization (0-1)
            cost_weight: Weight for cost optimization (0-1)
            auto_detect_dependencies: Whether to automatically detect and apply task dependencies
            baseline_schedule: Optional pre-computed baseline schedule to use instead of running baseline optimization
            
        Returns:
            Dict containing analysis results and the best scenario
        """
        if not (0 <= time_weight <= 1 and 0 <= cost_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")
            
        if abs(time_weight + cost_weight - 1.0) > 0.01:
            raise ValueError("Sum of weights must equal 1.0")
        
        # Use default RL scenarios if none provided
        if scenarios is None:
            scenarios = self._get_default_rl_scenarios()
            
        results = {
            'scenarios': {},
            'time_weight': time_weight,
            'cost_weight': cost_weight,
            'auto_detect_dependencies': auto_detect_dependencies
        }
        
        # Make a deep copy of the process to avoid modifying the original
        base_process = copy.deepcopy(process)
        
        # Auto-detect dependencies if enabled
        if auto_detect_dependencies:
            self._apply_auto_dependencies(base_process)
        
        # Use provided baseline schedule or generate one
        if baseline_schedule:
            print("\n" + "="*50)
            print("USING PROVIDED BASELINE SCHEDULE")
            print("="*50)
            # Use the provided baseline schedule
            baseline_metrics = self._calculate_metrics(base_process, baseline_schedule)
        else:
            # Get baseline schedule with RL optimizer
            print("\n" + "="*50)
            print("RUNNING BASELINE OPTIMIZATION")
            print("="*50)
            
            baseline_schedule = self._run_optimization_with_retry(base_process)
            if not baseline_schedule:
                raise RuntimeError("Failed to generate baseline schedule")
                
            # Calculate baseline metrics
            baseline_metrics = self._calculate_metrics(base_process, baseline_schedule)
        
        results['baseline'] = baseline_metrics
        results['baseline_schedule'] = baseline_schedule
        # Expose baseline on the analyzer instance for later calls like analyze_impact()
        self.baseline_metrics = baseline_metrics
        self.baseline_schedule = baseline_schedule
        
        print(f"\nBaseline metrics: Duration={baseline_metrics['total_duration']:.1f}h, "
              f"Cost=${baseline_metrics['total_cost']:,.2f}")
              
        # Analyze each scenario
        for i, scenario in enumerate(scenarios):
            scenario_id = f"scenario_{i+1}"
            scenario_name = scenario.get('name', f"Scenario {i+1}")
            
            print(f"\n{'='*50}")
            print(f"RUNNING SCENARIO: {scenario_name}")
            print("="*50)
            
            try:
                # Apply scenario modifications to a fresh copy of the process
                scenario_process = self._apply_scenario(copy.deepcopy(base_process), scenario)
                
                # Run optimization with retry and deadlock resolution
                scenario_schedule = self._run_optimization_with_retry(scenario_process)
                
                if not scenario_schedule:
                    results['scenarios'][scenario_id] = {
                        'name': scenario_name,
                        'error': 'Optimization failed',
                        'scenario': scenario
                    }
                    continue
                    
                # Calculate metrics and improvements
                scenario_metrics = self._calculate_metrics(scenario_process, scenario_schedule)
                improvement = self._calculate_improvement(
                    baseline_metrics, scenario_metrics, time_weight, cost_weight
                )
                
                # Store results
                results['scenarios'][scenario_id] = {
                    'name': scenario_name,
                    'scenario': scenario,
                    'metrics': scenario_metrics,
                    'improvement': improvement,
                    'schedule': scenario_schedule
                }
                
                print(f"\nScenario '{scenario_name}' results:")
                print(f"- Duration: {scenario_metrics['total_duration']:.1f}h "
                      f"({float(improvement.get('time_pct', 0.0)):+.1f}%)")
                print(f"- Cost: ${scenario_metrics['total_cost']:,.2f} "
                      f"({float(improvement.get('cost_pct', 0.0)):+.1f}%)")
                print(f"- Score: {float(improvement.get('score', 0.0)):.3f}")
                
            except Exception as e:
                print(f"Error in scenario '{scenario_name}': {str(e)}")
                import traceback
                traceback.print_exc()
                
                results['scenarios'][scenario_id] = {
                    'name': scenario_name,
                    'scenario': scenario,
                    'error': str(e)
                }
        
        # Determine the best scenario based on the weighted score
        valid_scenarios = [
            (scenario_id, data) 
            for scenario_id, data in results['scenarios'].items() 
            if 'metrics' in data and 'improvement' in data
        ]
        
        if valid_scenarios:
            best_scenario_id, best_scenario = max(
                valid_scenarios,
                key=lambda x: float(x[1].get('improvement', {}).get('score', float('-inf')))
            )
            results['best_scenario'] = {
                'id': best_scenario_id,
                'name': best_scenario['name'],
                'improvement': best_scenario['improvement']
            }
            
            print("\n" + "="*50)
            print(f"BEST SCENARIO: {best_scenario['name']}")
            print("="*50)
            print(f"Score: {float(best_scenario.get('improvement', {}).get('score', 0.0)):.3f}")
            print(f"Time Improvement: {float(best_scenario.get('improvement', {}).get('time_pct', 0.0)):+.1f}%")
            print(f"Cost Improvement: {float(best_scenario.get('improvement', {}).get('cost_pct', 0.0)):+.1f}%")
        else:
            results['best_scenario'] = None
            print("\nNo valid scenarios to determine best result")
            
        # Calculate absolute savings for the best scenario
        if 'best_scenario' in results and results['best_scenario'] is not None:
            best_scenario_id = results['best_scenario']['id']
            try:
                baseline_duration = results['baseline']['total_duration']
                best_duration = results['scenarios'][best_scenario_id]['metrics']['total_duration']
                time_savings = baseline_duration - best_duration
                
                baseline_cost = results['baseline']['total_cost']
                best_cost = results['scenarios'][best_scenario_id]['metrics']['total_cost']
                cost_savings = baseline_cost - best_cost
                
                results['savings'] = {
                    'time_hours': max(0, time_savings),
                    'time_percentage': (time_savings / baseline_duration * 100) if baseline_duration > 0 else 0,
                    'cost_dollars': max(0, cost_savings),
                    'cost_percentage': (cost_savings / baseline_cost * 100) if baseline_cost > 0 else 0
                }
            except (KeyError, TypeError, ZeroDivisionError) as e:
                print(f"[WARNING] Could not calculate savings: {str(e)}")
                results['savings'] = {
                    'time_hours': 0,
                    'time_percentage': 0,
                    'cost_dollars': 0,
                    'cost_percentage': 0
                }
        
        return results
    
    def _get_default_rl_scenarios(self) -> List[Dict[str, Any]]:
        """Get default RL-specific scenarios to test"""
        return [
            {
                'name': 'Balanced Optimization',
                'description': 'Balanced approach between time and cost',
                'parameters': {
                    'time_weight': 0.5,
                    'cost_weight': 0.5,
                    'exploration_rate': 0.1,
                    'discount_factor': 0.9
                }
            },
            {
                'name': 'High Exploration',
                'description': 'Increase exploration to diversify assignments',
                'parameters': {
                    'exploration_rate': 0.4,
                    'discount_factor': 0.9
                }
            },
            {
                'name': 'Time-Critical',
                'description': 'Prioritize fastest completion time',
                'parameters': {
                    'time_weight': 0.8,
                    'cost_weight': 0.2,
                    'exploration_rate': 0.15,
                    'discount_factor': 0.95
                }
            },
            {
                'name': 'Cost-Sensitive',
                'description': 'Prioritize lowest cost',
                'parameters': {
                    'time_weight': 0.2,
                    'cost_weight': 0.8,
                    'exploration_rate': 0.2,
                    'discount_factor': 0.85
                }
            },
            {
                'name': 'Resource-Efficient',
                'description': 'Optimize for resource utilization',
                'parameters': {
                    'time_weight': 0.4,
                    'cost_weight': 0.6,
                    'load_balancing_factor': 1.5,
                    'exploration_rate': 0.1
                }
            },
            {
                'name': 'More Parallelism',
                'description': 'Broaden choices per task and allow more parallel tasks',
                'parameters': {
                    'top_k_resources_per_task': 5,
                    'max_parallel_tasks': 6,
                    'exploration_rate': 0.25
                }
            },
            {
                'name': 'Load Balanced',
                'description': 'Penalize heavy concentration on few resources',
                'parameters': {
                    'load_balancing_factor': 2.0,
                    'exploration_rate': 0.15
                }
            },
            {
                'name': 'Cheapest-First',
                'description': 'Strongly prefer cost minimization',
                'parameters': {
                    'time_weight': 0.1,
                    'cost_weight': 0.9,
                    'exploration_rate': 0.2
                }
            },
            {
                'name': 'Temp Contractor',
                'description': 'Add a temporary contractor with broad skills and competitive rate',
                'add_resources': [
                    {
                        'id': 'contractor_temp',
                        'name': 'Contractor (Temp)',
                        'hourly_rate': 60.0,
                        'skills': [
                            {'name': 'business_analysis', 'level': 4},
                            {'name': 'documentation', 'level': 4},
                            {'name': 'system_design', 'level': 4},
                            {'name': 'architecture', 'level': 4},
                            {'name': 'database_design', 'level': 4},
                            {'name': 'sql', 'level': 4},
                            {'name': 'frontend_development', 'level': 4},
                            {'name': 'javascript', 'level': 4},
                            {'name': 'react', 'level': 4},
                            {'name': 'backend_development', 'level': 4},
                            {'name': 'api_design', 'level': 4},
                            {'name': 'api_integration', 'level': 4},
                            {'name': 'testing', 'level': 4},
                            {'name': 'python', 'level': 4},
                            {'name': 'api_testing', 'level': 4},
                            {'name': 'code_review', 'level': 4},
                            {'name': 'security', 'level': 3},
                            {'name': 'devops', 'level': 4},
                            {'name': 'deployment', 'level': 4},
                            {'name': 'technical_writing', 'level': 3},
                            {'name': 'ui_design', 'level': 4},
                            {'name': 'ux_design', 'level': 4}
                        ]
                    }
                ]
            }
        ]
        
    def _apply_auto_dependencies(self, process: Process) -> None:
        """Automatically detect and apply dependencies to tasks"""
        print("Detecting task dependencies...")
        
        # Detect dependencies using NLP, rule-based, and resource-based methods
        try:
            raw_dependencies = self.dependency_detector.detect_dependencies(
                tasks=process.tasks,
                other_tasks=process.tasks,
                resources=process.resources,
            )
        except Exception as _e:
            print(f"[WARNING] Dependency detection failed, proceeding without NLP/resource context: {_e}")
            raw_dependencies = self.dependency_detector.detect_dependencies(process.tasks)

        # Validate detected dependencies
        try:
            dependencies = self.dependency_detector.validate_dependencies(process.tasks, raw_dependencies)
        except Exception as _e:
            print(f"[WARNING] Dependency validation failed, using raw dependencies: {_e}")
            dependencies = raw_dependencies or {}

        # Apply validated dependencies to tasks
        any_applied = False
        for task in process.tasks:
            deps = set(dependencies.get(task.id, set()) or [])
            if deps:
                task.dependencies = deps
                any_applied = True
                print(f"  - {task.name} depends on: {', '.join(sorted(deps))}")

        # Fallback: if nothing detected, derive a simple chain from the 'order' field
        if not any_applied:
            try:
                ordered = [t for t in process.tasks if hasattr(t, 'order') and isinstance(getattr(t, 'order'), (int, float))]
                if len(ordered) == len(process.tasks):
                    ordered.sort(key=lambda t: getattr(t, 'order'))
                    # Ensure unique ordering before chaining
                    order_values = [getattr(t, 'order') for t in ordered]
                    if len(set(order_values)) == len(order_values) and len(ordered) > 1:
                        for prev, curr in zip(ordered, ordered[1:]):
                            curr.dependencies = set(getattr(curr, 'dependencies', set()) or set())
                            curr.dependencies.add(prev.id)
                        print("  - No dependencies detected; applied sequential chain based on 'order' field")
                        any_applied = True
            except Exception as _e:
                print(f"[WARNING] Fallback order-based dependency application failed: {_e}")

        # Resolve any circular dependencies
        if self.deadlock_detector.resolve_dependency_deadlocks(process):
            print("Resolved circular dependencies")
    
    def _run_optimization_with_retry(self, process: Process, max_retries: int = 3) -> Optional[Schedule]:
        """Run optimization with retry logic for deadlock resolution"""
        for attempt in range(max_retries):
            try:
                # Don't reset on the first attempt - preserve trained state
                # Only reset if we're retrying due to deadlocks
                if attempt > 0 and hasattr(self.rl_optimizer, 'reset'):
                    self.rl_optimizer.reset()
                
                # Configure RL optimizer parameters if specified in the process
                if hasattr(process, 'optimizer_params'):
                    for param, value in process.optimizer_params.items():
                        # Map scenario-friendly names to optimizer attribute names
                        target_param = param
                        if param == 'exploration_rate':
                            target_param = 'epsilon'
                        elif param == 'exploration_decay_rate':
                            target_param = 'exploration_decay'
                        # Set attribute if present or allow dynamic setattr for weights
                        if hasattr(self.rl_optimizer, target_param) or target_param in (
                            'time_weight', 'cost_weight', 'load_balancing_factor', 'top_k_resources_per_task',
                            'max_parallel_tasks', 'enable_parallel'
                        ):
                            setattr(self.rl_optimizer, target_param, value)
                            print(f"  Applied optimizer param: {target_param} = {value}")
                
                # Run optimization using the existing trained model
                # Use generate_schedule_from_model to avoid re-training
                if hasattr(self.rl_optimizer, 'generate_schedule_from_model'):
                    schedule = self.rl_optimizer.generate_schedule_from_model(process)
                else:
                    # Fallback to optimize if method not available
                    schedule = self.rl_optimizer.optimize(process)
                
                # Check for deadlocks
                deadlocks = self.deadlock_detector.detect_deadlocks(process, schedule)
                if deadlocks and attempt < max_retries - 1:
                    print(f"  Deadlock detected, attempting resolution (attempt {attempt + 1}/{max_retries})...")
                    self.deadlock_detector.resolve_resource_contention(process, schedule, deadlocks[0])
                    continue
                    
                return schedule
                
            except Exception as e:
                print(f"  Optimization attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    print("  Max retries reached, giving up")
                    return None
    
    def _calculate_metrics(self, process: Process, schedule: Schedule) -> Dict[str, Any]:
        """Calculate metrics for a schedule"""
        return {
            'total_duration': schedule.total_duration_hours,
            'total_cost': schedule.total_cost,
            'resource_utilization': schedule.resource_utilization,
            'task_count': len(schedule.entries),
            'parallelism': self._calculate_parallelism(schedule),
            'deadlocks': len(self.deadlock_detector.detect_deadlocks(process, schedule) or [])
        }
    
    def _calculate_parallelism(self, schedule: Schedule) -> float:
        """Calculate the average degree of parallelism in the schedule"""
        if not schedule.entries:
            return 0.0
            
        # Group tasks by start time
        time_slots = defaultdict(list)
        for entry in schedule.entries:
            time_slots[entry.start_time].append(entry)
            
        # Calculate average parallelism
        if not time_slots:
            return 0.0
            
        return sum(len(tasks) for tasks in time_slots.values()) / len(time_slots)
    
    def analyze_impact(self, scenario_schedule: Schedule, scenario_name: str) -> Dict[str, float]:
        """
        Analyze the impact of a scenario compared to the baseline.
        
        Args:
            scenario_schedule: The schedule from the scenario optimization
            scenario_name: Name of the scenario being analyzed
            
        Returns:
            Dictionary containing impact metrics
        """
        if not hasattr(self, 'baseline_metrics') or not self.baseline_metrics:
            print("Warning: No baseline metrics available for comparison")
            return {}
            
        # Calculate improvements using the existing _calculate_improvement method
        scenario_metrics = {
            'total_duration': getattr(scenario_schedule, 'total_duration_hours', 0.0),
            'total_cost': getattr(scenario_schedule, 'total_cost', 0.0),
        }
        improvements = self._calculate_improvement(
            baseline=self.baseline_metrics,
            scenario=scenario_metrics,
            time_weight=0.6,  # Default weights - can be made configurable
            cost_weight=0.4
        )
        
        # Format the results for the summary
        impact = {
            'scenario': scenario_name,
            'duration_change_pct': improvements.get('time_pct', 0.0),
            'cost_change_pct': improvements.get('cost_pct', 0.0),
            'total_improvement': improvements.get('score', 0.0),
            'parallelism_change': improvements.get('parallelism_change', 0.0)
        }
        
        print(f"\nScenario '{scenario_name}' impact analysis:")
        print(f"- Duration change: {impact['duration_change_pct']:+.1f}%")
        print(f"- Cost change: {impact['cost_change_pct']:+.1f}%")
        print(f"- Total improvement score: {impact['total_improvement']:.2f}")
        
        return impact
        
    def _calculate_improvement(self, baseline: Dict, scenario: Dict, 
                             time_weight: float = 0.6, cost_weight: float = 0.4) -> Dict[str, float]:
        """
        Calculate improvement metrics for a scenario compared to baseline.
        
        Args:
            baseline: Dictionary containing baseline metrics or baseline schedule
            scenario_schedule: Schedule object from scenario optimization
            time_weight: Weight for time optimization (0-1)
            cost_weight: Weight for cost optimization (0-1)
            
        Returns:
            Dict with improvement metrics including time, cost, and combined score
        """
        if not baseline:
            return {}
            
        improvements = {}
        
        try:
            # Expect both baseline and scenario as metrics dicts
            baseline_duration = float(baseline.get('total_duration', 0) or 0)
            baseline_cost = float(baseline.get('total_cost', 0) or 0)
            scenario_duration = float(scenario.get('total_duration', 0) or 0)
            scenario_cost = float(scenario.get('total_cost', 0) or 0)

            # Calculate time improvement (lower is better)
            if baseline_duration > 0:
                improvements['time_pct'] = (baseline_duration - scenario_duration) / baseline_duration * 100.0

            # Calculate cost improvement (lower is better)
            if baseline_cost > 0:
                improvements['cost_pct'] = (baseline_cost - scenario_cost) / baseline_cost * 100.0

            # Combined score (weighted)
            if 'time_pct' in improvements and 'cost_pct' in improvements:
                improvements['score'] = improvements['time_pct'] * time_weight + improvements['cost_pct'] * cost_weight

            return improvements

        except (TypeError, ZeroDivisionError, AttributeError) as e:
            print(f"Warning: Error calculating improvements: {e}")
            return {
                'time_pct': 0.0,
                'cost_pct': 0.0,
                'score': 0.0,
                'parallelism_change': 0.0
            }
    
    def _apply_scenario(self, process: Process, scenario: Dict[str, Any]) -> Process:
        """
        Apply scenario modifications to the process
        
        Args:
            process: The process to modify
            scenario: Scenario configuration with parameters to apply
            
        Returns:
            Process: A new Process instance with the scenario modifications applied
        """
        import copy
        
        # Create a deep copy of the process to avoid modifying the original
        modified_process = copy.deepcopy(process)
        
        # Apply modifications based on scenario parameters
        if 'parameters' in scenario:
            # Store optimizer parameters to be used during optimization
            if not hasattr(modified_process, 'optimizer_params'):
                modified_process.optimizer_params = {}
            # Update with scenario parameters
            modified_process.optimizer_params.update(scenario['parameters'])
            print(f"Applied scenario parameters: {scenario['parameters']}")

        # Add new resources if specified
        if 'add_resources' in scenario and scenario['add_resources']:
            try:
                from process_optimization_agent.models import Resource, Skill
                for res in scenario['add_resources']:
                    skills = []
                    for sk in res.get('skills', []):
                        if isinstance(sk, dict):
                            skills.append(Skill(name=sk.get('name'), level=sk.get('level', 1)))
                        else:
                            skills.append(Skill(name=str(sk), level=1))
                    modified_process.resources.append(
                        Resource(
                            id=res['id'],
                            name=res.get('name', res['id']),
                            hourly_rate=res.get('hourly_rate', 100.0),
                            skills=skills
                        )
                    )
                print(f"Applied scenario: added {len(scenario['add_resources'])} resource(s)")
            except Exception as _e:
                print(f"[WARNING] Failed to add resources for scenario '{scenario.get('name','')}': {_e}")

        # Modify existing resources if specified (e.g., hourly_rate tweaks)
        if 'modify_resources' in scenario and scenario['modify_resources']:
            try:
                res_by_id = {r.id: r for r in modified_process.resources}
                for change in scenario['modify_resources']:
                    rid = change.get('id')
                    if rid and rid in res_by_id:
                        r = res_by_id[rid]
                        if 'hourly_rate' in change:
                            r.hourly_rate = change['hourly_rate']
                print("Applied resource modifications")
            except Exception as _e:
                print(f"[WARNING] Failed to modify resources: {_e}")
        
        return modified_process
    
    def _calculate_improvement_legacy(self, baseline: Dict, scenario_schedule: Schedule) -> Dict[str, float]:
        """
        Calculate improvement metrics for a scenario compared to baseline.
        
        Args:
            baseline: Dictionary containing baseline metrics
            scenario_schedule: Schedule object from scenario optimization
            
        Returns:
            Dict with improvement metrics including time, cost, and combined score
        """
        if not baseline:
            return {}
            
        improvements = {}
        
        # Calculate time improvement (lower is better)
        if 'total_duration' in baseline and hasattr(scenario_schedule, 'total_duration_hours'):
            time_improvement = (baseline['total_duration'] - scenario_schedule.total_duration_hours) / \
                             baseline['total_duration'] * 100
            improvements['time_pct'] = time_improvement
        
        # Calculate cost improvement (lower is better)
        if 'total_cost' in baseline and hasattr(scenario_schedule, 'total_cost'):
            cost_improvement = (baseline['total_cost'] - scenario_schedule.total_cost) / \
                             baseline['total_cost'] * 100
            improvements['cost_pct'] = cost_improvement
            
        # Calculate combined score (weighted average)
        if 'time_pct' in improvements and 'cost_pct' in improvements:
            improvements['score'] = (improvements['time_pct'] * 0.6 + 
                                   improvements['cost_pct'] * 0.4)
        
        return improvements


class ProcessMiner:
    """Learns from historical process data to improve optimization"""
    
    def __init__(self):
        """Initialize process mining capabilities"""
        self.historical_data = []
        self.patterns = {}
        self.performance_metrics = {}
    
    def add_historical_process(self, process: Process, schedule: Schedule, 
                             actual_metrics: Optional[Dict[str, Any]] = None):
        """Add a completed process to historical data"""
        self.historical_data.append({
            'process': process,
            'schedule': schedule,
            'actual_metrics': actual_metrics or {},
            'timestamp': datetime.now()
        })
    
    def mine_patterns(self) -> Dict[str, Any]:
        """Mine patterns from historical data"""
        if not self.historical_data:
            return {}
        
        patterns = {
            'common_dependencies': self._mine_dependency_patterns(),
            'resource_preferences': self._mine_resource_patterns(),
            'duration_estimates': self._mine_duration_patterns(),
            'bottlenecks': self._mine_bottleneck_patterns()
        }
        
        self.patterns = patterns
        return patterns
    
    def _mine_dependency_patterns(self) -> Dict[str, float]:
        """Mine common dependency patterns"""
        dependency_counts = defaultdict(int)
        total_processes = len(self.historical_data)
        
        for data in self.historical_data:
            process = data['process']
            for task in process.tasks:
                for dep in task.dependencies:
                    dep_task = process.get_task_by_id(dep)
                    if dep_task:
                        pattern = f"{dep_task.name.lower()} -> {task.name.lower()}"
                        dependency_counts[pattern] += 1
        
        # Convert to probabilities
        dependency_patterns = {}
        for pattern, count in dependency_counts.items():
            dependency_patterns[pattern] = count / total_processes
        
        return dependency_patterns
    
    def _mine_resource_patterns(self) -> Dict[str, Dict[str, float]]:
        """Mine resource assignment patterns"""
        resource_patterns = defaultdict(lambda: defaultdict(int))
        
        for data in self.historical_data:
            schedule = data['schedule']
            process = data['process']
            
            for entry in schedule.entries:
                task = process.get_task_by_id(entry.task_id)
                resource = process.get_resource_by_id(entry.resource_id)
                
                if task and resource:
                    # Track task type -> resource skill patterns
                    for skill in task.required_skills:
                        for res_skill in resource.skills:
                            if skill.name.lower() == res_skill.name.lower():
                                pattern = f"{skill.name.lower()}"
                                resource_patterns[pattern][resource.id] += 1
        
        # Convert to preferences
        preferences = {}
        for skill, resource_counts in resource_patterns.items():
            total = sum(resource_counts.values())
            preferences[skill] = {
                res_id: count / total 
                for res_id, count in resource_counts.items()
            }
        
        return preferences
    
    def _mine_duration_patterns(self) -> Dict[str, Dict[str, float]]:
        """Mine task duration estimation patterns"""
        duration_patterns = defaultdict(list)
        
        for data in self.historical_data:
            actual_metrics = data.get('actual_metrics', {})
            if 'task_durations' in actual_metrics:
                process = data['process']
                for task_id, actual_duration in actual_metrics['task_durations'].items():
                    task = process.get_task_by_id(task_id)
                    if task:
                        estimated_duration = task.duration_hours
                        if estimated_duration > 0:
                            ratio = actual_duration / estimated_duration
                            task_type = self._categorize_task(task)
                            duration_patterns[task_type].append(ratio)
        
        # Calculate statistics
        duration_stats = {}
        for task_type, ratios in duration_patterns.items():
            if ratios:
                duration_stats[task_type] = {
                    'mean_ratio': sum(ratios) / len(ratios),
                    'std_ratio': (sum((r - sum(ratios)/len(ratios))**2 for r in ratios) / len(ratios))**0.5,
                    'sample_count': len(ratios)
                }
        
        return duration_stats
    
    def _mine_bottleneck_patterns(self) -> Dict[str, Any]:
        """Mine common bottleneck patterns"""
        bottlenecks = defaultdict(int)
        
        for data in self.historical_data:
            schedule = data['schedule']
            if schedule.critical_path:
                for task_id in schedule.critical_path:
                    bottlenecks[task_id] += 1
        
        return dict(bottlenecks)
    
    def _categorize_task(self, task: Task):
        """Categorize task based on name and description"""
        # Simple categorization based on task name keywords
        name = task.name.lower()
        if any(keyword in name for keyword in ['test', 'qa', 'verify']):
            return 'testing'
        elif any(keyword in name for keyword in ['dev', 'implement', 'code', 'build']):
            return 'development'
        elif any(keyword in name for keyword in ['review', 'inspect', 'audit']):
            return 'review'
        elif any(keyword in name for keyword in ['deploy', 'release', 'publish']):
            return 'deployment'
        elif any(keyword in name for keyword in ['plan', 'design', 'architecture']):
            return 'design'
        elif any(keyword in name for keyword in ['meeting', 'discuss', 'sync']):
            return 'meeting'
        return 'other'
        
    def analyze(self, process: Process, schedule: Schedule) -> List[str]:
        """
        Analyze process and schedule to generate insights
        
        Args:
            process: The process being analyzed
            schedule: The schedule to analyze
            
        Returns:
            List of insight strings
        """
        insights = []
        
        # Check for unscheduled tasks
        scheduled_task_ids = {entry.task_id for entry in schedule.entries}
        unscheduled_tasks = [t for t in process.tasks if t.id not in scheduled_task_ids]
        
        if unscheduled_tasks:
            insight = f"Found {len(unscheduled_tasks)} unscheduled tasks. "
            insight += "Consider reviewing task dependencies and resource assignments."
            insights.append(insight)
        
        # Check resource utilization
        if hasattr(schedule, 'resource_utilization') and schedule.resource_utilization:
            utilizations = [
                (rid, hours) 
                for rid, hours in schedule.resource_utilization.items() 
                if isinstance(hours, (int, float))
            ]
            
            if utilizations:
                resource_map = {r.id: r for r in process.resources}
                max_util = max(utilizations, key=lambda x: x[1])
                min_util = min(utilizations, key=lambda x: x[1])
                
                max_resource = resource_map.get(max_util[0], None)
                min_resource = resource_map.get(min_util[0], None)
                
                if max_resource and min_resource and max_util[1] > 0:
                    insight = f"Resource utilization varies from {min_util[1]:.1f}h ({min_resource.name}) "
                    insight += f"to {max_util[1]:.1f}h ({max_resource.name}). Consider rebalancing workload."
                    insights.append(insight)
        
        # Check for long-running tasks
        if schedule.entries:
            max_duration = max(entry.duration_hours for entry in schedule.entries)
            long_tasks = [
                (process.get_task_by_id(entry.task_id).name, entry.duration_hours)
                for entry in schedule.entries 
                if entry.duration_hours >= 8  # Consider tasks > 8h as long-running
            ]
            
            for task_name, duration in long_tasks:
                insights.append(f"Long-running task: '{task_name}' takes {duration:.1f} hours. Consider breaking it down.")
        
        # Add some general recommendations if no specific insights
        if not insights:
            insights.extend([
                "Process schedule looks well-optimized.",
                "All tasks have been successfully scheduled.",
                "Resource utilization appears balanced across the team."
            ])
        
        return insights[:5]  # Return up to 5 most important insights
    
    def get_recommendations(self, process: Process) -> Dict[str, Any]:
        """Get optimization recommendations based on mined patterns"""
        if not self.patterns:
            self.mine_patterns()
        
        recommendations = {
            'dependency_suggestions': [],
            'resource_suggestions': [],
            'duration_adjustments': [],
            'bottleneck_warnings': []
        }
        
        # Dependency recommendations
        if 'common_dependencies' in self.patterns:
            for task in process.tasks:
                task_name = task.name.lower()
                for pattern, probability in self.patterns['common_dependencies'].items():
                    if probability > 0.5 and task_name in pattern:
                        recommendations['dependency_suggestions'].append({
                            'task': task.name,
                            'suggested_pattern': pattern,
                            'confidence': probability
                        })
        
        # Resource recommendations
        if 'resource_preferences' in self.patterns:
            for task in process.tasks:
                for skill in task.required_skills:
                    skill_name = skill.name.lower()
                    if skill_name in self.patterns['resource_preferences']:
                        preferences = self.patterns['resource_preferences'][skill_name]
                        best_resource = max(preferences.items(), key=lambda x: x[1])
                        recommendations['resource_suggestions'].append({
                            'task': task.name,
                            'skill': skill_name,
                            'recommended_resource': best_resource[0],
                            'confidence': best_resource[1]
                        })
        
        return recommendations
