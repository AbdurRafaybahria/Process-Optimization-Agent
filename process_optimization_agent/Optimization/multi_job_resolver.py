"""
Multi-Job Task Resolver

Resolves tasks with multiple jobs assigned to maintain 1:1 task-job relationship.
Either finds best-fit single job or splits task into sub-tasks based on skill matching.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ResolutionType(Enum):
    """Type of resolution applied to a multi-job task"""
    BEST_FIT = "best_fit"  # Single job matched all requirements
    SPLIT = "split"  # Task split into sub-tasks
    SINGLE_JOB = "single_job"  # Task already had single job
    NO_MATCH = "no_match"  # No suitable job found


@dataclass
class SkillMatch:
    """Represents skill matching result for a job"""
    job_id: str
    job_name: str
    matched_skills: List[str]
    unmatched_skills: List[str]
    match_percentage: float
    job_data: Dict[str, Any]


@dataclass
class SubTask:
    """Represents a sub-task created from splitting a multi-job task"""
    id: str
    parent_task_id: str
    name: str
    description: str
    duration_minutes: float
    duration_hours: float
    required_skills: List[str]
    assigned_job: Dict[str, Any]
    dependencies: List[str]
    order: int
    is_sub_task: bool = True


@dataclass
class TaskResolution:
    """Result of resolving a multi-job task"""
    original_task_id: str
    original_task_name: str
    resolution_type: ResolutionType
    reason: str
    kept_jobs: List[Dict[str, Any]]
    removed_jobs: List[Dict[str, Any]]
    sub_tasks: List[SubTask]
    skill_analysis: Dict[str, Any]


class MultiJobResolver:
    """
    Resolves tasks with multiple jobs to maintain 1:1 task-job relationship.
    
    Strategy:
    1. Extract required skills from task description OR use real skills from CMS
    2. Match each job's capabilities against required skills
    3. If one job has ≥90% match → Keep that job, remove others
    4. If no best fit → Split task into sub-tasks based on skill groups
    5. Each sub-task gets exactly one job assigned
    """
    
    def __init__(self, best_fit_threshold: float = 0.90, jobs_with_skills: Dict[int, Dict[str, Any]] = None):
        """
        Initialize the resolver.
        
        Args:
            best_fit_threshold: Minimum match percentage to consider a job as "best fit" (default 90%)
            jobs_with_skills: Dictionary mapping job_id to job data with real skills from CMS
        """
        self.best_fit_threshold = best_fit_threshold
        self.jobs_with_skills = jobs_with_skills or {}
        self._init_skill_patterns()
    
    def _init_skill_patterns(self):
        """Initialize keyword patterns for skill extraction"""
        
        # Keywords in task description → Required skills
        self.task_skill_patterns = {
            # Document handling
            'prepare': 'document_preparation',
            'preparing': 'document_preparation',
            'collect': 'document_collection',
            'collects': 'document_collection',
            'gather': 'document_collection',
            'document': 'document_handling',
            'documentation': 'document_handling',
            
            # Form and data entry
            'fill': 'form_filling',
            'fills': 'form_filling',
            'complete': 'form_filling',
            'enter': 'data_entry',
            'input': 'data_entry',
            
            # Submission and upload
            'submit': 'submission',
            'submits': 'submission',
            'submission': 'submission',
            'upload': 'portal_upload',
            'uploads': 'portal_upload',
            'send': 'submission',
            
            # Review and verification
            'review': 'review',
            'reviews': 'review',
            'check': 'verification',
            'verify': 'verification',
            'validate': 'validation',
            'validates': 'validation',
            'validation': 'validation',
            
            # Approval
            'approve': 'approval',
            'approves': 'approval',
            'approval': 'approval',
            'authorize': 'authorization',
            'authorizes': 'authorization',
            
            # Recording and logging
            'record': 'data_recording',
            'records': 'data_recording',
            'log': 'logging',
            'logs': 'logging',
            'store': 'data_storage',
            'save': 'data_storage',
            
            # Notification
            'notify': 'notification',
            'notifies': 'notification',
            'notification': 'notification',
            'alert': 'notification',
            'inform': 'notification',
            
            # Routing and assignment
            'route': 'routing',
            'routes': 'routing',
            'routing': 'routing',
            'assign': 'assignment',
            'forward': 'routing',
            
            # Tracking
            'track': 'tracking',
            'tracking': 'tracking',
            'timestamp': 'tracking',
            
            # Processing
            'process': 'processing',
            'processes': 'processing',
            'processing': 'processing',
            'ingest': 'data_ingestion',
            'ingests': 'data_ingestion',
            
            # Coordination and management
            'coordinate': 'coordination',
            'schedule': 'scheduling',
            'manage': 'management',
            'oversee': 'oversight',
            'oversight': 'oversight',
            'supervise': 'supervision',
            
            # Creation and development
            'create': 'creation',
            'creates': 'creation',
            'develop': 'development',
            'develops': 'development',
            'build': 'development',
            
            # Evaluation
            'evaluate': 'evaluation',
            'assess': 'assessment',
            'grade': 'grading',
            'mark': 'status_update',
            'marks': 'status_update',
        }
        
        # Job name/description keywords → Job capabilities
        self.job_capability_patterns = {
            # Student/Submitter type
            'student': ['submission', 'document_preparation', 'document_collection', 'form_filling', 'portal_upload', 'creation', 'development', 'document_handling'],
            'submitter': ['submission', 'document_preparation', 'form_filling', 'portal_upload'],
            'applicant': ['submission', 'document_preparation', 'form_filling'],
            
            # System/Automated type
            'system': ['data_recording', 'validation', 'notification', 'routing', 'tracking', 'data_ingestion', 'processing', 'data_storage', 'logging'],
            'automated': ['data_recording', 'validation', 'notification', 'routing', 'tracking', 'processing'],
            'bot': ['data_recording', 'validation', 'notification', 'routing', 'processing'],
            
            # Coordinator/Admin type
            'coordinator': ['coordination', 'scheduling', 'management', 'oversight', 'review', 'status_update'],
            'admin': ['review', 'verification', 'status_update', 'data_entry', 'logging'],
            'administrator': ['review', 'verification', 'status_update', 'management'],
            'manager': ['management', 'oversight', 'approval', 'coordination', 'review'],
            
            # Reviewer type
            'reviewer': ['review', 'verification', 'evaluation', 'assessment'],
            'evaluator': ['evaluation', 'assessment', 'grading', 'review'],
            'examiner': ['evaluation', 'assessment', 'grading'],
            
            # Supervisor type
            'supervisor': ['oversight', 'approval', 'management', 'supervision', 'review'],
            'head': ['approval', 'oversight', 'management'],
            'director': ['approval', 'oversight', 'management'],
            
            # Technical roles
            'developer': ['development', 'creation', 'processing'],
            'analyst': ['analysis', 'review', 'verification'],
            'specialist': ['processing', 'verification', 'review'],
        }
        
        # Skill categories for grouping during task splitting
        self.skill_categories = {
            'manual_submission': ['submission', 'document_preparation', 'document_collection', 'form_filling', 'portal_upload', 'document_handling'],
            'automated_processing': ['data_recording', 'validation', 'notification', 'routing', 'tracking', 'data_ingestion', 'processing', 'data_storage', 'logging'],
            'review_approval': ['review', 'verification', 'approval', 'authorization', 'evaluation', 'assessment', 'status_update'],
            'coordination': ['coordination', 'scheduling', 'management', 'oversight', 'supervision'],
            'creation': ['creation', 'development', 'document_handling'],
        }
    
    def resolve_process(self, process_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve all multi-job tasks in a process.
        
        Args:
            process_data: CMS process data with process_task array
            
        Returns:
            Modified process data with resolved tasks
        """
        process_tasks = process_data.get('process_task', [])
        resolved_tasks = []
        resolutions = []
        
        # Track total costs for summary
        total_before_cost = 0.0
        total_after_cost = 0.0
        
        for pt in process_tasks:
            task = pt.get('task', {})
            job_tasks = task.get('jobTasks', [])
            task_duration_minutes = task.get('task_capacity_minutes', 0) or 60
            task_duration_hours = task_duration_minutes / 60
            
            if len(job_tasks) <= 1:
                # Single job or no job - keep as-is
                resolved_tasks.append(pt)
                if len(job_tasks) == 1:
                    job = job_tasks[0].get('job', {})
                    hourly_rate = job.get('hourlyRate', 0) or 0
                    task_cost = hourly_rate * task_duration_hours
                    total_before_cost += task_cost
                    total_after_cost += task_cost
                    
                    resolutions.append({
                        'task_id': task.get('task_id'),
                        'task_name': task.get('task_name'),
                        'resolution': 'single_job',
                        'reason': 'Task already has single job assigned',
                        'cost_analysis': {
                            'before_cost': round(task_cost, 2),
                            'after_cost': round(task_cost, 2),
                            'savings': 0,
                            'task_duration_hours': round(task_duration_hours, 4)
                        }
                    })
            else:
                # Multiple jobs - resolve
                resolution = self.resolve_task(task, pt.get('order', 1))
                
                if resolution.resolution_type == ResolutionType.BEST_FIT:
                    # Keep task with single best-fit job
                    modified_pt = self._create_single_job_task(pt, resolution)
                    resolved_tasks.append(modified_pt)
                elif resolution.resolution_type == ResolutionType.SPLIT:
                    # Replace with sub-tasks
                    sub_task_pts = self._create_sub_task_entries(pt, resolution)
                    resolved_tasks.extend(sub_task_pts)
                else:
                    # No match - keep original with warning
                    resolved_tasks.append(pt)
                
                # Calculate BEFORE cost: sum of ALL assigned jobs' hourly rates × task duration
                # (Assumption: all jobs work in parallel on the task)
                before_cost = 0.0
                for jt in job_tasks:
                    job = jt.get('job', {})
                    hourly_rate = job.get('hourlyRate', 0) or 0
                    before_cost += hourly_rate * task_duration_hours
                
                # Calculate AFTER cost: single kept job's hourly rate × task duration
                after_cost = 0.0
                if resolution.kept_jobs:
                    # For BEST_FIT: single job
                    # For SPLIT: sum of kept jobs for sub-tasks (duration is split proportionally)
                    if resolution.resolution_type == ResolutionType.SPLIT and resolution.sub_tasks:
                        # Sub-tasks get their own durations
                        for st in resolution.sub_tasks:
                            st_hourly = st.assigned_job.get('hourlyRate', 0) or 0
                            after_cost += st_hourly * st.duration_hours
                    else:
                        # BEST_FIT - single job
                        kept_job = resolution.kept_jobs[0]
                        kept_rate = kept_job.get('hourlyRate', 0) or 0
                        after_cost = kept_rate * task_duration_hours
                
                task_savings = before_cost - after_cost
                total_before_cost += before_cost
                total_after_cost += after_cost
                
                resolutions.append({
                    'task_id': str(task.get('task_id')),
                    'task_name': task.get('task_name'),
                    'resolution': resolution.resolution_type.value,
                    'reason': resolution.reason,
                    'kept_jobs': [{'job_id': j.get('job_id'), 'name': j.get('name'), 'hourlyRate': j.get('hourlyRate', 0)} for j in resolution.kept_jobs],
                    'removed_jobs': [{'job_id': j.get('job_id'), 'name': j.get('name'), 'hourlyRate': j.get('hourlyRate', 0), 'reason': 'Low skill match'} for j in resolution.removed_jobs],
                    'sub_tasks': [{'id': st.id, 'name': st.name, 'job': st.assigned_job.get('name'), 'hourlyRate': st.assigned_job.get('hourlyRate', 0), 'duration_hours': st.duration_hours} for st in resolution.sub_tasks] if resolution.sub_tasks else [],
                    'skill_analysis': resolution.skill_analysis,
                    'cost_analysis': {
                        'before_cost': round(before_cost, 2),
                        'after_cost': round(after_cost, 2),
                        'savings': round(task_savings, 2),
                        'task_duration_hours': round(task_duration_hours, 4),
                        'jobs_before': len(job_tasks),
                        'jobs_after': len(resolution.kept_jobs) if resolution.resolution_type != ResolutionType.SPLIT else len(resolution.sub_tasks)
                    }
                })
        
        # Calculate totals
        total_savings = total_before_cost - total_after_cost
        savings_percentage = (total_savings / total_before_cost * 100) if total_before_cost > 0 else 0
        
        # Update process data
        result = process_data.copy()
        result['process_task'] = resolved_tasks
        result['_multi_job_resolutions'] = resolutions
        result['_job_resolution_cost_summary'] = {
            'total_before_cost': round(total_before_cost, 2),
            'total_after_cost': round(total_after_cost, 2),
            'total_savings': round(total_savings, 2),
            'savings_percentage': round(savings_percentage, 2)
        }
        
        print(f"[JOB-RESOLUTION] Cost Analysis: Before=${total_before_cost:.2f}, After=${total_after_cost:.2f}, Savings=${total_savings:.2f} ({savings_percentage:.1f}%)")
        
        return result
    
    def resolve_task(self, task: Dict[str, Any], order: int = 1) -> TaskResolution:
        """
        Resolve a single task with multiple jobs.
        
        Args:
            task: Task data with jobTasks array
            order: Task order in process
            
        Returns:
            TaskResolution with resolution details
        """
        task_id = str(task.get('task_id', ''))
        task_name = task.get('task_name', '')
        task_description = task.get('task_overview', '') or task.get('description', '')
        task_duration = task.get('task_capacity_minutes', 0) or task.get('duration', 0)
        job_tasks = task.get('jobTasks', [])
        
        # Step 1: Extract required skills from task
        required_skills = self._extract_required_skills(task_name, task_description)
        
        # Step 2: Analyze each job's capabilities and match
        skill_matches = []
        for jt in job_tasks:
            job = jt.get('job', {})
            match = self._calculate_skill_match(job, required_skills)
            skill_matches.append(match)
        
        # Sort by match percentage (highest first)
        skill_matches.sort(key=lambda m: m.match_percentage, reverse=True)
        
        # Step 3: Check for best fit
        best_match = skill_matches[0] if skill_matches else None
        
        skill_analysis = {
            'required_skills': required_skills,
            'job_matches': [
                {
                    'job_id': m.job_id,
                    'job_name': m.job_name,
                    'match_percentage': round(m.match_percentage * 100, 1),
                    'matched_skills': m.matched_skills,
                    'unmatched_skills': m.unmatched_skills
                }
                for m in skill_matches
            ]
        }
        
        if best_match and best_match.match_percentage >= self.best_fit_threshold:
            # Best fit found - keep this job only
            kept_jobs = [best_match.job_data]
            removed_jobs = [m.job_data for m in skill_matches[1:]]
            
            return TaskResolution(
                original_task_id=task_id,
                original_task_name=task_name,
                resolution_type=ResolutionType.BEST_FIT,
                reason=f"Job '{best_match.job_name}' has {round(best_match.match_percentage * 100)}% skill match (≥{round(self.best_fit_threshold * 100)}% threshold)",
                kept_jobs=kept_jobs,
                removed_jobs=removed_jobs,
                sub_tasks=[],
                skill_analysis=skill_analysis
            )
        
        # Step 4: No best fit - split task into sub-tasks
        sub_tasks, kept_jobs, removed_jobs = self._split_task_into_subtasks(
            task_id, task_name, task_description, task_duration, 
            required_skills, skill_matches, order
        )
        
        if sub_tasks:
            return TaskResolution(
                original_task_id=task_id,
                original_task_name=task_name,
                resolution_type=ResolutionType.SPLIT,
                reason=f"No single job had ≥{round(self.best_fit_threshold * 100)}% skill match. Task split into {len(sub_tasks)} sub-tasks.",
                kept_jobs=kept_jobs,
                removed_jobs=removed_jobs,
                sub_tasks=sub_tasks,
                skill_analysis=skill_analysis
            )
        
        # Fallback: keep job with highest match
        if best_match:
            return TaskResolution(
                original_task_id=task_id,
                original_task_name=task_name,
                resolution_type=ResolutionType.BEST_FIT,
                reason=f"Fallback: Keeping job '{best_match.job_name}' with highest match ({round(best_match.match_percentage * 100)}%)",
                kept_jobs=[best_match.job_data],
                removed_jobs=[m.job_data for m in skill_matches[1:]],
                sub_tasks=[],
                skill_analysis=skill_analysis
            )
        
        return TaskResolution(
            original_task_id=task_id,
            original_task_name=task_name,
            resolution_type=ResolutionType.NO_MATCH,
            reason="No jobs found or no skill matches",
            kept_jobs=[],
            removed_jobs=[m.job_data for m in skill_matches],
            sub_tasks=[],
            skill_analysis=skill_analysis
        )
    
    def _extract_required_skills(self, task_name: str, task_description: str) -> List[str]:
        """Extract required skills from task name and description"""
        text = f"{task_name} {task_description}".lower()
        
        # Clean HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        skills = set()
        
        # Extract skills based on keyword patterns
        for keyword, skill in self.task_skill_patterns.items():
            # Use word boundary matching
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text):
                skills.add(skill)
        
        # Additional FYP/Academic specific skill extraction
        academic_patterns = {
            'research': 'research_skills',
            'thesis': 'technical_writing',
            'proposal': 'technical_writing',
            'draft': 'technical_writing',
            'idea': 'research_skills',
            'formulate': 'research_skills',
            'gap': 'research_skills',
            'presentation': 'presentation_skills',
            'demo': 'presentation_skills',
            'review': 'review_evaluation',
            'evaluate': 'review_evaluation',
            'supervisor': 'mentoring',
            'consultation': 'communication',
            'coordinate': 'management',
            'aggregate': 'data_management',
            'score': 'evaluation',
            'lab': 'resource_allocation',
            'environment': 'resource_allocation',
            'booking': 'resource_allocation',
        }
        
        for keyword, skill in academic_patterns.items():
            if keyword in text:
                skills.add(skill)
        
        return list(skills)
    
    def _get_job_capabilities(self, job: Dict[str, Any]) -> List[str]:
        """Get capabilities of a job based on its real skills from CMS or fallback to text patterns"""
        job_id = job.get('job_id')
        
        # First, try to get real skills from CMS data
        if job_id and int(job_id) in self.jobs_with_skills:
            cms_job = self.jobs_with_skills[int(job_id)]
            real_skills = cms_job.get('skills', [])
            if real_skills:
                # Return actual skill names from CMS
                capabilities = []
                for skill in real_skills:
                    skill_name = skill.get('name', '').lower().strip()
                    if skill_name:
                        capabilities.append(skill_name)
                        # Also add normalized versions
                        capabilities.append(skill_name.replace(' ', '_'))
                if capabilities:
                    print(f"[DEBUG] Job {job.get('name')} using CMS skills: {capabilities}")
                    return list(set(capabilities))
        
        # Fallback: Extract from job name and description using patterns
        job_name = (job.get('name', '') or '').lower()
        job_description = (job.get('description', '') or '').lower()
        
        # Clean HTML tags
        job_description = re.sub(r'<[^>]+>', ' ', job_description)
        
        capabilities = set()
        
        # Match job name against capability patterns
        for job_type, caps in self.job_capability_patterns.items():
            if job_type in job_name:
                capabilities.update(caps)
        
        # Also extract from description using task skill patterns
        for keyword, skill in self.task_skill_patterns.items():
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, job_description):
                capabilities.add(skill)
        
        return list(capabilities)
    
    def _normalize_skill(self, skill: str) -> str:
        """Normalize a skill name for matching"""
        return skill.lower().strip().replace('_', ' ').replace('-', ' ')
    
    def _skill_similarity(self, skill1: str, skill2: str) -> float:
        """
        Calculate similarity between two skills using keyword matching.
        Returns a score between 0 and 1.
        """
        s1 = self._normalize_skill(skill1)
        s2 = self._normalize_skill(skill2)
        
        # Exact match
        if s1 == s2:
            return 1.0
        
        # Check if one contains the other
        if s1 in s2 or s2 in s1:
            return 0.8
        
        # Word overlap matching
        words1 = set(s1.split())
        words2 = set(s2.split())
        
        if words1 & words2:  # Any common words
            overlap = len(words1 & words2)
            total = len(words1 | words2)
            return overlap / total * 0.7  # Max 0.7 for partial matches
        
        # Keyword-based semantic matching
        skill_synonyms = {
            'research': ['research', 'analysis', 'investigation', 'study', 'formulate'],
            'writing': ['writing', 'documentation', 'document', 'technical', 'thesis', 'draft'],
            'review': ['review', 'verification', 'check', 'evaluate', 'assessment'],
            'management': ['management', 'coordination', 'process', 'administrative', 'organization'],
            'presentation': ['presentation', 'demo', 'demonstrate', 'present'],
            'evaluation': ['evaluation', 'assessment', 'grading', 'judgment', 'auditing'],
            'mentoring': ['mentoring', 'guidance', 'supervision', 'oversight'],
            'development': ['development', 'software', 'creation', 'build', 'produce'],
            'communication': ['communication', 'feedback', 'consultation'],
            'planning': ['planning', 'strategic', 'scheduling'],
            'leadership': ['leadership', 'decision', 'head', 'director'],
            'allocation': ['allocation', 'resource', 'booking', 'lab', 'environment'],
        }
        
        for category, keywords in skill_synonyms.items():
            s1_match = any(kw in s1 for kw in keywords)
            s2_match = any(kw in s2 for kw in keywords)
            if s1_match and s2_match:
                return 0.6  # Semantic similarity through category
        
        return 0.0
    
    def _calculate_skill_match(self, job: Dict[str, Any], required_skills: List[str]) -> SkillMatch:
        """Calculate how well a job matches the required skills using fuzzy matching"""
        job_capabilities = self._get_job_capabilities(job)
        
        matched = []
        unmatched = []
        
        for req_skill in required_skills:
            # Find best matching capability
            best_similarity = 0.0
            for cap in job_capabilities:
                sim = self._skill_similarity(req_skill, cap)
                best_similarity = max(best_similarity, sim)
            
            if best_similarity >= 0.5:  # Threshold for considering a match
                matched.append(req_skill)
            else:
                unmatched.append(req_skill)
        
        match_pct = len(matched) / len(required_skills) if required_skills else 0.0
        
        return SkillMatch(
            job_id=str(job.get('job_id', '')),
            job_name=job.get('name', ''),
            matched_skills=matched,
            unmatched_skills=unmatched,
            match_percentage=match_pct,
            job_data=job
        )
    
    def _split_task_into_subtasks(
        self,
        task_id: str,
        task_name: str,
        task_description: str,
        task_duration: float,
        required_skills: List[str],
        skill_matches: List[SkillMatch],
        order: int
    ) -> Tuple[List[SubTask], List[Dict], List[Dict]]:
        """
        Split a task into sub-tasks based on skill categories.
        
        Returns:
            Tuple of (sub_tasks, kept_jobs, removed_jobs)
        """
        # Group required skills by category
        skill_groups = {}
        for skill in required_skills:
            category = self._get_skill_category(skill)
            if category not in skill_groups:
                skill_groups[category] = []
            skill_groups[category].append(skill)
        
        # If no clear grouping, try to split by matched skills per job
        if len(skill_groups) <= 1:
            skill_groups = self._group_skills_by_job_match(required_skills, skill_matches)
        
        sub_tasks = []
        kept_jobs = []
        used_job_ids = set()
        sub_order = 1
        
        # Calculate duration per skill group (proportional)
        total_skills = len(required_skills) if required_skills else 1
        
        previous_subtask_id = None
        
        for category, skills in skill_groups.items():
            if not skills:
                continue
            
            # Find best job for this skill group
            best_job = self._find_best_job_for_skills(skills, skill_matches, used_job_ids)
            
            if not best_job:
                continue
            
            used_job_ids.add(best_job.job_id)
            
            # Calculate proportional duration
            proportion = len(skills) / total_skills
            sub_duration = max(1, round(task_duration * proportion))  # At least 1 minute
            
            # Create sub-task
            sub_task_id = f"{task_id}_{sub_order}"
            sub_task_name = f"{task_name} - {self._format_category_name(category)}"
            sub_task_desc = self._generate_subtask_description(skills, task_description)
            
            # Set dependencies (sequential by default)
            dependencies = [previous_subtask_id] if previous_subtask_id else []
            
            sub_task = SubTask(
                id=sub_task_id,
                parent_task_id=task_id,
                name=sub_task_name,
                description=sub_task_desc,
                duration_minutes=sub_duration,
                duration_hours=sub_duration / 60,
                required_skills=skills,
                assigned_job={
                    'job_id': best_job.job_id,
                    'name': best_job.job_name,
                    'hourlyRate': best_job.job_data.get('hourlyRate', 0),
                    'job_data': best_job.job_data
                },
                dependencies=dependencies,
                order=order * 100 + sub_order,  # Sub-order within parent order
                is_sub_task=True
            )
            
            sub_tasks.append(sub_task)
            kept_jobs.append(best_job.job_data)
            previous_subtask_id = sub_task_id
            sub_order += 1
        
        # Determine removed jobs
        kept_job_ids = {j.get('job_id') for j in kept_jobs}
        removed_jobs = [m.job_data for m in skill_matches if m.job_id not in kept_job_ids]
        
        return sub_tasks, kept_jobs, removed_jobs
    
    def _get_skill_category(self, skill: str) -> str:
        """Get the category a skill belongs to"""
        for category, skills in self.skill_categories.items():
            if skill in skills:
                return category
        return 'other'
    
    def _group_skills_by_job_match(
        self, 
        required_skills: List[str], 
        skill_matches: List[SkillMatch]
    ) -> Dict[str, List[str]]:
        """Group skills based on which job best matches them"""
        groups = {}
        
        for skill in required_skills:
            # Find which job has this skill
            best_job_for_skill = None
            for match in skill_matches:
                if skill in match.matched_skills:
                    best_job_for_skill = match.job_name
                    break
            
            if best_job_for_skill:
                if best_job_for_skill not in groups:
                    groups[best_job_for_skill] = []
                groups[best_job_for_skill].append(skill)
            else:
                # Unmatched skills go to 'other'
                if 'other' not in groups:
                    groups['other'] = []
                groups['other'].append(skill)
        
        return groups
    
    def _find_best_job_for_skills(
        self, 
        skills: List[str], 
        skill_matches: List[SkillMatch],
        used_job_ids: set
    ) -> Optional[SkillMatch]:
        """Find the best job for a set of skills, excluding already used jobs"""
        best_match = None
        best_count = 0
        
        for match in skill_matches:
            if match.job_id in used_job_ids:
                continue
            
            # Count how many of the target skills this job has
            matched_count = sum(1 for s in skills if s in match.matched_skills)
            
            if matched_count > best_count:
                best_count = matched_count
                best_match = match
        
        return best_match
    
    def _format_category_name(self, category: str) -> str:
        """Format category name for display"""
        names = {
            'manual_submission': 'Manual Submission',
            'automated_processing': 'System Processing',
            'review_approval': 'Review & Approval',
            'coordination': 'Coordination',
            'creation': 'Creation',
            'other': 'Processing',
        }
        return names.get(category, category.replace('_', ' ').title())
    
    def _generate_subtask_description(self, skills: List[str], original_description: str) -> str:
        """Generate a description for a sub-task based on its skills"""
        skill_descriptions = {
            'submission': 'Submit required materials',
            'document_preparation': 'Prepare necessary documents',
            'document_collection': 'Collect required documents',
            'form_filling': 'Fill out required forms',
            'portal_upload': 'Upload to portal',
            'data_recording': 'Record data in system',
            'validation': 'Validate submitted information',
            'notification': 'Send notifications',
            'routing': 'Route to appropriate parties',
            'tracking': 'Generate tracking information',
            'review': 'Review submitted materials',
            'verification': 'Verify information',
            'approval': 'Provide approval',
            'status_update': 'Update status',
            'coordination': 'Coordinate activities',
            'processing': 'Process submitted data',
            'data_ingestion': 'Ingest submitted data',
        }
        
        parts = [skill_descriptions.get(s, s.replace('_', ' ')) for s in skills[:3]]
        return '. '.join(parts) + '.'
    
    def _create_single_job_task(self, process_task: Dict, resolution: TaskResolution) -> Dict:
        """Create a modified process_task with single job"""
        result = process_task.copy()
        task = result.get('task', {}).copy()
        
        # Keep only the best-fit job
        if resolution.kept_jobs:
            kept_job = resolution.kept_jobs[0]
            task['jobTasks'] = [{
                'job_id': kept_job.get('job_id'),
                'task_id': task.get('task_id'),
                'job': kept_job
            }]
        
        task['_resolution'] = {
            'type': resolution.resolution_type.value,
            'reason': resolution.reason,
            'removed_jobs': [{'job_id': j.get('job_id'), 'name': j.get('name')} for j in resolution.removed_jobs]
        }
        
        result['task'] = task
        return result
    
    def _create_sub_task_entries(self, process_task: Dict, resolution: TaskResolution) -> List[Dict]:
        """Create process_task entries for sub-tasks"""
        entries = []
        original_task = process_task.get('task', {})
        
        for sub_task in resolution.sub_tasks:
            entry = {
                'process_task_id': f"{process_task.get('process_task_id')}_{sub_task.id.split('_')[-1]}",
                'process_id': process_task.get('process_id'),
                'task_id': sub_task.id,
                'order': sub_task.order,
                'child_process_id': None,
                'task': {
                    'task_id': sub_task.id,
                    'parent_task_id': sub_task.parent_task_id,
                    'is_sub_task': True,
                    'created_at': original_task.get('created_at'),
                    'updated_at': original_task.get('updated_at'),
                    'task_capacity_minutes': sub_task.duration_minutes,
                    'task_code': f"{original_task.get('task_code', 'T')}-{sub_task.id.split('_')[-1]}",
                    'task_company_id': original_task.get('task_company_id'),
                    'task_name': sub_task.name,
                    'task_overview': sub_task.description,
                    'task_process_id': None,
                    'dependencies': sub_task.dependencies,
                    'required_skills': sub_task.required_skills,
                    'jobTasks': [{
                        'job_id': sub_task.assigned_job.get('job_id'),
                        'task_id': sub_task.id,
                        'job': sub_task.assigned_job.get('job_data', sub_task.assigned_job)
                    }],
                    '_resolution': {
                        'type': 'sub_task',
                        'parent_task': resolution.original_task_name,
                        'assigned_job': sub_task.assigned_job.get('name')
                    }
                }
            }
            entries.append(entry)
        
        return entries


# Convenience function
def resolve_multi_job_tasks(process_data: Dict[str, Any], threshold: float = 0.90) -> Dict[str, Any]:
    """
    Resolve all multi-job tasks in a process.
    
    Args:
        process_data: CMS process data
        threshold: Best-fit threshold (default 90%)
        
    Returns:
        Modified process data with resolved tasks
    """
    resolver = MultiJobResolver(best_fit_threshold=threshold)
    return resolver.resolve_process(process_data)


# =============================================================================
# COST OPTIMIZER
# =============================================================================

@dataclass
class JobReplacement:
    """Represents a job replacement for cost optimization"""
    task_id: str
    task_name: str
    original_job_id: int
    original_job_name: str
    original_hourly_rate: float
    new_job_id: int
    new_job_name: str
    new_hourly_rate: float
    task_duration_hours: float
    cost_savings_per_task: float
    reason: str


@dataclass
class CostOptimizationResult:
    """Result of cost optimization"""
    original_total_cost: float
    optimized_total_cost: float
    total_savings: float
    savings_percentage: float
    replacements: List[JobReplacement]
    tasks_analyzed: int
    tasks_optimized: int


class CostOptimizer:
    """
    Optimizes process cost by finding cheaper qualified jobs for each task.
    
    Strategy:
    1. For each task, get the currently assigned job and its hourly rate
    2. Get the task's required skills
    3. Find ALL jobs from CMS that have those required skills (≥90% match)
    4. Check skill levels (candidate skill level ≥ required level)
    5. If a qualified job has lower hourly rate → Replace
    6. Report total savings
    """
    
    def __init__(self, all_jobs_map: Dict[int, Dict[str, Any]], skill_match_threshold: float = 0.90):
        """
        Initialize the cost optimizer.
        
        Args:
            all_jobs_map: Dictionary of ALL jobs from CMS (job_id -> job_data with skills)
            skill_match_threshold: Minimum skill match percentage (default 90%)
        """
        self.all_jobs_map = all_jobs_map
        self.skill_match_threshold = skill_match_threshold
    
    def optimize_process(self, process_data: Dict[str, Any]) -> Tuple[Dict[str, Any], CostOptimizationResult]:
        """
        Optimize job assignments in a process for minimum cost.
        
        Args:
            process_data: CMS process data (after multi-job resolution)
            
        Returns:
            Tuple of (modified_process_data, CostOptimizationResult)
        """
        process_tasks = process_data.get('process_task', [])
        replacements = []
        original_total_cost = 0.0
        optimized_total_cost = 0.0
        tasks_analyzed = 0
        tasks_optimized = 0
        
        for pt in process_tasks:
            task = pt.get('task', {})
            task_id = str(task.get('task_id', ''))
            task_name = task.get('task_name', '')
            task_duration_minutes = task.get('task_capacity_minutes', 0) or task.get('duration', 60)
            task_duration_hours = task_duration_minutes / 60
            
            job_tasks = task.get('jobTasks', [])
            if not job_tasks:
                continue
            
            tasks_analyzed += 1
            
            # Get currently assigned job
            current_jt = job_tasks[0]
            current_job = current_jt.get('job', {})
            current_job_id = current_job.get('job_id')
            current_job_name = current_job.get('name', '')
            current_hourly_rate = current_job.get('hourlyRate', 0) or 0
            
            # Calculate original cost for this task
            task_original_cost = current_hourly_rate * task_duration_hours
            original_total_cost += task_original_cost
            
            # Extract required skills from task
            required_skills = self._extract_task_required_skills(task)
            
            if not required_skills:
                # No skills to match, keep current job
                optimized_total_cost += task_original_cost
                continue
            
            # Find cheapest qualified job
            cheapest_job = self._find_cheapest_qualified_job(
                required_skills, 
                current_job_id, 
                current_hourly_rate
            )
            
            if cheapest_job:
                # Replace with cheaper job
                new_job_id = cheapest_job['job_id']
                new_job_name = cheapest_job['name']
                new_hourly_rate = cheapest_job['hourlyRate']
                task_new_cost = new_hourly_rate * task_duration_hours
                savings = task_original_cost - task_new_cost
                
                # Update the job assignment in process_data
                self._replace_job_in_task(pt, cheapest_job)
                
                replacement = JobReplacement(
                    task_id=task_id,
                    task_name=task_name,
                    original_job_id=current_job_id,
                    original_job_name=current_job_name,
                    original_hourly_rate=current_hourly_rate,
                    new_job_id=new_job_id,
                    new_job_name=new_job_name,
                    new_hourly_rate=new_hourly_rate,
                    task_duration_hours=task_duration_hours,
                    cost_savings_per_task=savings,
                    reason=f"Found cheaper qualified job: ${new_hourly_rate}/hr vs ${current_hourly_rate}/hr"
                )
                replacements.append(replacement)
                optimized_total_cost += task_new_cost
                tasks_optimized += 1
                
                print(f"[COST-OPT] Task '{task_name}': Replaced '{current_job_name}' (${current_hourly_rate}/hr) "
                      f"with '{new_job_name}' (${new_hourly_rate}/hr) - Saving ${savings:.2f}")
            else:
                # Keep current job
                optimized_total_cost += task_original_cost
        
        total_savings = original_total_cost - optimized_total_cost
        savings_percentage = (total_savings / original_total_cost * 100) if original_total_cost > 0 else 0
        
        result = CostOptimizationResult(
            original_total_cost=round(original_total_cost, 2),
            optimized_total_cost=round(optimized_total_cost, 2),
            total_savings=round(total_savings, 2),
            savings_percentage=round(savings_percentage, 2),
            replacements=replacements,
            tasks_analyzed=tasks_analyzed,
            tasks_optimized=tasks_optimized
        )
        
        # Store optimization results in process_data for later retrieval
        process_data['_cost_optimization'] = {
            'original_total_cost': result.original_total_cost,
            'optimized_total_cost': result.optimized_total_cost,
            'total_savings': result.total_savings,
            'savings_percentage': result.savings_percentage,
            'tasks_analyzed': result.tasks_analyzed,
            'tasks_optimized': result.tasks_optimized,
            'replacements': [
                {
                    'task_id': r.task_id,
                    'task_name': r.task_name,
                    'original_job': {
                        'job_id': r.original_job_id,
                        'name': r.original_job_name,
                        'hourlyRate': r.original_hourly_rate
                    },
                    'new_job': {
                        'job_id': r.new_job_id,
                        'name': r.new_job_name,
                        'hourlyRate': r.new_hourly_rate
                    },
                    'task_duration_hours': r.task_duration_hours,
                    'cost_savings': r.cost_savings_per_task,
                    'reason': r.reason
                }
                for r in replacements
            ]
        }
        
        print(f"[COST-OPT] Summary: Analyzed {tasks_analyzed} tasks, optimized {tasks_optimized}")
        print(f"[COST-OPT] Original cost: ${original_total_cost:.2f}, Optimized: ${optimized_total_cost:.2f}")
        print(f"[COST-OPT] Total savings: ${total_savings:.2f} ({savings_percentage:.1f}%)")
        
        return process_data, result
    
    def _extract_task_required_skills(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract required skills from task's assigned job"""
        job_tasks = task.get('jobTasks', [])
        if not job_tasks:
            return []
        
        # Get skills from the currently assigned job
        current_job = job_tasks[0].get('job', {})
        current_job_id = current_job.get('job_id')
        
        # Try to get skills from all_jobs_map (more complete)
        if current_job_id and int(current_job_id) in self.all_jobs_map:
            job_from_map = self.all_jobs_map[int(current_job_id)]
            return job_from_map.get('skills', [])
        
        # Fallback: check if job has jobSkills in the task data
        return current_job.get('jobSkills', [])
    
    def _find_cheapest_qualified_job(
        self, 
        required_skills: List[Dict[str, Any]], 
        current_job_id: int,
        current_hourly_rate: float
    ) -> Optional[Dict[str, Any]]:
        """
        Find the cheapest job that meets the skill requirements.
        
        Args:
            required_skills: List of required skill dicts
            current_job_id: ID of currently assigned job (to exclude)
            current_hourly_rate: Current job's hourly rate (must be cheaper)
            
        Returns:
            Cheapest qualified job dict, or None if no cheaper option
        """
        cheapest_job = None
        cheapest_rate = current_hourly_rate  # Must be cheaper than current
        
        required_skill_names = set()
        required_skill_levels = {}
        
        for skill in required_skills:
            skill_name = (skill.get('name', '') or '').lower().strip()
            if skill_name:
                required_skill_names.add(skill_name)
                # Store required level (default to 3 = INTERMEDIATE)
                level_rank = skill.get('level_rank', 3) or 3
                required_skill_levels[skill_name] = level_rank
        
        if not required_skill_names:
            return None
        
        for job_id, job_data in self.all_jobs_map.items():
            # Skip current job
            if job_id == current_job_id:
                continue
            
            job_hourly_rate = job_data.get('hourlyRate', 0) or 0
            
            # Skip if not cheaper
            if job_hourly_rate >= cheapest_rate:
                continue
            
            # Check if job has required skills with adequate levels
            job_skills = job_data.get('skills', [])
            job_skill_map = {}
            for js in job_skills:
                js_name = (js.get('name', '') or '').lower().strip()
                js_level = js.get('level_rank', 3) or 3
                if js_name:
                    job_skill_map[js_name] = js_level
            
            # Calculate skill match
            matched_skills = 0
            level_ok = True
            
            for req_skill in required_skill_names:
                if req_skill in job_skill_map:
                    matched_skills += 1
                    # Check level requirement
                    if job_skill_map[req_skill] < required_skill_levels.get(req_skill, 3):
                        level_ok = False
                        break
                else:
                    # Try fuzzy match
                    fuzzy_matched = False
                    for job_skill_name, job_skill_level in job_skill_map.items():
                        if self._fuzzy_skill_match(req_skill, job_skill_name):
                            matched_skills += 1
                            if job_skill_level < required_skill_levels.get(req_skill, 3):
                                level_ok = False
                            fuzzy_matched = True
                            break
                    if not fuzzy_matched:
                        pass  # Skill not matched
            
            if not level_ok:
                continue
            
            # Check if match percentage meets threshold
            match_percentage = matched_skills / len(required_skill_names) if required_skill_names else 0
            
            if match_percentage >= self.skill_match_threshold:
                # This job qualifies and is cheaper
                cheapest_job = job_data
                cheapest_rate = job_hourly_rate
        
        return cheapest_job
    
    def _fuzzy_skill_match(self, skill1: str, skill2: str) -> bool:
        """Check if two skill names are similar enough to match"""
        s1 = skill1.lower().strip().replace('_', ' ').replace('-', ' ')
        s2 = skill2.lower().strip().replace('_', ' ').replace('-', ' ')
        
        # Exact match
        if s1 == s2:
            return True
        
        # Containment
        if s1 in s2 or s2 in s1:
            return True
        
        # Word overlap
        words1 = set(s1.split())
        words2 = set(s2.split())
        if words1 & words2:
            overlap = len(words1 & words2)
            total = max(len(words1), len(words2))
            if overlap / total >= 0.5:
                return True
        
        return False
    
    def _replace_job_in_task(self, process_task: Dict[str, Any], new_job: Dict[str, Any]) -> None:
        """Replace the job assignment in a process_task"""
        task = process_task.get('task', {})
        job_tasks = task.get('jobTasks', [])
        
        if job_tasks:
            # Store original job info for reference
            original_job = job_tasks[0].get('job', {})
            
            # Replace with new job
            job_tasks[0]['job'] = new_job
            job_tasks[0]['job_id'] = new_job.get('job_id')
            
            # Add cost optimization metadata
            task['_cost_optimized'] = {
                'original_job_id': original_job.get('job_id'),
                'original_job_name': original_job.get('name'),
                'original_hourly_rate': original_job.get('hourlyRate'),
                'new_job_id': new_job.get('job_id'),
                'new_job_name': new_job.get('name'),
                'new_hourly_rate': new_job.get('hourlyRate')
            }


def optimize_process_cost(
    process_data: Dict[str, Any], 
    all_jobs_map: Dict[int, Dict[str, Any]],
    skill_match_threshold: float = 0.90
) -> Tuple[Dict[str, Any], CostOptimizationResult]:
    """
    Convenience function to optimize process cost.
    
    Args:
        process_data: CMS process data (after multi-job resolution)
        all_jobs_map: All jobs from CMS
        skill_match_threshold: Minimum skill match (default 90%)
        
    Returns:
        Tuple of (modified_process_data, CostOptimizationResult)
    """
    optimizer = CostOptimizer(all_jobs_map, skill_match_threshold)
    return optimizer.optimize_process(process_data)

