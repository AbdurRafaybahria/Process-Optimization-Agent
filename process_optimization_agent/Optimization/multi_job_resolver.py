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
    1. Extract required skills from task description
    2. Match each job's capabilities against required skills
    3. If one job has ≥90% match → Keep that job, remove others
    4. If no best fit → Split task into sub-tasks based on skill groups
    5. Each sub-task gets exactly one job assigned
    """
    
    def __init__(self, best_fit_threshold: float = 0.90):
        """
        Initialize the resolver.
        
        Args:
            best_fit_threshold: Minimum match percentage to consider a job as "best fit" (default 90%)
        """
        self.best_fit_threshold = best_fit_threshold
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
        
        for pt in process_tasks:
            task = pt.get('task', {})
            job_tasks = task.get('jobTasks', [])
            
            if len(job_tasks) <= 1:
                # Single job or no job - keep as-is
                resolved_tasks.append(pt)
                if len(job_tasks) == 1:
                    resolutions.append({
                        'task_id': task.get('task_id'),
                        'task_name': task.get('task_name'),
                        'resolution': 'single_job',
                        'reason': 'Task already has single job assigned'
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
                
                resolutions.append({
                    'task_id': str(task.get('task_id')),
                    'task_name': task.get('task_name'),
                    'resolution': resolution.resolution_type.value,
                    'reason': resolution.reason,
                    'kept_jobs': [{'job_id': j.get('job_id'), 'name': j.get('name')} for j in resolution.kept_jobs],
                    'removed_jobs': [{'job_id': j.get('job_id'), 'name': j.get('name'), 'reason': 'Low skill match'} for j in resolution.removed_jobs],
                    'sub_tasks': [{'id': st.id, 'name': st.name, 'job': st.assigned_job.get('name')} for st in resolution.sub_tasks] if resolution.sub_tasks else [],
                    'skill_analysis': resolution.skill_analysis
                })
        
        # Update process data
        result = process_data.copy()
        result['process_task'] = resolved_tasks
        result['_multi_job_resolutions'] = resolutions
        
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
        
        return list(skills)
    
    def _get_job_capabilities(self, job: Dict[str, Any]) -> List[str]:
        """Get capabilities of a job based on its name and description"""
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
    
    def _calculate_skill_match(self, job: Dict[str, Any], required_skills: List[str]) -> SkillMatch:
        """Calculate how well a job matches the required skills"""
        job_capabilities = self._get_job_capabilities(job)
        
        matched = [s for s in required_skills if s in job_capabilities]
        unmatched = [s for s in required_skills if s not in job_capabilities]
        
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
