"""
Visualization components for the Process Optimization Agent
Healthcare-specific visualizations
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import json
import os

from .models import Process, Schedule, Task, Resource, ScheduleEntry


class Visualizer:
    """Healthcare-specific visualization for process optimization results"""
    
    def __init__(self, style: str = "seaborn-v0_8"):
        """Initialize visualizer with plotting style"""
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        # Color palette for consistent visualization
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#2ecc71',
            'warning': '#e74c3c',
            'info': '#17A2B8',
            'light': '#F8F9FA',
            'dark': '#343A40'
        }
    
    # ========================================================================
    # UNIFIED VISUALIZATION INTERFACE (Separation of Concerns)
    # ========================================================================
    
    def create_summary_page(self, process: Process, schedule: Schedule,
                           process_type: str = None,
                           before_metrics: Optional[Dict] = None,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Create summary page based on detected process type.
        Automatically routes to healthcare or manufacturing visualization.
        """
        # Auto-detect process type if not provided
        if process_type is None:
            process_type = self._detect_process_type(process)
        
        # Route to appropriate visualization
        if process_type.upper() == "HEALTHCARE":
            return self.create_healthcare_summary_page(process, schedule, process_type, before_metrics, save_path)
        elif process_type.upper() in ["MANUFACTURING", "PRODUCTION"]:
            return self.create_manufacturing_summary_page(process, schedule, process_type, before_metrics, save_path)
        else:
            # Default to manufacturing for unknown types
            return self.create_manufacturing_summary_page(process, schedule, process_type, before_metrics, save_path)
    
    def create_allocation_page(self, process: Process, schedule: Schedule,
                              process_type: str = None,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Create allocation page based on detected process type.
        Automatically routes to healthcare or manufacturing visualization.
        """
        # Auto-detect process type if not provided
        if process_type is None:
            process_type = self._detect_process_type(process)
        
        # Route to appropriate visualization
        if process_type.upper() == "HEALTHCARE":
            return self.create_healthcare_allocation_page(process, schedule, process_type, save_path)
        elif process_type.upper() in ["MANUFACTURING", "PRODUCTION"]:
            return self.create_manufacturing_allocation_page(process, schedule, process_type, save_path)
        else:
            # Default to manufacturing for unknown types
            return self.create_manufacturing_allocation_page(process, schedule, process_type, save_path)
    
    def _detect_process_type(self, process: Process) -> str:
        """Detect process type based on process characteristics"""
        # Check for healthcare indicators
        healthcare_keywords = ['patient', 'medical', 'doctor', 'nurse', 'hospital', 'clinic', 
                              'consultation', 'examination', 'treatment', 'diagnosis']
        
        process_name_lower = process.name.lower()
        for keyword in healthcare_keywords:
            if keyword in process_name_lower:
                return "Healthcare"
        
        # Check task names for healthcare indicators
        for task in process.tasks:
            task_name_lower = task.name.lower()
            for keyword in healthcare_keywords:
                if keyword in task_name_lower:
                    return "Healthcare"
        
        # Default to manufacturing
        return "Manufacturing"
    
    # ========================================================================
    # HEALTHCARE-SPECIFIC VISUALIZATIONS
    # ========================================================================
    
    def create_healthcare_summary_page(self, process: Process, schedule: Schedule, 
                                       process_type: str = "Healthcare",
                                       before_metrics: Optional[Dict] = None,
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Create healthcare optimization summary page with:
        - Patient journey timeline (line graph with waiting times)
        - Summary table below the graph
        """
        fig = plt.figure(figsize=(16, 8))
        gs = GridSpec(2, 1, figure=fig, hspace=0.4, height_ratios=[3, 1])
        
        # Main title
        fig.suptitle(f'{process_type} Process Optimization: {process.name}', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        # Patient Journey Timeline (Top)
        ax1 = fig.add_subplot(gs[0])
        self._plot_patient_journey_timeline(ax1, process, schedule)
        
        # Summary Table (Bottom)
        ax2 = fig.add_subplot(gs[1])
        self._plot_summary_table(ax2, schedule)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Healthcare summary saved to: {save_path}")
        
        return fig
    
    def create_healthcare_allocation_page(self, process: Process, schedule: Schedule,
                                         process_type: str = "Healthcare",
                                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Create healthcare resource allocation page with:
        - Resource to task assignment timeline
        - Time and cost utilization per resource (bar charts)
        - Parallel task execution visualization
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
        
        # Main title
        fig.suptitle(f'{process_type} Process - Resource Allocation: {process.name}', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # 1. Resource to Task Assignment Timeline (Top - spans both columns)
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_resource_task_timeline(ax1, process, schedule)
        
        # 2. Time Utilization per Resource (Bar chart)
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_time_utilization_per_resource(ax2, process, schedule)
        
        # 3. Cost per Resource (Bar chart)
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_cost_per_resource(ax3, process, schedule)
        
        # 4. Parallel Task Execution (if any)
        ax4 = fig.add_subplot(gs[2, :])
        self._plot_parallel_tasks(ax4, process, schedule)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Healthcare allocation page saved to: {save_path}")
        
        return fig
    
    def _plot_patient_journey_timeline(self, ax, process: Process, schedule: Schedule):
        """Plot patient journey as an appealing line chart with step bands and labels"""
        # Get patient-facing tasks (user_involvement = DIRECT)
        patient_tasks = []
        for entry in schedule.entries:
            task = process.get_task_by_id(entry.task_id)
            if task and hasattr(task, 'user_involvement'):
                if str(task.user_involvement).upper() in ['DIRECT', 'UserInvolvement.DIRECT']:
                    patient_tasks.append((entry, task))
        
        if not patient_tasks:
            # If no user_involvement attribute, use all tasks
            patient_tasks = [(entry, process.get_task_by_id(entry.task_id)) 
                           for entry in schedule.entries]
        
        # Sort by start time
        patient_tasks.sort(key=lambda x: x[0].start_hour)
        
        # Prepare data
        steps = []
        cumulative_time = []
        task_durations = []
        waiting_times = []
        
        current_time = 0
        prev_end = 0
        
        for i, (entry, task) in enumerate(patient_tasks):
            step_name = task.name if task else entry.task_id
            steps.append(step_name)
            
            # Calculate waiting time
            wait_time = entry.start_hour - prev_end if i > 0 else 0
            waiting_times.append(wait_time)
            
            # Add waiting time to cumulative
            current_time += wait_time
            
            # Task duration
            duration = entry.end_hour - entry.start_hour
            task_durations.append(duration)
            
            # Add task duration to cumulative
            current_time += duration
            cumulative_time.append(current_time)
            
            prev_end = entry.end_hour
        
        # Create clean line plot without background bands
        x_pos = np.arange(len(steps))
        
        # Smooth-looking line with subtle area fill under curve
        ax.plot(x_pos, cumulative_time, marker='o', linewidth=3.5,
                markersize=10, color=self.colors['primary'], label='Cumulative Time', 
                zorder=3, markerfacecolor='white', markeredgewidth=2, markeredgecolor=self.colors['primary'])
        ax.fill_between(x_pos, cumulative_time, alpha=0.1, color=self.colors['primary'])
        
        # Add consistent-sized bars for task duration and waiting time
        fixed_bar_width = 0.6  # Consistent width for all bars
        fixed_bar_height = 0.4  # Fixed height for all green bars (increased to fit wrapped text)
        
        for i, (step, duration, wait) in enumerate(zip(steps, task_durations, waiting_times)):
            # Task duration block (green with fixed consistent size)
            bar_bottom = cumulative_time[i] - fixed_bar_height
            ax.bar(i, fixed_bar_height, bottom=bar_bottom, alpha=0.8, color='#2ecc71',
                   width=fixed_bar_width, label='Active Care' if i == 0 else '', 
                   zorder=2, edgecolor='#27ae60', linewidth=1.5)
            
            # Waiting time block (red with fixed size) - only show if there's waiting
            if wait > 0:
                wait_bottom = bar_bottom - fixed_bar_height
                ax.bar(i, fixed_bar_height, bottom=wait_bottom, alpha=0.8, color='#e74c3c',
                       width=fixed_bar_width, label='Waiting Time' if i == 1 else '', 
                       zorder=1, edgecolor='#c0392b', linewidth=1.5)
            
            # Add cumulative time marker label above each point (in minutes)
            cumulative_minutes = int(cumulative_time[i] * 60)
            ax.annotate(f"{cumulative_minutes}m",
                        xy=(i, cumulative_time[i]), xycoords='data',
                        xytext=(0, 15), textcoords='offset points', ha='center',
                        fontsize=10, weight='bold', color=self.colors['dark'],
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                 edgecolor=self.colors['primary'], alpha=0.9))
            
            # Add task name in center of green bar (with proper text wrapping)
            # Wrap text to fit within the box width
            import textwrap
            max_chars_per_line = 20  # Adjust based on box width
            wrapped_text = '\n'.join(textwrap.wrap(step, width=max_chars_per_line, break_long_words=False))
            
            ax.text(i, bar_bottom + fixed_bar_height/2, wrapped_text,
                   ha='center', va='center', fontsize=8, color='white', 
                   fontweight='bold', multialignment='center')
        
        ax.set_xlabel('Patient Journey Steps', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Time (hours)', fontsize=12, fontweight='bold')
        ax.set_title('Patient Journey Timeline', fontsize=15, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'Step {i+1}' for i in range(len(steps))], rotation=0, ha='center')
        ax.grid(True, alpha=0.25, axis='y', linestyle='--')
        ax.legend(loc='upper left', frameon=True)
        
        # Remove the summary annotation from the timeline (will be shown in table below)
    
    def _plot_summary_table(self, ax, schedule: Schedule):
        """Plot summary table below the timeline"""
        ax.axis('off')
        
        # Calculate summary metrics from schedule entries
        if not schedule.entries:
            ax.text(0.5, 0.5, 'No schedule data available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            return
        
        # Calculate patient journey metrics
        patient_tasks = []
        for entry in schedule.entries:
            patient_tasks.append(entry)
        
        if patient_tasks:
            # Sort by start time
            patient_tasks.sort(key=lambda x: x.start_hour)
            
            # Calculate metrics
            total_time = max(entry.end_hour for entry in patient_tasks) - min(entry.start_hour for entry in patient_tasks)
            total_active = sum(entry.end_hour - entry.start_hour for entry in patient_tasks)
            
            # Calculate waiting time (gaps between tasks)
            total_wait = 0
            for i in range(1, len(patient_tasks)):
                wait = patient_tasks[i].start_hour - patient_tasks[i-1].end_hour
                if wait > 0:
                    total_wait += wait
        else:
            total_time = total_active = total_wait = 0
        
        # Create table data (convert hours to minutes)
        table_data = [
            ['Metric', 'Value'],
            ['Total Patient Time', f'{int(total_time * 60)}m'],
            ['Active Care Time', f'{int(total_active * 60)}m'],
            ['Waiting Time', f'{int(total_wait * 60)}m']
        ]
        
        # Create table
        table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                        colWidths=[0.4, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style data rows
        for i in range(1, len(table_data)):
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f8f9fa')
                else:
                    table[(i, j)].set_facecolor('white')
        
        ax.set_title('Patient Journey Summary', fontsize=14, fontweight='bold', pad=20)
    
    def _plot_before_after_duration(self, ax, schedule: Schedule, before_metrics: Optional[Dict]):
        """Plot before/after duration comparison"""
        after_duration = schedule.total_duration_hours if hasattr(schedule, 'total_duration_hours') else 0
        
        # Calculate before duration (if not provided, estimate as 1.5x after)
        if before_metrics and 'duration' in before_metrics:
            before_duration = before_metrics['duration']
        else:
            before_duration = after_duration * 1.5
        
        durations = [before_duration, after_duration]
        labels = ['Before\nOptimization', 'After\nOptimization']
        colors = ['#F18F01', '#2E86AB']
        
        bars = ax.bar(labels, durations, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, duration in zip(bars, durations):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{duration:.1f} hours',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Add time saved annotation
        time_saved = before_duration - after_duration
        pct_saved = (time_saved / before_duration * 100) if before_duration > 0 else 0
        
        ax.text(0.5, 0.5, f'Time Saved\n{time_saved:.1f} hours\n({pct_saved:.1f}%)',
               transform=ax.transAxes, fontsize=11, ha='center',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        ax.set_ylabel('Duration (hours)', fontsize=12, fontweight='bold')
        ax.set_title('Process Duration Comparison', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_before_after_cost(self, ax, schedule: Schedule, before_metrics: Optional[Dict]):
        """Plot before/after cost comparison"""
        after_cost = schedule.total_cost if hasattr(schedule, 'total_cost') else 0
        
        # Calculate before cost (if not provided, estimate as 1.2x after)
        if before_metrics and 'cost' in before_metrics:
            before_cost = before_metrics['cost']
        else:
            before_cost = after_cost * 1.2
        
        costs = [before_cost, after_cost]
        labels = ['Before\nOptimization', 'After\nOptimization']
        colors = ['#F18F01', '#2E86AB']
        
        bars = ax.bar(labels, costs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, cost in zip(bars, costs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${cost:.2f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Add cost saved annotation
        cost_saved = before_cost - after_cost
        pct_saved = (cost_saved / before_cost * 100) if before_cost > 0 else 0
        
        if cost_saved > 0:
            ax.text(0.5, 0.5, f'Cost Saved\n${cost_saved:.2f}\n({pct_saved:.1f}%)',
                   transform=ax.transAxes, fontsize=11, ha='center',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        else:
            ax.text(0.5, 0.5, f'Cost Increase\n${abs(cost_saved):.2f}\n({abs(pct_saved):.1f}%)',
                   transform=ax.transAxes, fontsize=11, ha='center',
                   bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.7))
        
        ax.set_ylabel('Total Cost ($)', fontsize=12, fontweight='bold')
        ax.set_title('Process Cost Comparison', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_resource_utilization_summary(self, ax, process: Process, schedule: Schedule):
        """Plot resource utilization summary"""
        resource_hours = {}
        for entry in schedule.entries:
            resource_id = entry.resource_id
            duration = entry.end_hour - entry.start_hour
            resource_hours[resource_id] = resource_hours.get(resource_id, 0) + duration
        
        if not resource_hours:
            ax.text(0.5, 0.5, 'No resource data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            return
        
        # Get resource names
        resource_names = []
        hours = []
        for res_id, hrs in resource_hours.items():
            resource = process.get_resource_by_id(res_id)
            name = resource.name if resource else res_id
            resource_names.append(name)
            hours.append(hrs)
        
        # Create horizontal bar chart
        y_pos = range(len(resource_names))
        bars = ax.barh(y_pos, hours, color='#2E86AB', alpha=0.7, edgecolor='black')
        
        # Add value labels
        for i, (bar, hr) in enumerate(zip(bars, hours)):
            ax.text(hr + 0.5, bar.get_y() + bar.get_height()/2, f'{hr:.1f}h',
                   va='center', fontsize=9, fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(resource_names, fontsize=10)
        ax.set_xlabel('Hours Assigned', fontsize=11, fontweight='bold')
        ax.set_title('Resource Utilization', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    def _plot_key_metrics_summary(self, ax, schedule: Schedule, before_metrics: Optional[Dict]):
        """Plot key metrics summary as a table"""
        ax.axis('off')
        
        # Calculate metrics
        total_duration = schedule.total_duration_hours if hasattr(schedule, 'total_duration_hours') else 0
        total_cost = schedule.total_cost if hasattr(schedule, 'total_cost') else 0
        num_tasks = len(schedule.entries)
        
        # Calculate improvements
        if before_metrics:
            before_duration = before_metrics.get('duration', total_duration * 1.5)
            before_cost = before_metrics.get('cost', total_cost * 1.2)
            time_improvement = ((before_duration - total_duration) / before_duration * 100) if before_duration > 0 else 0
            cost_improvement = ((before_cost - total_cost) / before_cost * 100) if before_cost > 0 else 0
        else:
            time_improvement = 0
            cost_improvement = 0
        
        # Create table data
        table_data = [
            ['Metric', 'Value'],
            ['Total Duration', f'{total_duration:.1f} hours'],
            ['Total Cost', f'${total_cost:.2f}'],
            ['Tasks Completed', f'{num_tasks}'],
            ['Time Improvement', f'{time_improvement:.1f}%'],
            ['Cost Improvement', f'{cost_improvement:.1f}%']
        ]
        
        table = ax.table(cellText=table_data, loc='center', cellLoc='left',
                        colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style data rows
        for i in range(1, len(table_data)):
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        ax.set_title('Key Performance Metrics', fontsize=13, fontweight='bold', pad=20)
    
    def _plot_resource_task_timeline(self, ax, process: Process, schedule: Schedule):
        """Plot resource to task assignment timeline with consistent arrows and aligned labels"""
        if not schedule.entries:
            ax.text(0.5, 0.5, 'No scheduled tasks', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            return
        
        # Group entries by resource
        resource_tasks = {}
        for entry in schedule.entries:
            if entry.resource_id not in resource_tasks:
                resource_tasks[entry.resource_id] = []
            resource_tasks[entry.resource_id].append(entry)
        
        # Sort resources
        resources = sorted(resource_tasks.keys())
        
        # Calculate max time for consistent arrow sizing
        max_time = max(entry.end_hour for entry in schedule.entries) if schedule.entries else 3
        
        # Plot timeline for each resource
        y_pos = 0
        y_labels = []
        y_positions = []
        task_names_right = []
        
        for resource_id in resources:
            entries = sorted(resource_tasks[resource_id], key=lambda e: e.start_hour)
            resource = process.get_resource_by_id(resource_id)
            resource_name = resource.name if resource else resource_id
            
            # Plot each task assignment on a separate row
            for entry in entries:
                task = process.get_task_by_id(entry.task_id)
                task_name = task.name if task else entry.task_id
                duration = entry.end_hour - entry.start_hour
                
                # Draw end-to-end arrow (from resource name area to task name area)
                arrow_start = 0.1  # Start right after resource name
                arrow_end = max_time + 0.1  # End just before task name
                arrow_length = arrow_end - arrow_start
                
                # Draw arrow spanning full width
                ax.arrow(arrow_start, y_pos, arrow_length - 0.08, 0,
                        head_width=0.08, head_length=0.08, fc='#2E86AB', ec='#2E86AB',
                        linewidth=3, alpha=0.8, length_includes_head=True)
                
                # Add duration label in the center of the full arrow
                arrow_center = arrow_start + arrow_length / 2
                ax.text(arrow_center, y_pos, f'{int(duration*60)}m',
                       ha='center', va='center', fontsize=8, fontweight='bold', 
                       color='white', bbox=dict(boxstyle='round,pad=0.2', 
                       facecolor='#2E86AB', alpha=0.9))
                
                # Store task name for right side listing
                task_names_right.append((y_pos, task_name))
                
                # Add resource name at the left (as blue blocks like task names)
                ax.text(-0.2, y_pos, resource_name, ha='right', va='center',
                       fontsize=9, fontweight='normal',
                       bbox=dict(boxstyle='round,pad=0.25', facecolor='lightblue', 
                                edgecolor='#2E86AB', alpha=0.8))
                
                y_labels.append(resource_name)
                y_positions.append(y_pos)
                y_pos += 1  # Increment for EACH task, not just each resource
        
        # Add task names on the right side in a list format
        right_x_pos = max_time + 0.3
        for task_y, task_name in task_names_right:
            ax.text(right_x_pos, task_y, task_name, ha='left', va='center',
                   fontsize=9, fontweight='normal',
                   bbox=dict(boxstyle='round,pad=0.25', facecolor='lightblue', 
                            edgecolor='#2E86AB', alpha=0.8))
        
        # Set clean limits (accommodate both resource names on left and task names on right)
        ax.set_ylim(-0.5, y_pos - 0.5)
        ax.set_xlim(-0.8, max_time + 2.5)
        
        # Clean up the plot
        ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
        ax.set_title('Resource → Task Assignments (Timeline)', fontsize=14, fontweight='bold')
        ax.set_yticks(y_positions)
        ax.set_yticklabels([])  # Remove y-axis labels since resource names are inline
        
        # Remove all spines and grid for clean look
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.grid(False)
        
        # Only show x-axis ticks at meaningful intervals (starting from 0)
        ax.set_xticks(np.arange(0, max_time + 1, 0.5))
        ax.tick_params(axis='y', which='both', left=False, right=False)
        
        # Ensure x-axis starts at 0 visually
        ax.axvline(x=0, color='black', linewidth=1, alpha=0.3)
    
    def _plot_time_utilization_per_resource(self, ax, process: Process, schedule: Schedule):
        """Plot time utilization per resource as bar chart (in minutes)"""
        resource_minutes = {}
        for entry in schedule.entries:
            duration_hours = entry.end_hour - entry.start_hour
            duration_minutes = duration_hours * 60  # Convert to minutes
            resource_minutes[entry.resource_id] = resource_minutes.get(entry.resource_id, 0) + duration_minutes
        
        if not resource_minutes:
            ax.text(0.5, 0.5, 'No resource data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            return
        
        # Get resource names and minutes
        names = []
        minutes = []
        colors_list = []
        
        for res_id, mins in sorted(resource_minutes.items(), key=lambda x: x[1], reverse=True):
            resource = process.get_resource_by_id(res_id)
            name = resource.name if resource else res_id
            names.append(name)
            minutes.append(mins)
            
            # Color based on utilization (still calculate based on hours for percentage)
            if resource:
                hours = mins / 60
                utilization = (hours / resource.total_available_hours * 100) if resource.total_available_hours > 0 else 0
                if utilization > 80:
                    colors_list.append('#e74c3c')  # Red - high
                elif utilization > 50:
                    colors_list.append('#2ecc71')  # Green - good
                else:
                    colors_list.append('#f39c12')  # Orange - low
            else:
                colors_list.append('#2E86AB')
        
        # Create bar chart
        bars = ax.bar(range(len(names)), minutes, color=colors_list, alpha=0.7, edgecolor='black')
        
        # Add value labels (in minutes)
        for bar, mins in zip(bars, minutes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(mins)}m',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Remove x-axis labels and add resource names as blocks below bars
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([])  # Remove default labels
        
        # Add resource names as text blocks below each bar (with better wrapping for long names)
        for i, name in enumerate(names):
            # Wrap long names to prevent overlap - more aggressive wrapping
            if len(name) > 12:
                words = name.split()
                if len(words) > 2:
                    # Split into 2-3 lines for better readability
                    third = len(words) // 3
                    if third > 0:
                        name = '\n'.join([' '.join(words[:third]), ' '.join(words[third:2*third]), ' '.join(words[2*third:])])
                    else:
                        mid = len(words) // 2
                        name = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
                elif len(words) == 2:
                    name = '\n'.join(words)
                else:
                    # Single long word - truncate with ellipsis
                    name = name[:10] + '...'
            
            ax.text(i, -max(minutes) * 0.22, name, ha='center', va='top',
                   fontsize=7, rotation=0, fontweight='normal',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', 
                            edgecolor='#2E86AB', alpha=0.8, linewidth=1))
        
        ax.set_ylabel('Minutes Assigned', fontsize=11, fontweight='bold')
        ax.set_title('Time Utilization per Resource', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(bottom=-max(minutes) * 0.35)  # Add more space for wrapped labels
    
    def _plot_cost_per_resource(self, ax, process: Process, schedule: Schedule):
        """Plot cost per resource as bar chart"""
        resource_costs = {}
        for entry in schedule.entries:
            # Calculate cost based on resource hourly rate and task duration
            resource = process.get_resource_by_id(entry.resource_id)
            if resource:
                duration = entry.end_hour - entry.start_hour
                hourly_rate = getattr(resource, 'hourly_rate', 0)
                cost = duration * hourly_rate
                resource_costs[entry.resource_id] = resource_costs.get(entry.resource_id, 0) + cost
        
        if not resource_costs or all(cost == 0 for cost in resource_costs.values()):
            ax.text(0.5, 0.5, 'No cost data available\n(Resource hourly rates not set)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
            return
        
        # Get resource names and costs
        names = []
        costs = []
        
        for res_id, cost in sorted(resource_costs.items(), key=lambda x: x[1], reverse=True):
            resource = process.get_resource_by_id(res_id)
            name = resource.name if resource else res_id
            names.append(name)
            costs.append(cost)
        
        # Create bar chart
        bars = ax.bar(range(len(names)), costs, color='#A23B72', alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bar, cost in zip(bars, costs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${cost:.2f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Remove x-axis labels and add resource names as blocks below bars
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([])  # Remove default labels
        
        # Add resource names as text blocks below each bar (with better wrapping for long names)
        for i, name in enumerate(names):
            # Wrap long names to prevent overlap - more aggressive wrapping
            if len(name) > 12:
                words = name.split()
                if len(words) > 2:
                    # Split into 2-3 lines for better readability
                    third = len(words) // 3
                    if third > 0:
                        name = '\n'.join([' '.join(words[:third]), ' '.join(words[third:2*third]), ' '.join(words[2*third:])])
                    else:
                        mid = len(words) // 2
                        name = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
                elif len(words) == 2:
                    name = '\n'.join(words)
                else:
                    # Single long word - truncate with ellipsis
                    name = name[:10] + '...'
            
            ax.text(i, -max(costs) * 0.22, name, ha='center', va='top',
                   fontsize=7, rotation=0, fontweight='normal',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', 
                            edgecolor='#2E86AB', alpha=0.8, linewidth=1))
        
        ax.set_ylabel('Total Cost ($)', fontsize=11, fontweight='bold')
        ax.set_title('Cost per Resource', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(bottom=-max(costs) * 0.35)  # Add more space for wrapped labels
    
    def _plot_parallel_tasks(self, ax, process: Process, schedule: Schedule):
        """Plot parallel task execution visualization (similar to image 3)"""
        if not schedule.entries:
            ax.text(0.5, 0.5, 'No scheduled tasks', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            return
        
        # Find parallel tasks (tasks with overlapping time windows)
        parallel_groups = []
        sorted_entries = sorted(schedule.entries, key=lambda e: e.start_hour)
        
        # Group tasks by time overlap
        for i, entry1 in enumerate(sorted_entries):
            parallel_set = {entry1.task_id}
            for entry2 in sorted_entries[i+1:]:
                # Check if tasks overlap
                if entry2.start_hour < entry1.end_hour:
                    parallel_set.add(entry2.task_id)
            
            if len(parallel_set) > 1:
                # Check if this group already exists
                is_new = True
                for existing_group in parallel_groups:
                    if parallel_set == existing_group:
                        is_new = False
                        break
                if is_new:
                    parallel_groups.append(parallel_set)
        
        if not parallel_groups:
            ax.text(0.5, 0.5, 'No parallel tasks detected\n(All tasks run sequentially)',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
            ax.axis('off')
            return
        
        # Plot parallel groups
        ax.axis('off')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, len(parallel_groups) + 1)
        
        ax.text(5, len(parallel_groups) + 0.5, 'Parallel Task Groups',
               ha='center', fontsize=14, fontweight='bold')
        
        for i, group in enumerate(parallel_groups):
            y_pos = len(parallel_groups) - i - 0.5
            
            # Draw boxes for each task in the group
            x_start = 1
            box_width = 8 / len(group)
            
            for j, task_id in enumerate(sorted(group)):
                task = process.get_task_by_id(task_id)
                task_name = task.name if task else task_id
                
                x_pos = x_start + j * box_width
                
                # Draw box
                rect = Rectangle((x_pos, y_pos - 0.3), box_width - 0.1, 0.6,
                                facecolor='#A8DADC', edgecolor='#457B9D',
                                linewidth=2, alpha=0.8)
                ax.add_patch(rect)
                
                # Add task name
                ax.text(x_pos + box_width/2 - 0.05, y_pos, task_name,
                       ha='center', va='center', fontsize=8, fontweight='bold',
                       wrap=True)
            
            # Add group label
            ax.text(0.5, y_pos, f'Step {i+1}',
                   ha='right', va='center', fontsize=10, fontweight='bold')
        
        # Add legend
        ax.text(5, 0.2, f'Total Parallel Groups: {len(parallel_groups)}',
               ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # ========================================================================
    # MANUFACTURING-SPECIFIC VISUALIZATIONS
    # ========================================================================
    
    def create_manufacturing_allocation_page(self, process: Process, schedule: Schedule,
                                            process_type: str = "Manufacturing",
                                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Create manufacturing resource allocation page with:
        - Resource → Task Assignments (Timeline)
        - Time Utilization per Resource
        - Cost per Resource
        - Parallel Task Groups
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3, height_ratios=[2, 1.5, 1.5])
        
        # Main title - show process type and process name
        # Map generic types to more specific labels
        type_label_map = {
            'manufacturing': 'Manufacturing Process',
            'healthcare': 'Healthcare Process',
            'insurance': 'Insurance Process',
            'banking': 'Banking Process',
            'finance': 'Finance Process'
        }
        
        # Get the appropriate label
        process_type_lower = process_type.lower() if process_type else 'process'
        type_label = type_label_map.get(process_type_lower, f'{process_type} Process' if process_type else 'Process')
        
        fig.suptitle(f'{type_label} - Resource Allocation: {process.name}', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # 1. Resource → Task Timeline (Top - spans both columns)
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_resource_task_timeline(ax1, process, schedule)
        
        # 2. Time Utilization per Resource (Middle Left)
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_time_utilization_per_resource(ax2, process, schedule)
        
        # 3. Cost per Resource (Middle Right)
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_cost_per_resource(ax3, process, schedule)
        
        # 4. Parallel Task Groups (Bottom - spans both columns)
        ax4 = fig.add_subplot(gs[2, :])
        self._plot_parallel_tasks(ax4, process, schedule)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Manufacturing allocation saved to: {save_path}")
        
        return fig
    
    def create_manufacturing_summary_page(self, process: Process, schedule: Schedule,
                                         process_type: str = "Manufacturing",
                                         before_metrics: Optional[Dict] = None,
                                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Create manufacturing optimization summary page with:
        - Duration comparison (Before/After)
        - Peak Resource Usage comparison
        - Total Cost comparison
        - Summary table at bottom
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3, height_ratios=[2, 2, 1])
        
        # Main title
        fig.suptitle(f'Optimization Summary — {process.name}', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # 1. Duration Comparison (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_manufacturing_duration_comparison(ax1, schedule, before_metrics)
        
        # 2. Peak Resource Usage (Top Middle)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_manufacturing_resource_comparison(ax2, process, schedule, before_metrics)
        
        # 3. Total Cost (Top Right)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_manufacturing_cost_comparison(ax3, process, schedule, before_metrics)
        
        # 4. Summary Table (Bottom - spans all columns)
        ax4 = fig.add_subplot(gs[2, :])
        self._plot_manufacturing_summary_table(ax4, process, schedule, before_metrics)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Manufacturing summary saved to: {save_path}")
        
        return fig
    
    def _plot_manufacturing_duration_comparison(self, ax, schedule, before_metrics):
        """Plot before/after duration comparison for manufacturing"""
        # Calculate after duration
        if schedule.entries:
            after_duration = max(entry.end_hour for entry in schedule.entries)
        else:
            after_duration = 0
        
        # Get before duration
        if before_metrics and 'duration' in before_metrics:
            before_duration = before_metrics['duration']
        else:
            before_duration = after_duration * 1.5  # Estimate
        
        # Create bar chart
        categories = ['Before', 'After']
        values = [before_duration, after_duration]
        colors = ['#F18F01', '#2E86AB']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f} hours',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add time saved annotation
        time_saved = before_duration - after_duration
        improvement_pct = (time_saved / before_duration * 100) if before_duration > 0 else 0
        
        ax.text(1.5, max(values) * 0.5, 
               f'Time Saved:\n{time_saved:.1f} hours\n({improvement_pct:.1f}%)',
               ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
        
        ax.set_ylabel('Hours', fontsize=12, fontweight='bold')
        ax.set_title('Duration (hours)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(values) * 1.2)
    
    def _plot_manufacturing_resource_comparison(self, ax, process, schedule, before_metrics):
        """Plot before/after peak resource usage comparison"""
        # Calculate after resource usage
        after_resources = len(set(entry.resource_id for entry in schedule.entries))
        
        # Get before resource usage
        if before_metrics and 'resources' in before_metrics:
            before_resources = before_metrics['resources']
        else:
            before_resources = len(process.resources)  # Assume all resources used before
        
        # Create bar chart
        categories = ['Before', 'After']
        values = [before_resources, after_resources]
        colors = ['#F18F01', '#2E86AB']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(val)} resources',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add efficiency annotation
        efficiency = (after_resources / before_resources) if before_resources > 0 else 1
        ax.text(1.5, max(values) * 0.5,
               f'Efficiency:\n{efficiency:.1f} ± {1-efficiency:.1f} hours/resource',
               ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
        
        ax.set_ylabel('Number of Resources', fontsize=12, fontweight='bold')
        ax.set_title('Peak Resource Usage', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(values) * 1.3)
    
    def _plot_manufacturing_cost_comparison(self, ax, process, schedule, before_metrics):
        """Plot before/after total cost comparison"""
        # Calculate after cost from schedule
        if hasattr(schedule, 'total_cost') and schedule.total_cost > 0:
            after_cost = schedule.total_cost
        else:
            # Calculate from entries with actual resource rates
            after_cost = 0
            for entry in schedule.entries:
                duration = entry.end_hour - entry.start_hour
                resource = process.get_resource_by_id(entry.resource_id)
                if resource:
                    after_cost += duration * resource.hourly_rate
                else:
                    after_cost += duration * 50  # Default rate estimate
        
        # Get before cost
        if before_metrics and 'cost' in before_metrics:
            before_cost = before_metrics['cost']
        else:
            before_cost = after_cost * 1.2  # Estimate 20% higher
        
        # Create bar chart
        categories = ['Before', 'After']
        values = [before_cost, after_cost]
        colors = ['#F18F01', '#2E86AB']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${val:.2f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add cost savings annotation
        cost_saved = before_cost - after_cost
        savings_pct = (cost_saved / before_cost * 100) if before_cost > 0 else 0
        
        ax.text(1.5, max(values) * 0.5,
               f'Cost Savings:\n${cost_saved:.2f} ({savings_pct:.1f}%)',
               ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))
        
        ax.set_ylabel('Cost (USD)', fontsize=12, fontweight='bold')
        ax.set_title('Total Cost ($)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(values) * 1.2)
    
    def _plot_manufacturing_summary_table(self, ax, process, schedule, before_metrics):
        """Plot summary table for manufacturing optimization"""
        ax.axis('off')
        
        # Calculate metrics from schedule
        if schedule.entries:
            after_duration = max(entry.end_hour for entry in schedule.entries)
            after_resources = len(set(entry.resource_id for entry in schedule.entries))
            
            # Calculate actual cost from entries with resource rates
            after_cost = 0
            for entry in schedule.entries:
                duration = entry.end_hour - entry.start_hour
                resource = process.get_resource_by_id(entry.resource_id)
                if resource:
                    after_cost += duration * resource.hourly_rate
                else:
                    after_cost += duration * 50  # Default rate estimate
        else:
            after_duration = after_resources = after_cost = 0
        
        # Get before metrics
        if before_metrics:
            before_duration = before_metrics.get('duration', after_duration * 1.5)
            before_resources = before_metrics.get('resources', after_resources + 2)
            before_cost = before_metrics.get('cost', after_cost * 1.2)
        else:
            before_duration = after_duration * 1.5
            before_resources = after_resources + 2
            before_cost = after_cost * 1.2
        
        # Calculate improvements
        time_saved = before_duration - after_duration
        time_improvement = (time_saved / before_duration * 100) if before_duration > 0 else 0
        resource_reduction = before_resources - after_resources
        cost_saved = before_cost - after_cost
        cost_improvement = (cost_saved / before_cost * 100) if before_cost > 0 else 0
        
        # Create table data
        table_data = [
            ['Metric', 'Before', 'After', 'Improvement'],
            ['Duration', f'{before_duration:.1f}h', f'{after_duration:.1f}h', f'{time_saved:.1f}h ({time_improvement:.1f}%)'],
            ['Resources', f'{int(before_resources)}', f'{int(after_resources)}', f'-{int(resource_reduction)}'],
            ['Total Cost', f'${before_cost:.2f}', f'${after_cost:.2f}', f'${cost_saved:.2f} ({cost_improvement:.1f}%)']
        ]
        
        # Create table
        table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                        colWidths=[0.25, 0.2, 0.2, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style data rows
        for i in range(1, len(table_data)):
            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f8f9fa')
                else:
                    table[(i, j)].set_facecolor('white')
                
                # Highlight improvement column
                if j == 3:
                    table[(i, j)].set_facecolor('#d4edda')
        
        ax.set_title('Optimization Results Summary', fontsize=14, fontweight='bold', pad=20)
