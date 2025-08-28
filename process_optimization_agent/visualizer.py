"""
Visualization components for the Process Optimization Agent
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
import tempfile
import webbrowser

# Graphviz is optional; import lazily inside methods that need it
# This avoids crashing the app at import time when graphviz isn't installed

from .models import Process, Schedule, Task, Resource, ScheduleEntry


class Visualizer:
    """Comprehensive visualization for process optimization results"""
    
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
            'success': '#C73E1D',
            'warning': '#FFB400',
            'info': '#17A2B8',
            'light': '#F8F9FA',
            'dark': '#343A40'
        }
        
        # Task type colors
        self.task_colors = {
            'development': '#2E86AB',
            'testing': '#A23B72', 
            'design': '#F18F01',
            'review': '#C73E1D',
            'documentation': '#FFB400',
            'deployment': '#17A2B8',
            'other': '#6C757D'
        }
        
        # Graphviz style settings
        self.graph_style = {
            'graph': {
                'rankdir': 'LR',
                'splines': 'ortho',
                'nodesep': '0.8',
                'ranksep': '1.0',
                'fontname': 'Arial',
                'fontsize': '12',
                'fontcolor': '#333333',
                'bgcolor': '#ffffff',
                'dpi': '150',
                'pad': '0.5'
            },
            'node': {
                'shape': 'box',
                'style': 'rounded,filled',
                'fillcolor': '#f8f9fa',
                'color': '#2E86AB',
                'fontname': 'Arial',
                'fontsize': '10',
                'height': '0.2',
                'width': '0.4',
                'margin': '0.1,0.05'
            },
            'edge': {
                'color': '#6c757d',
                'arrowhead': 'vee',
                'arrowsize': '0.7',
                'penwidth': '1.2'
            }
        }
    
    def create_gantt_chart(self, process: Process, schedule: Schedule, 
                          save_path: Optional[str] = None) -> plt.Figure:
        """Create a Gantt chart showing task timeline using simplified hours"""
        fig, ax = plt.subplots(figsize=(14, max(8, len(schedule.entries) * 0.5)))
        
        if not schedule.entries:
            ax.text(0.5, 0.5, 'No scheduled tasks', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=16)
            return fig
        
        # Sort entries by start hour (use start_time as fallback)
        sorted_entries = sorted(schedule.entries, 
                              key=lambda x: getattr(x, 'start_hour', 
                                                   (x.start_time - process.start_date).total_seconds() / 3600.0))
        
        # Create task bars
        y_positions = {}
        current_y = 0
        
        for i, entry in enumerate(sorted_entries):
            task = process.get_task_by_id(entry.task_id)
            resource = process.get_resource_by_id(entry.resource_id)
            
            if not task:
                continue
            
            # Assign y position
            if entry.task_id not in y_positions:
                y_positions[entry.task_id] = current_y
                current_y += 1
            
            y_pos = y_positions[entry.task_id]
            
            # Get start and end hours (use hour fields if available, otherwise calculate)
            start_hour = getattr(entry, 'start_hour', 
                               (entry.start_time - process.start_date).total_seconds() / 3600.0)
            end_hour = getattr(entry, 'end_hour',
                             (entry.end_time - process.start_date).total_seconds() / 3600.0)
            duration_hours = end_hour - start_hour
            
            # Determine task color
            task_type = self._categorize_task(task)
            color = self.task_colors.get(task_type, self.task_colors['other'])
            
            # Create bar using hours
            ax.barh(y_pos, duration_hours, left=start_hour,
                    height=0.6, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Add task label
            label = f"{task.name}"
            if resource:
                label += f" ({resource.name})"
            
            ax.text(start_hour + duration_hours/2, y_pos,
                   label, ha='center', va='center', fontsize=9, weight='bold')
        
        # Format axes
        ax.set_ylim(-0.5, current_y - 0.5)
        ax.set_yticks(range(current_y))
        ax.set_yticklabels([process.get_task_by_id(tid).name 
                           for tid, _ in sorted(y_positions.items(), key=lambda x: x[1])])
        
        # Format x-axis for hours
        ax.set_xlabel('Hours from Project Start', fontsize=12)
        ax.set_ylabel('Tasks', fontsize=12)
        ax.set_title(f'Gantt Chart - {process.name}', fontsize=16, weight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Set x-axis limits
        max_hour = max(getattr(e, 'end_hour', 
                              (e.end_time - process.start_date).total_seconds() / 3600.0) 
                      for e in schedule.entries)
        ax.set_xlim(0, max_hour * 1.05)
        
        # Add legend for task types
        legend_elements = []
        used_types = set()
        for entry in schedule.entries:
            task = process.get_task_by_id(entry.task_id)
            if task:
                task_type = self._categorize_task(task)
                if task_type not in used_types:
                    legend_elements.append(plt.Rectangle((0,0),1,1, 
                                                       color=self.task_colors[task_type], 
                                                       label=task_type.title()))
                    used_types.add(task_type)
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

    def plot_resource_allocation_summary(self, process: Process, schedule: Schedule,
                                         output_file: str, show: bool = False) -> str:
        """
        Render a resource allocation summary as an image with a tabular view.

        Columns: Task, Resource, Start, End, Hours, Rate, Cost

        Returns the saved image path.
        """
        try:
            import os
            import matplotlib
            try:
                matplotlib.use('Agg')
            except Exception:
                pass
            import matplotlib.pyplot as plt

            root, ext = os.path.splitext(output_file)
            if not ext:
                output_file = root + '.png'

            # Build grouped rows by resource with optional detailed lines
            show_detailed = False  # concise by default
            rows = []
            total_cost = 0.0
            total_hours = 0.0
            tasks_count = len(schedule.entries)
            resources_used = set()

            # Group entries by resource
            by_res: Dict[str, List] = {}
            for e in sorted(schedule.entries, key=lambda x: (x.resource_id, x.start_time, x.end_time)):
                by_res.setdefault(e.resource_id, []).append(e)

            # Keep sequential (pre-optimization) totals per resource
            seq_totals: Dict[str, Dict[str, float]] = {}

            for res_id, entries in by_res.items():
                res = process.get_resource_by_id(res_id)
                res_label = f"Resource: {res.name} ({getattr(res, 'role', '')})".strip() if res else f"Resource: {res_id}"
                rate_display = f"${float(getattr(res, 'hourly_rate', 0.0) or 0.0):,.2f}/hr" if res else ""
                resources_used.add(res_id)

                # Resource header row (only if detailed)
                if show_detailed:
                    rows.append([res_label, rate_display, "", "", "", "", ""])

                res_hours = 0.0
                res_cost = 0.0
                for e in sorted(entries, key=lambda x: (x.start_time, x.end_time)):
                    task = process.get_task_by_id(e.task_id)
                    task_name = task.name if task else e.task_id
                    
                    # Use simplified hour-based calculations
                    if hasattr(e, 'start_hour') and hasattr(e, 'end_hour'):
                        start_h = e.start_hour
                        end_h = e.end_hour
                        hours = end_h - start_h
                    else:
                        hours = task.duration_hours if task else 0.0
                        start_h = (e.start_time - process.start_date).total_seconds() / 3600.0
                        end_h = start_h + hours
                    
                    start_s = f"Hour {start_h:.1f}"
                    end_s = f"Hour {end_h:.1f}"
                    rate = float(getattr(res, 'hourly_rate', 0.0) or 0.0) if res else 0.0
                    cost = float(getattr(e, 'cost', hours * rate) or 0.0)
                    res_hours += hours
                    res_cost += cost
                    total_hours += hours
                    total_cost += cost
                    if show_detailed:
                        rows.append([f"- {task_name}", "", start_s, end_s, f"{hours:.1f}", f"${rate:,.2f}", f"${cost:,.2f}"])

                # Subtotal row per resource
                if show_detailed:
                    rows.append(["Subtotal", "", "", "", f"{res_hours:.1f}", "", f"${res_cost:,.2f}"])
                    # Spacer row
                    rows.append(["", "", "", "", "", "", ""])

                # Save sequential totals for compact section
                seq_totals[res_id] = {
                    "name": res.name if res else res_id,
                    "hours": res_hours,
                    "cost": res_cost,
                    "rate": float(getattr(res, 'hourly_rate', 0.0) or 0.0) if res else 0.0,
                }

            # Compute summary stats before building table so we can prepend a summary section
            def _compute_peak_parallel(entries_list: List) -> int:
                events = []
                for en in entries_list:
                    events.append((en.start_time, 1))
                    events.append((en.end_time, -1))
                events.sort()
                cur = peak = 0
                for _, d in events:
                    cur += d
                    peak = max(peak, cur)
                return peak

            # Compute project time range in hours
            if schedule.entries:
                if all(hasattr(e, 'start_hour') and hasattr(e, 'end_hour') for e in schedule.entries):
                    min_hour = min(e.start_hour for e in schedule.entries)
                    max_hour = max(e.end_hour for e in schedule.entries)
                    date_range = f"Hour {min_hour:.1f} → Hour {max_hour:.1f}"
                else:
                    min_start = min((e.start_time for e in schedule.entries), default=None)
                    max_end = max((e.end_time for e in schedule.entries), default=None)
                    date_range = f"{min_start.strftime('%Y-%m-%d %H:%M')} → {max_end.strftime('%Y-%m-%d %H:%M')}" if (min_start and max_end) else "N/A"
            else:
                date_range = "N/A"
            peak = _compute_peak_parallel(schedule.entries) if schedule.entries else 0

            # Compute total elapsed hours (simplified - no business hours)
            def _compute_total_elapsed_hours(entries_list) -> float:
                if not entries_list:
                    return 0.0
                
                # If entries have hour fields, use them
                if all(hasattr(e, 'start_hour') and hasattr(e, 'end_hour') for e in entries_list):
                    min_start = min(e.start_hour for e in entries_list)
                    max_end = max(e.end_hour for e in entries_list)
                    return max_end - min_start
                
                # Otherwise compute from datetime
                min_start_time = min(e.start_time for e in entries_list)
                max_end_time = max(e.end_time for e in entries_list)
                return (max_end_time - min_start_time).total_seconds() / 3600.0

            elapsed_hours = _compute_total_elapsed_hours(schedule.entries)

            # Detect parallel task groups
            def _parallel_groups(entries_list: List) -> List[List[str]]:
                if not entries_list:
                    return []
                events = []
                for en in entries_list:
                    events.append((en.start_time, 1, en.task_id))
                    events.append((en.end_time, -1, en.task_id))
                events.sort(key=lambda x: (x[0], x[1]))
                active = set(); groups = []
                for t, typ, tid in events:
                    if typ == 1:
                        active.add(tid)
                        if len(active) >= 2:
                            groups.append(tuple(sorted(active)))
                    else:
                        active.discard(tid)
                unique = []
                seen = set()
                for g in groups:
                    key = frozenset(g)
                    if key not in seen and len(key) >= 2:
                        seen.add(key)
                        unique.append(list(key))
                return unique

            def _tname(tid: str) -> str:
                t = process.get_task_by_id(tid)
                return t.name if t else tid
            par_groups = [[_tname(tid) for tid in grp] for grp in _parallel_groups(schedule.entries)]

            # Calculate before/after costs
            before_cost = total_cost  # Sequential cost
            after_cost = sum(e.cost for e in schedule.entries if hasattr(e, 'cost'))  # Actual assigned cost
            
            # Prepend a table summary section so it's always visible
            summary_section = [
                ["Summary", "", "", "", "", "", ""],
                ["Total task hours scheduled", f"{total_hours:.1f} hours", "", "", "", "", f"${after_cost:,.2f}"],
                ["Elapsed window (hours)", f"{elapsed_hours:.1f} hours", "", "", "", "", "—"],
                ["Sequential baseline (sum of task hours)", f"{total_hours:.1f} hours", "", "", "", "", f"${before_cost:,.2f}"],
                ["Resources used", f"{len(resources_used)}", "", "", "", "", ""],
                ["Tasks scheduled", f"{tasks_count}", "", "", "", "", ""],
                ["Peak parallel tasks", f"{peak}", "", "", "", "", ""],
                ["Date range", date_range, "", "", "", "", ""],
                ["Parallel groups", " | ".join([" / ".join(g) for g in par_groups]) if par_groups else "None", "", "", "", "", ""],
                ["", "", "", "", "", "", ""],  # spacer
                ["Cost Comparison", "", "", "", "", "", ""],
                ["Before (Sequential)", "", "", "", f"{total_hours:.1f} hours", "", f"${before_cost:,.2f}"],
                ["After (Optimized)", "", "", "", f"{total_hours:.1f} hours", "", f"${after_cost:,.2f}"],
                ["Savings", "", "", "", "", "", f"${(before_cost - after_cost):+,.2f} ({(1 - after_cost/before_cost)*100:.1f}%)" if before_cost > 0 else "N/A"],
                ["", "", "", "", "", "", ""],  # spacer
            ]

            # Start with compact summary KPIs first
            rows = summary_section

            # Add 'Before Optimization' compact per-resource table (sequential totals)
            try:
                rows.append(["", "", "", "", "", "", ""])  # spacer
                rows.append([f"Before Optimization (sequential {total_hours:.1f}h)", "", "", "", "", "", ""])
                rows.append(["Resource", "", "", "", "Hours", "Share", "Cost"])
                # Sort by hours desc and show top 5 + Others
                seq_items = sorted(seq_totals.items(), key=lambda kv: kv[1]["hours"], reverse=True)
                top = seq_items[:5]
                others = seq_items[5:]
                others_hours = sum(v["hours"] for _, v in others)
                others_cost = sum(v["cost"] for _, v in others)
                for _, info in top:
                    share = (info["hours"] / total_hours * 100.0) if total_hours > 0 else 0.0
                    rows.append([info["name"], "", "", "", f"{info['hours']:.1f}", f"{share:.0f}%", f"${info['cost']:,.2f}"])
                if others:
                    share = (others_hours / total_hours * 100.0) if total_hours > 0 else 0.0
                    rows.append(["Others", "", "", "", f"{others_hours:.1f}", f"{share:.0f}%", f"${others_cost:,.2f}"])
                rows.append(["", "", "", "", "", "", ""])  # spacer
            except Exception:
                pass

            # Add optimized allocation table (relative to elapsed_hours)
            try:
                if elapsed_hours > 0:
                    rows.append(["", "", "", "", "", "", ""])  # spacer
                    rows.append([f"After Optimization (total task hours {total_hours:.1f}h)", "", "", "", "", "", ""])
                    rows.append(["Resource", "", "", "", f"Hours (window {elapsed_hours:.0f}h)", "Share", "Cost"])

                    # Compute hours per resource and normalized share of the window
                    per_res_hours: Dict[str, float] = {}
                    for res_id, entries in by_res.items():
                        hours = 0.0
                        for e in entries:
                            if hasattr(e, 'start_hour') and hasattr(e, 'end_hour'):
                                hours += e.end_hour - e.start_hour
                            else:
                                task = process.get_task_by_id(e.task_id)
                                hours += task.duration_hours if task else 0.0
                        per_res_hours[res_id] = hours
                    total_active_hours = sum(per_res_hours.values()) or 0.0

                    # Build per-resource optimized window allocations
                    optimized_rows = []
                    for res_id, entries in by_res.items():
                        res = process.get_resource_by_id(res_id)
                        res_name = res.name if res else res_id
                        rate = float(getattr(res, 'hourly_rate', 0.0) or 0.0) if res else 0.0
                        active_hours = per_res_hours.get(res_id, 0.0)
                        # Normalized window hours (sum across resources == elapsed_hours)
                        window_hours = (active_hours / total_active_hours * elapsed_hours) if total_active_hours > 0 else 0.0
                        cost_share = window_hours * rate
                        optimized_rows.append((res_name, window_hours, cost_share))
                    # Sort desc by hours and show top 5 + Others
                    optimized_rows.sort(key=lambda x: x[1], reverse=True)
                    top_opt = optimized_rows[:5]
                    others_opt = optimized_rows[5:]
                    others_opt_h = sum(h for _, h, _ in others_opt)
                    others_opt_c = sum(c for _, _, c in others_opt)
                    for name, h, c in top_opt:
                        share = (h / elapsed_hours * 100.0) if elapsed_hours > 0 else 0.0
                        rows.append([name, "", "", "", f"{h:.1f}", f"{share:.0f}%", f"${c:,.2f}"])
                    if others_opt:
                        share = (others_opt_h / elapsed_hours * 100.0) if elapsed_hours > 0 else 0.0
                        rows.append(["Others", "", "", "", f"{others_opt_h:.1f}", f"{share:.0f}%", f"${others_opt_c:,.2f}"])
                    rows.append(["Optimized Totals", "", "", "", f"{elapsed_hours:.1f}", "100%", "—"])
                    rows.append(["", "", "", "", "", "", ""])  # spacer
            except Exception:
                # Fail silently; image generation should not break due to this table
                pass

            # Add parallel execution explanation
            try:
                rows.append(["Parallel Execution (what and why)", "", "", "", "", "", ""])
                if par_groups:
                    # Build reasons based on no-dependency, different resources, and overlap
                    def _has_dependency(a: str, b: str) -> bool:
                        ta = process.get_task_by_id(a)
                        tb = process.get_task_by_id(b)
                        deps_a = set(getattr(ta, 'dependencies', []) or []) if ta else set()
                        deps_b = set(getattr(tb, 'dependencies', []) or []) if tb else set()
                        return (b in deps_a) or (a in deps_b)
                    entry_by_tid = {e.task_id: e for e in schedule.entries}
                    for idx, grp in enumerate(par_groups[:3], 1):  # limit to top 3 groups for readability
                        # Reason parts
                        reason = []
                        # 1) Dependency
                        dep_exists = any(_has_dependency(a, b) for i, a in enumerate(grp) for b in grp[i+1:])
                        reason.append("no direct dependencies" if not dep_exists else "compatible ordering")
                        # 2) Resources
                        res_set = {getattr(entry_by_tid.get(tid), 'resource_id', None) for tid in grp}
                        if len([r for r in res_set if r]) > 1:
                            reason.append("different resources")
                        else:
                            reason.append("resource availability")
                        # 3) Overlap by schedule
                        reason.append("overlapping hours")
                        rows.append([f"Group {idx}: {' / '.join(grp)}", ", ".join(reason), "", "", "", "", ""])
                    if len(par_groups) > 3:
                        rows.append([f"(+{len(par_groups)-3} more groups)", "", "", "", "", "", ""])
                else:
                    rows.append(["No parallel tasks detected", "", "", "", "", "", ""])
                rows.append(["", "", "", "", "", "", ""])  # spacer
            except Exception:
                pass
            # In concise mode we skip bottom grand total row to reduce clutter
            if show_detailed:
                rows.append(["Grand Total (task hours)", "", "", "", f"{total_hours:.1f}", "", f"${total_cost:,.2f}"])

            # Figure size adaptive to number of rows
            n = max(len(rows), 1)
            height = max(6.0, n * 0.24 + 2.0)
            fig, ax = plt.subplots(figsize=(14, height))
            ax.axis('off')
            ax.set_title(f"Resource Allocation Summary - {process.name}", fontsize=16, fontweight='bold')

            col_labels = ["Task / Resource", "Info", "Start", "End", "Hours", "Rate", "Cost"]
            table = ax.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='left')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.2)
            try:
                table.set_zorder(1)
            except Exception:
                pass

            # Summary box with totals and key stats (overlay, in addition to top rows)
            summary_lines = [
                f"Sequential hours: {total_hours:.1f}",
                f"Optimized elapsed: {elapsed_hours:.1f}",
                f"Time saved: {(total_hours - elapsed_hours):.1f}",
                f"Cost: ${total_cost:,.2f}",
                f"Resources used: {len(resources_used)}",
                f"Tasks scheduled: {tasks_count}",
                f"Peak parallel tasks: {peak}",
                f"Date range: {date_range}",
            ]
            # Use AnchoredText for reliable visibility
            try:
                from matplotlib.offsetbox import AnchoredText
                at = AnchoredText("\n".join(summary_lines), loc='upper right', prop=dict(size=10), frameon=True, pad=0.5, borderpad=0.6)
                at.patch.set_facecolor('#f7f7f7')
                at.patch.set_edgecolor('#cccccc')
                at.set_zorder(3)
                ax.add_artist(at)
            except Exception:
                ax.text(0.99, 0.99, "\n".join(summary_lines), ha='right', va='top',
                        transform=ax.transAxes, fontsize=10, color='#333', bbox=dict(boxstyle='round', fc='#f7f7f7', ec='#cccccc'))

            # Reserve right margin for the summary box so it's not clipped
            try:
                plt.tight_layout(rect=[0.02, 0.02, 0.88, 0.98])
            except Exception:
                plt.tight_layout()
            plt.savefig(output_file, dpi=200, bbox_inches='tight')
            if show:
                try:
                    import webbrowser
                    webbrowser.open(f'file://{os.path.abspath(output_file)}')
                except Exception:
                    pass
            return output_file
        except Exception as e:
            print(f"Error generating resource allocation summary: {str(e)}")
            return None

    def plot_allocation_charts(self, process: 'Process', schedule: 'Schedule',
                               title: str = 'Resource Allocation (Before vs After)',
                               output_file: Optional[str] = None,
                               show: bool = False,
                               baseline_schedule: Optional['Schedule'] = None) -> Optional[str]:
        """
        Create a side-by-side visualization with:
          - Left: pie charts (Before vs After) of resource share
          - Right: bar charts (Before vs After) of resource hours
        And a text box below listing detected parallel task groups.

        "Before": computed from the provided baseline_schedule using task duration hours.
        If baseline_schedule is None, we fall back to using the optimized schedule (so Before == After).

        "After": computed from the optimized schedule using task duration hours.
        """
        try:
            # Compute resource allocations
            resources = process.resources
            ids = [r.id for r in resources]
            names = [r.name for r in resources]
            
            # Before: equal distribution of total task hours across all resources
            total_task_hours = sum(t.duration_hours for t in process.tasks)
            before_vals = [total_task_hours / len(resources) for _ in resources]
            
            # After: actual hours assigned to each resource in schedule
            after_vals = [0.0] * len(resources)
            resource_idx = {r.id: i for i, r in enumerate(resources)}
            
            for entry in schedule.entries:
                if entry.resource_id in resource_idx:
                    # Use task duration hours
                    task = process.get_task_by_id(entry.task_id)
                    if task:
                        after_vals[resource_idx[entry.resource_id]] += task.duration_hours
            
            # Sort by after values descending (most utilized first)
            order = sorted(range(len(after_vals)), key=lambda i: after_vals[i], reverse=True)
            ids = [ids[i] for i in order]
            names = [names[i] for i in order]
            before_vals = [before_vals[i] for i in order]
            after_vals = [after_vals[i] for i in order]
            
            # Calculate totals for summary
            before_total_hours = sum(before_vals)

            # Calculate baseline cost (Before): cheapest qualified resource per task
            before_total_cost = 0.0
            for t in process.tasks:
                # Find the cheapest qualified resource for this task
                min_rate = float('inf')
                required = getattr(t, 'required_skills', []) or []
                for r in resources:
                    try:
                        if r.has_all_skills(required):
                            min_rate = min(min_rate, float(getattr(r, 'hourly_rate', 0.0) or 0.0))
                    except Exception:
                        # Be resilient if any resource lacks method/fields
                        continue
                if min_rate == float('inf'):
                    # Fallback: use average rate if no qualified resource found
                    avg_rate = sum(float(getattr(rr, 'hourly_rate', 0.0) or 0.0) for rr in resources) / float(len(resources)) if resources else 0.0
                    min_rate = avg_rate
                before_total_cost += float(getattr(t, 'duration_hours', 0.0) or 0.0) * float(min_rate)
            
            # Calculate after cost from actual assignments
            after_total_cost = 0.0
            for entry in schedule.entries:
                resource = process.get_resource_by_id(entry.resource_id)
                task = process.get_task_by_id(entry.task_id)
                if resource and task:
                    after_total_cost += task.duration_hours * resource.hourly_rate
            
            # Calculate elapsed hours
            if schedule.entries:
                min_start = min(e.start_time for e in schedule.entries)
                max_end = max(e.end_time for e in schedule.entries)
                after_elapsed_hours = (max_end - min_start).total_seconds() / 3600.0
            else:
                after_elapsed_hours = 0.0

            # Detect parallel groups
            def _parallel_groups(entries_list: List) -> List[List[str]]:
                if not entries_list:
                    return []
                events = []
                for en in entries_list:
                    events.append((en.start_time, 1, en.task_id))
                    events.append((en.end_time, -1, en.task_id))
                events.sort(key=lambda x: (x[0], x[1]))
                active = set(); groups = []
                for _t0, typ, tid in events:
                    if typ == 1:
                        active.add(tid)
                        if len(active) >= 2:
                            groups.append(tuple(sorted(active)))
                    else:
                        active.discard(tid)
                uniq = []
                seen = set()
                for g in groups:
                    key = frozenset(g)
                    if key not in seen and len(key) >= 2:
                        seen.add(key)
                        uniq.append(list(key))
                return uniq

            def _tname(tid: str) -> str:
                t = process.get_task_by_id(tid)
                return t.name if t else tid
            par_groups = [[_tname(tid) for tid in grp] for grp in _parallel_groups(schedule.entries)]

            # Figure layout with expanded width for better space utilization
            fig = plt.figure(figsize=(24, 12))  # Increased width to 24 for more space
            gs = GridSpec(3, 3, height_ratios=[3, 1.2, 2], width_ratios=[1, 1.2, 0.8])  # 3 columns for better layout
            ax_pies = fig.add_subplot(gs[0, 0])
            ax_bars = fig.add_subplot(gs[0, 1:])
            ax_text = fig.add_subplot(gs[1, :])
            ax_taskmap = fig.add_subplot(gs[2, :])
            fig.suptitle(title, fontsize=14)

            # Pie charts (Resource Utilization)
            def _pct_fmt(pct):
                # Hide tiny percentages introduced by epsilon placeholders
                return f"{pct:.1f}%" if pct >= 0.5 else ''
            ax_pies.set_title("Resource Utilization (Task Hours Distribution)")
            import numpy as np
            # Use a single consistent palette so inner (After) colors align with outer (Before)
            base_colors = plt.get_cmap('tab20')(np.linspace(0.05, 0.95, len(names)))
            outer_colors = base_colors
            # Use EXACT same colors for inner ring to align hues between rings
            inner_colors = base_colors
            # To keep color alignment across rings, ensure both pies have the SAME number of wedges.
            # Matplotlib may drop zero-size slices, which would shift colors. We guard by replacing
            # zero values in the inner ring with a tiny epsilon so wedges are retained.
            eps = 1e-6
            after_vals_for_inner = [v if v > 0 else eps for v in after_vals]

            # First draw Inner ring = AFTER (actual task hours assigned)
            wedges2, texts2, autotexts2 = ax_pies.pie(
                after_vals_for_inner, labels=None, autopct=_pct_fmt, radius=0.65, labeldistance=0.55, pctdistance=0.72,
                colors=inner_colors, startangle=90, counterclock=False,
                wedgeprops=dict(edgecolor='white', linewidth=0.8, width=0.35))

            # Now draw the Outer ring so each sector aligns angularly with the inner ring
            from matplotlib.patches import Wedge
            wedges1, texts1, autotexts1 = [], [], []
            equal_pct = 100.0 / max(1, len(names))
            for i, w in enumerate(wedges2):
                theta1, theta2 = w.theta1, w.theta2
                # Draw aligned outer wedge
                ww = Wedge(center=(0, 0), r=1.0, theta1=theta1, theta2=theta2, width=0.35,
                           facecolor=outer_colors[i], edgecolor='white', linewidth=0.8)
                ax_pies.add_patch(ww)
                wedges1.append(ww)
                # REMOVED: Resource names were causing overlap
                texts1.append(None)
                # Place equal-split percentage on outer ring
                ang = np.deg2rad((theta1 + theta2) / 2.0)
                px, py = np.cos(ang) * 0.88, np.sin(ang) * 0.88
                t_pct = ax_pies.text(px, py, _pct_fmt(equal_pct), ha='center', va='center', fontsize=9, color='white')
                autotexts1.append(t_pct)
            # Improve percentage label readability
            try:
                import matplotlib.patheffects as pe
                for t in list(autotexts1) + list(autotexts2):
                    t.set_fontsize(9)
                    t.set_color('white')
                    t.set_path_effects([pe.withStroke(linewidth=2, foreground='black', alpha=0.6)])
            except Exception:
                pass
            # Create a proper legend showing resource names with colors
            legend_items = []
            legend_labels = []
            for i, name in enumerate(names):
                legend_items.append(wedges1[i])
                after_hours = after_vals[i]
                legend_labels.append(f"{name} ({after_hours:.1f}h)")
            
            # Place legend below the pie chart to avoid overlap
            ax_pies.legend(legend_items, legend_labels, 
                         loc='upper center', bbox_to_anchor=(0.5, -0.1),
                         ncol=2, frameon=False, fontsize=8)

            # Bar charts (Resource Time Comparison)
            ax_bars.set_title("Resource Hours: Task Hours (Sequential) vs Working Window (Actual Parallel)")
            
            # Calculate before values (task hours per resource)
            # These are already in after_vals from task assignments
            
            # Calculate after values as ACTUAL working hours per resource (excluding gaps)
            # This sums up the actual calendar time each resource is working
            resource_windows: Dict[str, float] = {}
            for r in resources:
                res_entries = [e for e in schedule.entries if e.resource_id == r.id]
                if res_entries:
                    # Sort entries by start time
                    sorted_entries = sorted(res_entries, key=lambda e: e.start_time)
                    # Merge overlapping time periods and sum actual working hours
                    working_hours = 0.0
                    current_start = sorted_entries[0].start_time
                    current_end = sorted_entries[0].end_time
                    
                    for entry in sorted_entries[1:]:
                        if entry.start_time <= current_end:
                            # Overlapping or adjacent, extend the period
                            current_end = max(current_end, entry.end_time)
                        else:
                            # Gap found, add the previous period and start a new one
                            working_hours += (current_end - current_start).total_seconds() / 3600.0
                            current_start = entry.start_time
                            current_end = entry.end_time
                    
                    # Add the last period
                    working_hours += (current_end - current_start).total_seconds() / 3600.0
                    resource_windows[r.id] = working_hours
                else:
                    resource_windows[r.id] = 0.0
            # Align order to sorted ids
            after_elapsed_vals = [resource_windows.get(rid, 0.0) for rid in ids]
            
            # Create grouped bar chart
            import numpy as np
            x = np.arange(len(names))
            width = 0.38
            before_color = '#e74c3c'  # red for sequential
            after_color = '#27ae60'   # green for parallel
            
            # Before bars show task hours (sum of task durations)
            before_bars = ax_bars.bar(x - width/2, after_vals, width, 
                                     label='Task Hours (Sum of Durations)', 
                                     color=before_color, alpha=0.8)
            # After bars show actual calendar working time (excluding gaps)
            after_bars = ax_bars.bar(x + width/2, after_elapsed_vals, width,
                                   label='Calendar Working Time (Actual)', 
                                   color=after_color, alpha=0.8)
            
            ax_bars.set_xlabel('Resources')
            ax_bars.set_ylabel('Hours')
            ax_bars.set_xticks(x)
            ax_bars.set_xticklabels(names, rotation=30, ha='right')
            ax_bars.legend(loc='upper right')
            ax_bars.grid(axis='y', linestyle='--', alpha=0.4)
            
            # Add value labels on bars
            for bar, val in zip(before_bars, after_vals):
                if val > 0:
                    height = bar.get_height()
                    ax_bars.text(bar.get_x() + bar.get_width()/2., height + 1,
                               f'{val:.0f}h', ha='center', va='bottom',
                               fontsize=8, color=before_color)
            
            for bar, val in zip(after_bars, after_elapsed_vals):
                if val > 0:
                    height = bar.get_height()
                    ax_bars.text(bar.get_x() + bar.get_width()/2., height + 1,
                               f'{val:.0f}h', ha='center', va='bottom',
                               fontsize=8, color=after_color)
            
            # Calculate totals for annotation
            total_task_hours = sum(after_vals)
            max_window = max(after_elapsed_vals) if after_elapsed_vals else 0.0
            
            # Add summary annotation on the right side of the chart
            ax_bars.text(1.02, 0.98,
                       f'• Red bars: Sum of task durations ({total_task_hours:.0f}h total)\n'
                       f'• Green bars: Actual working time\n'
                       f'• Process elapsed: {after_elapsed_hours:.0f}h\n'
                       f'• Time saved: {total_task_hours - after_elapsed_hours:.0f}h',
                       transform=ax_bars.transAxes, ha='left', va='top',
                       fontsize=9, bbox=dict(boxstyle='round,pad=0.3', 
                                          facecolor='#ffffcc', alpha=0.8,
                                          edgecolor='#dddd88', linewidth=0.5))

            # Task-to-Resource mapping (expanded to utilize full horizontal space)
            ax_taskmap.clear()
            ax_taskmap.set_title('Task Assignment (After Schedule)', pad=20, fontsize=14)
            
            # Calculate task hours per resource
            task_hours: Dict[Tuple[str, str], float] = {}
            for e in schedule.entries:
                task = process.get_task_by_id(e.task_id)
                if task:
                    # Use task's duration_hours to match pie chart calculation
                    dur = task.duration_hours
                else:
                    dur = 0.0
                # Accumulate task hours per resource-task pair
                if dur > 0.0:
                    task_hours[(e.resource_id, e.task_id)] = task_hours.get((e.resource_id, e.task_id), 0.0) + dur

            import numpy as _np
            ax_taskmap.set_title("Resource → Task assignments (After schedule)")
            ax_taskmap.axis('off')
            
            # Helper function to get resource name from ID
            def _rname(resource_id: str) -> str:
                r = process.get_resource_by_id(resource_id)
                return r.name if r else resource_id
            
            # Build per-assignment rows sorted by resource order and hours
            assignments = []  # (resource_name, task_name, hours)
            for (rid, tid), hrs in sorted(task_hours.items(), key=lambda kv: (ids.index(kv[0][0]) if kv[0][0] in ids else 1e9, -kv[1])):
                t = process.get_task_by_id(tid)
                # Only add if hours > 0 to avoid clutter
                if hrs > 0:
                    assignments.append((_rname(rid), t.name if t else tid, hrs))

            if assignments:
                n = len(assignments)
                y_vals = _np.linspace(0.95, 0.05, n)  # Use more vertical space
            else:
                y_vals = _np.array([])

            # Expand to use full horizontal space
            x_left, x_right = 0.02, 0.98
            # Place labels with better formatting and spacing
            left_texts = []
            right_texts = []
            for y, (rname, tname, hrs) in zip(y_vals, assignments):
                # Enhanced resource labels with background
                lt = ax_taskmap.text(x_left, y, f'🔧 {rname}', va='center', ha='left', fontsize=10, 
                                   transform=ax_taskmap.transAxes, weight='bold',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f4fd', alpha=0.7))
                # Enhanced task labels with background
                rt = ax_taskmap.text(x_right, y, f'{tname} ({hrs:.1f}h)', va='center', ha='right', fontsize=10,
                                   transform=ax_taskmap.transAxes, weight='bold',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff2e8', alpha=0.7))
                left_texts.append(lt)
                right_texts.append(rt)

            # Render to compute text bounds
            fig.canvas.draw()
            inv = ax_taskmap.transAxes.inverted()
            pad = 0.01

            for y, (_rname_str, _tname_str, hrs), lt, rt in zip(y_vals, assignments, left_texts, right_texts):
                # Convert text bounding boxes to axes fraction coordinates
                lt_bb = lt.get_window_extent(renderer=fig.canvas.get_renderer())
                rt_bb = rt.get_window_extent(renderer=fig.canvas.get_renderer())
                lt_x1 = inv.transform((lt_bb.x1, lt_bb.y1))[0]
                rt_x0 = inv.transform((rt_bb.x0, rt_bb.y0))[0]
                start_x = min(max(lt_x1 + pad, x_left + pad), x_right - 2*pad)
                end_x = max(min(rt_x0 - pad, x_right - pad), start_x + 0.02)
                # Enhanced center label with better styling
                mx = (start_x + end_x) / 2
                ttxt = ax_taskmap.text(mx, y, f"⏱️ {hrs:.1f}h", fontsize=9, color='#2c3e50', ha='center', va='center', 
                                     transform=ax_taskmap.transAxes, weight='bold',
                                     bbox=dict(boxstyle='round,pad=0.2', facecolor='#f8f9fa', edgecolor='#dee2e6'))
                fig.canvas.draw()
                t_bb = ttxt.get_window_extent(renderer=fig.canvas.get_renderer())
                t_x0 = inv.transform((t_bb.x0, t_bb.y0))[0]
                t_x1 = inv.transform((t_bb.x1, t_bb.y1))[0]
                gap_l = max(start_x, t_x0 - pad)
                gap_r = min(end_x,   t_x1 + pad)
                # Enhanced arrow styling with gradient effect
                if gap_l > start_x:
                    ax_taskmap.plot([start_x, gap_l], [y, y], color='#3498db', lw=2.0, transform=ax_taskmap.transAxes, alpha=0.8)
                # Draw right segment with enhanced arrow head
                if end_x > gap_r:
                    from matplotlib.patches import FancyArrowPatch
                    arr = FancyArrowPatch((gap_r, y), (end_x, y),
                                          arrowstyle='-|>', mutation_scale=15,
                                          linewidth=2.0, color='#3498db', alpha=0.8,
                                          transform=ax_taskmap.transAxes)
                    ax_taskmap.add_patch(arr)
            # Remove old annotation code - values are now shown inside bars

            # Parallel groups chart (expanded to utilize more horizontal space)
            import numpy as _np
            from matplotlib.patches import Rectangle
            ax_text.clear()
            ax_text.set_title("Parallel steps (each column) and tasks in each step", fontsize=14, pad=20)
            G = len(par_groups)
            # Determine max stack height across groups
            max_stack = max((len(g) for g in par_groups), default=0)
            
            # Expand columns to fill more horizontal space
            if G > 0:
                col_w = min(1.2, 8.0 / G)  # Adaptive column width based on number of groups
                spacing = max(0.3, 1.0 / G)  # Adaptive spacing
                x_positions = _np.linspace(1, max(8, G * 1.5), G)  # Spread across more space
            else:
                col_w = 0.8
                x_positions = _np.array([])
                
            for j, (grp, x_pos) in enumerate(zip(par_groups, x_positions)):
                for idx, task_name in enumerate(grp):
                    y0 = idx  # stack vertically within the group
                    rect = Rectangle((x_pos - col_w/2, y0 + 0.05), col_w, 0.9,
                                     facecolor='#4e79a7', alpha=0.2, edgecolor='#4e79a7', linewidth=1.2)
                    ax_text.add_patch(rect)
                    # Wrap long task names for better display
                    wrapped_name = task_name if len(task_name) <= 20 else task_name[:17] + '...'
                    ax_text.text(x_pos, y0 + 0.5, wrapped_name, ha='center', va='center', 
                               fontsize=9, color='#2f2f2f', weight='bold')
            
            # Axes cosmetics with expanded range
            if G > 0:
                ax_text.set_xlim(0.2, max(8.8, max(x_positions) + 1))
            else:
                ax_text.set_xlim(0.2, 8.8)
            ax_text.set_ylim(-0.2, max_stack + 1.0)
            ax_text.set_xticks(x_positions)
            ax_text.set_xticklabels([f'Step {i+1}' for i in range(len(x_positions))], fontsize=10)
            ax_text.set_xlabel('Parallel step sequence', fontsize=12, labelpad=10)
            ax_text.set_yticks([])
            # Vertical gridlines at each column with better styling
            for x in x_positions:
                ax_text.axvline(x, color='#e0e0e0', linewidth=0.8, alpha=0.7)
            # Add summary box with metrics (moved to the right side of the figure)
            after_assigned_total = float(sum(after_vals))
            time_reduction = before_total_hours - after_elapsed_hours
            time_reduction_pct = (time_reduction / before_total_hours * 100) if before_total_hours > 0 else 0
            
            # Create a text box on the right side of the figure
            summary_text = (
                f"📊 Performance Summary\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"• Duration: {before_total_hours:.0f}h → {after_elapsed_hours:.0f}h\n"
                f"• Time Saved: {time_reduction:.0f}h ({time_reduction_pct:.1f}%)\n"
                f"• Cost: ${before_total_cost:,.0f} → ${after_total_cost:,.0f}\n"
                f"• Parallel Groups: {len(par_groups)}"
            )
            
            # Add text to the right side of the parallel steps chart
            ax_text.text(1.02, 0.98, summary_text, ha='left', va='top', fontsize=9,
                       transform=ax_text.transAxes,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', 
                               edgecolor='#dee2e6', alpha=0.9))
            
            # Parallel task groups summary has been removed as requested

            fig.tight_layout(rect=[0, 0, 1, 0.97])
            if output_file:
                out = output_file if output_file.lower().endswith('.png') else f"{output_file}.png"
                plt.savefig(out, dpi=140, bbox_inches='tight')
                plt.close(fig)
                return out
            else:
                if show:
                    plt.show()
                return None
        except Exception as e:
            print(f"Error generating allocation charts: {e}")
            return None

    def plot_whatif_allocations(self, wi_results: Dict[str, Any], process: 'Process',
                                title: str = 'What-If Scenario Allocations',
                                output_file: Optional[str] = None,
                                show: bool = False) -> Optional[str]:
        """
        Render a grid of Task (rows) vs Scenarios (columns) showing assigned resource names.
        Includes a Baseline column and highlights the best scenario column.

        Args:
            wi_results: dict from WhatIfAnalyzer.analyze_scenarios including baseline_schedule and scenario schedules
            process: Process object (for task/resource names)
            title: chart title
            output_file: path for PNG (with or without .png)
            show: whether to display interactively

        Returns: path to PNG or None
        """
        # Disabled by request: do not generate What-If allocations PNGs
        return None
        try:
            import matplotlib.pyplot as plt
            import os

            if not wi_results:
                return None

            scenarios = wi_results.get('scenarios', {})
            best = wi_results.get('best_scenario') or {}
            best_id = best.get('id')

            # Build ordered columns: Baseline + scenarios by key order
            columns = [('baseline', 'Baseline', wi_results.get('baseline_schedule'))]
            for sid, data in scenarios.items():
                columns.append((sid, data.get('name', sid), data.get('schedule')))

            # Gather rows for tasks in process order
            headers = ['Task'] + [name for (_, name, _) in columns]
            table_rows = []
            for task in process.tasks:
                row = [task.name]
                for sid, _, sched in columns:
                    cell = ''
                    if sched:
                        entry = sched.get_task_schedule(task.id)
                        if entry:
                            res = process.get_resource_by_id(entry.resource_id)
                            cell = res.name if res else entry.resource_id
                    row.append(cell)
                table_rows.append(row)

            # Figure and table
            nrows = len(table_rows) + 1
            ncols = len(headers)
            fig_w = max(8, 2 + 1.8 * ncols)
            fig_h = min(20, 1.5 + 0.5 * nrows)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            ax.axis('off')
            ax.set_title(title, fontsize=12, pad=10)

            table = ax.table(cellText=table_rows, colLabels=headers, loc='center', cellLoc='left', colLoc='left')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.2)

            # Highlight best scenario column
            if best_id is not None:
                # find column index by id
                best_col_idx = None
                for idx, (sid, _name, _sched) in enumerate(columns):
                    if sid == best_id:
                        best_col_idx = idx + 1  # +1 to account for Task column at 0
                        break
                if best_col_idx is not None:
                    # Header cell
                    header_key = (0, best_col_idx)
                    if header_key in table._cells:
                        table._cells[header_key].set_facecolor('#e8f8f5')
                    # Data cells
                    for r in range(1, nrows):
                        key = (r, best_col_idx)
                        if key in table._cells:
                            table._cells[key].set_facecolor('#e8f8f5')

            plt.tight_layout()
            if output_file:
                out = output_file if output_file.lower().endswith('.png') else f"{output_file}.png"
                plt.savefig(out, dpi=140, bbox_inches='tight')
                if not show:
                    plt.close(fig)
                return out
            else:
                if show:
                    plt.show()
                return None
        except Exception as e:
            # Disabled; keep silent/return None
            return None
    
    def create_resource_utilization_chart(self, process: Process, schedule: Schedule,
                                        save_path: Optional[str] = None) -> plt.Figure:
        """Create resource utilization chart"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        if not schedule.resource_utilization:
            ax1.text(0.5, 0.5, 'No resource utilization data', ha='center', va='center',
                    transform=ax1.transAxes, fontsize=16)
            ax2.text(0.5, 0.5, 'No resource utilization data', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=16)
            return fig
        
        # Prepare data
        resources = []
        utilizations = []
        idle_times = []
        
        for resource in process.resources:
            resources.append(resource.name)
            util = schedule.resource_utilization.get(resource.id, 0)
            utilizations.append(util)
            idle = schedule.idle_resources.get(resource.id, 0)
            idle_times.append(idle)
        
        # Utilization bar chart
        bars1 = ax1.bar(resources, utilizations, color=self.colors['primary'], alpha=0.7)
        ax1.set_ylabel('Utilization (%)', fontsize=12)
        ax1.set_title('Resource Utilization', fontsize=14, weight='bold')
        ax1.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, util in zip(bars1, utilizations):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{util:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # Add horizontal line at 80% (good utilization target)
        ax1.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Target (80%)')
        ax1.legend()
        
        # Idle time chart
        bars2 = ax2.bar(resources, idle_times, color=self.colors['warning'], alpha=0.7)
        ax2.set_ylabel('Idle Time (hours)', fontsize=12)
        ax2.set_title('Resource Idle Time', fontsize=14, weight='bold')
        
        # Add value labels on bars
        for bar, idle in zip(bars2, idle_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{idle:.1f}h', ha='center', va='bottom', fontsize=10)
        
        # Rotate x-axis labels if needed
        if len(resources) > 5:
            ax1.tick_params(axis='x', rotation=45)
            ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_cost_breakdown_chart(self, process: Process, schedule: Schedule,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """Create cost breakdown visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        if not schedule.entries:
            ax1.text(0.5, 0.5, 'No cost data available', ha='center', va='center',
                    transform=ax1.transAxes, fontsize=16)
            ax2.text(0.5, 0.5, 'No cost data available', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=16)
            return fig
        
        # Cost by resource
        resource_costs = {}
        for entry in schedule.entries:
            resource = process.get_resource_by_id(entry.resource_id)
            if resource:
                resource_costs[resource.name] = resource_costs.get(resource.name, 0) + entry.cost
        
        # Pie chart for resource costs
        if resource_costs:
            # Sort resources by cost in descending order
            sorted_resources = sorted(resource_costs.items(), key=lambda x: x[1], reverse=True)
            labels = [f"{k} (${v:,.2f})" for k, v in sorted_resources]
            sizes = [v for k, v in sorted_resources]
            
            # Create figure with massive vertical space
            fig = plt.gcf()
            fig.clear()
            fig.set_size_inches(12, 18)  # Even taller figure
            
            # Create THREE distinct zones with clear separation
            
            # ZONE 1: Title area (top 20% of figure)
            title_y = 0.92
            fig.text(0.5, title_y, 'Resource Utilization', 
                    ha='center', va='center', fontsize=24, weight='bold')
            
            subtitle_y = 0.88
            fig.text(0.5, subtitle_y, 'Task Hours Distribution',
                    ha='center', va='center', fontsize=18, color='gray')
            
            # ZONE 2: Pie chart (middle 40% of figure, y: 0.30-0.70)
            ax1 = fig.add_axes([0.1, 0.30, 0.8, 0.40])  # [left, bottom, width, height]
            
            # Create clean donut chart WITHOUT any labels
            wedges, texts, autotexts = ax1.pie(
                sizes,
                labels=None,  # NO labels on pie
                autopct='',  # NO percentages on pie slices either
                startangle=90,
                colors=plt.cm.Set3.colors[:len(sizes)],
                wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
                labeldistance=None  # Explicitly disable label distance
            )
            
            # Remove any text elements that might have been created
            for text in texts:
                text.set_visible(False)
            for text in autotexts:
                text.set_visible(False)
            
            ax1.axis('equal')
            
            # ZONE 3: Legend area (bottom 25% of figure)
            legend_title_y = 0.22
            fig.text(0.5, legend_title_y, 'Resources', 
                    ha='center', va='center', fontsize=16, weight='bold')
            
            # Create a clean table-like legend layout
            legend_base_y = 0.17
            num_items = len(labels)
            
            if num_items <= 3:
                # Single column for few items
                for i, (label, wedge, size) in enumerate(zip(labels, wedges, sizes)):
                    pct = size / sum(sizes) * 100
                    y = legend_base_y - i * 0.04
                    
                    # Color marker
                    fig.text(0.35, y, '●', 
                            color=wedge.get_facecolor(), fontsize=16, ha='right', va='center')
                    # Label text
                    fig.text(0.37, y, f"{label} ({pct:.1f}%)", 
                            fontsize=11, va='center')
            else:
                # Two columns for many items
                cols = 2
                for i, (label, wedge, size) in enumerate(zip(labels, wedges, sizes)):
                    pct = size / sum(sizes) * 100
                    col = i % cols
                    row = i // cols
                    
                    x_base = 0.25 if col == 0 else 0.55
                    y = legend_base_y - row * 0.04
                    
                    # Color marker
                    fig.text(x_base, y, '●', 
                            color=wedge.get_facecolor(), fontsize=14, ha='right', va='center')
                    # Label text  
                    fig.text(x_base + 0.02, y, f"{label} ({pct:.1f}%)", 
                            fontsize=10, va='center')
        
        # Cost by task type
        task_type_costs = {}
        for entry in schedule.entries:
            task = process.get_task_by_id(entry.task_id)
            if task:
                task_type = self._get_task_type(task.name)
                task_type_costs[task_type] = task_type_costs.get(task_type, 0) + entry.cost
        
        # Bar chart for task type costs
        if task_type_costs:
            types = list(task_type_costs.keys())
            costs = list(task_type_costs.values())
            
            bars = ax2.bar(types, costs, color=self.colors['primary'], alpha=0.7)
            ax2.set_ylabel('Cost ($)', fontsize=12)
            ax2.set_title('Cost by Task Type', fontsize=14, weight='bold')
            
            # Add value labels
            for bar, cost in zip(bars, costs):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(costs)*0.01,
                       f'${cost:.0f}', ha='center', va='bottom', fontsize=10)
            
            if len(types) > 5:
                ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
        
        
        
    
    
    
    def create_optimization_metrics_dashboard(self, process: Process, schedule: Schedule,
                                              optimization_history: Optional[List[Dict]] = None,
                                              save_path: Optional[str] = None) -> plt.Figure:
        """Create a compact metrics dashboard safely (repaired implementation)."""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_key_metrics(ax1, schedule)

        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_utilization_summary(ax2, process, schedule)

        ax3 = fig.add_subplot(gs[1, :])
        self._plot_timeline_overview(ax3, process, schedule)

        ax4 = fig.add_subplot(gs[2, :2])
        self._plot_cost_analysis(ax4, process, schedule)

        ax5 = fig.add_subplot(gs[2, 2:])
        if optimization_history:
            self._plot_optimization_progress(ax5, optimization_history)
        else:
            ax5.text(0.5, 0.5, 'No optimization history available',
                     ha='center', va='center', transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Optimization Progress', fontsize=12, weight='bold')

        plt.suptitle(f'Process Optimization Dashboard - {process.name}',
                     fontsize=18, weight='bold', y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig
    
    def _get_task_levels(self, tasks: List[Task]) -> Dict[str, int]:
        """Get dependency levels for tasks"""
        levels = {}
        
        def get_level(task_id: str, visited: set) -> int:
            if task_id in visited:
                return 0  # Circular dependency
            if task_id in levels:
                return levels[task_id]
            
            visited.add(task_id)
            task = next((t for t in tasks if t.id == task_id), None)
            
            if not task or not task.dependencies:
                levels[task_id] = 0
                return 0
            
            max_dep_level = max(get_level(dep_id, visited.copy()) 
                              for dep_id in task.dependencies)
            levels[task_id] = max_dep_level + 1
            return levels[task_id]
        
        for task in tasks:
            get_level(task.id, set())
        
        return levels
    
    def _plot_key_metrics(self, ax, schedule: Schedule):
        """Plot key metrics summary"""
        metrics = [
            ('Duration', f'{schedule.total_duration_hours:.1f}h', self.colors['primary']),
            ('Cost', f'${schedule.total_cost:.0f}', self.colors['secondary']),
            ('Tasks', f'{len(schedule.entries)}', self.colors['accent']),
            ('Deadlocks', f'{len(schedule.deadlocks_detected)}', self.colors['warning'])
        ]
        
        for i, (label, value, color) in enumerate(metrics):
            ax.text(i * 0.25, 0.6, value, ha='center', va='center', 
                   fontsize=20, weight='bold', color=color)
            ax.text(i * 0.25, 0.3, label, ha='center', va='center', 
                   fontsize=12, color='gray')
        
        ax.set_xlim(-0.1, 0.9)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Key Metrics', fontsize=14, weight='bold')
    
    def _plot_utilization_summary(self, ax, process: Process, schedule: Schedule):
        """Plot resource utilization summary"""
        if schedule.resource_utilization:
            utilizations = list(schedule.resource_utilization.values())
            avg_util = np.mean(utilizations)
            
            ax.hist(utilizations, bins=10, alpha=0.7, color=self.colors['info'])
            ax.axvline(avg_util, color='red', linestyle='--', 
                      label=f'Average: {avg_util:.1f}%')
            ax.set_xlabel('Utilization (%)')
            ax.set_ylabel('Resources')
            ax.set_title('Resource Utilization Distribution', fontsize=12, weight='bold')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No utilization data', ha='center', va='center',
                   transform=ax.transAxes)
    
    def _plot_timeline_overview(self, ax, process: Process, schedule: Schedule):
        """Plot simplified timeline overview"""
        if not schedule.entries:
            ax.text(0.5, 0.5, 'No timeline data', ha='center', va='center',
                   transform=ax.transAxes)
            return
        
        # Group by day
        daily_tasks = {}
        for entry in schedule.entries:
            day = entry.start_time.date()
            if day not in daily_tasks:
                daily_tasks[day] = 0
            daily_tasks[day] += 1
        
        if daily_tasks:
            days = sorted(daily_tasks.keys())
            counts = [daily_tasks[day] for day in days]
            
            ax.bar(range(len(days)), counts, color=self.colors['primary'], alpha=0.7)
            ax.set_xlabel('Days')
            ax.set_ylabel('Active Tasks')
            ax.set_title('Daily Task Activity', fontsize=12, weight='bold')
            
            # Set x-axis labels
            if len(days) <= 10:
                ax.set_xticks(range(len(days)))
                ax.set_xticklabels([d.strftime('%m-%d') for d in days], rotation=45)
    
    def _plot_cost_analysis(self, ax, process: Process, schedule: Schedule):
        """Plot cost analysis"""
        if not schedule.entries:
            ax.text(0.5, 0.5, 'No cost data', ha='center', va='center',
                   transform=ax.transAxes)
            return
        
        # Cost by resource
        resource_costs = {}
        for entry in schedule.entries:
            resource = process.get_resource_by_id(entry.resource_id)
            if resource:
                name = resource.name
                resource_costs[name] = resource_costs.get(name, 0) + entry.cost
        
        if resource_costs:
            names = list(resource_costs.keys())
            costs = list(resource_costs.values())
            
            bars = ax.bar(names, costs, color=self.colors['secondary'], alpha=0.7)
            ax.set_ylabel('Cost ($)')
            ax.set_title('Cost by Resource', fontsize=12, weight='bold')
            
            if len(names) > 5:
                ax.tick_params(axis='x', rotation=45)
    
    def _plot_optimization_progress(self, ax, history: List[Dict]):
        """Plot optimization progress over iterations"""
        iterations = range(len(history))
        durations = [h.get('duration', 0) for h in history]
        costs = [h.get('cost', 0) for h in history]
        
        ax2 = ax.twinx()
        
        line1 = ax.plot(iterations, durations, 'b-', label='Duration (h)')
        line2 = ax2.plot(iterations, costs, 'r-', label='Cost ($)')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Duration (hours)', color='b')
        ax2.set_ylabel('Cost ($)', color='r')
        ax.set_title('Optimization Progress', fontsize=12, weight='bold')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right')
    
    def save_all_charts(self, process: Process, schedule: Schedule, 
                       output_dir: str = "output") -> Dict[str, str]:
        """Generate and save all visualization charts"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        # Gantt chart
        gantt_path = os.path.join(output_dir, f"{process.id}_gantt.png")
        self.create_gantt_chart(process, schedule, gantt_path)
        saved_files['gantt'] = gantt_path
        
        # Resource utilization
        util_path = os.path.join(output_dir, f"{process.id}_utilization.png")
        self.create_resource_utilization_chart(process, schedule, util_path)
        saved_files['utilization'] = util_path
        
        # Cost breakdown
        cost_path = os.path.join(output_dir, f"{process.id}_costs.png")
        self.create_cost_breakdown_chart(process, schedule, cost_path)
        saved_files['costs'] = cost_path
        
        # Critical path
        critical_path = os.path.join(output_dir, f"{process.id}_critical_path.png")
        self.create_critical_path_chart(process, schedule, critical_path)
        saved_files['critical_path'] = critical_path
        
        # Dashboard
        dashboard_path = os.path.join(output_dir, f"{process.id}_dashboard.png")
        self.create_optimization_metrics_dashboard(process, schedule, None, dashboard_path)
        saved_files['dashboard'] = dashboard_path
        
        return saved_files
        
    def _get_task_type(self, task_name: str) -> str:
        """Determine task type based on name"""
        name_lower = task_name.lower()
        if any(x in name_lower for x in ['test', 'qa', 'quality']):
            return 'testing'
        elif any(x in name_lower for x in ['design', 'ui', 'ux']):
            return 'design'
        elif any(x in name_lower for x in ['review', 'audit']):
            return 'review'
        elif any(x in name_lower for x in ['doc', 'manual', 'guide']):
            return 'documentation'
        elif any(x in name_lower for x in ['deploy', 'release', 'production']):
            return 'deployment'
        elif any(x in name_lower for x in ['dev', 'implement', 'code', 'program']):
            return 'development'
        return 'other'
        
    def plot_schedule_gantt(self, schedule: Schedule, process: Process, title: str = 'Project Gantt Chart', 
                          output_file: str = None, show: bool = True) -> str:
        """
        Generate an interactive Gantt chart of the schedule using Graphviz
        
        Args:
            schedule: The schedule to visualize
            process: The process containing task and resource information
            title: Chart title
            output_file: Path to save the output file (without extension)
            show: Whether to open the generated visualization
            
        Returns:
            str: Path to the generated visualization file or None if failed
        """
        # Disabled by request: do not generate Gantt charts
        return None
        try:
            import os
            import tempfile
            import webbrowser
            
            # Try to import graphviz and set the path to Graphviz binaries
            try:
                import graphviz
                
                # Set the path to Graphviz binaries if not in system PATH
                if os.path.exists('D:\\Graphviz\\Graphviz\\bin'):
                    os.environ['PATH'] += os.pathsep + 'D:\\Graphviz\\Graphviz\\bin'
                
                # Test if dot command is available
                try:
                    from graphviz import Graph
                    g = Graph()
                    g.format = 'svg'
                    g.render(view=False, cleanup=True)
                except Exception as e:
                    print("Warning: Graphviz 'dot' command not found. Please ensure Graphviz is installed and in your PATH.")
                    print("You can download it from: https://graphviz.org/download/")
                    print(f"Error details: {str(e)}")
                    return None
                
                from graphviz import Digraph
                
            except ImportError:
                print("Warning: graphviz Python package not installed. Please install it with: pip install graphviz")
                return None
            
            # Create a directed graph with explicit format
            dot = Digraph(comment=title, format='svg')

            # Improve readability: left-to-right layout, comfortable spacing, sensible DPI, padding
            dot.graph_attr.update({
                'rankdir': 'LR',           # Left-to-right timeline
                'ranksep': '1.2',          # Vertical separation between ranks
                'nodesep': '0.8',          # Horizontal separation between nodes
                'fontsize': '18',
                'fontname': 'Segoe UI',
                'dpi': '110',              # Reasonable DPI for on-screen viewing
                'pad': '0.5',              # Padding around drawing
                'margin': '0.2',
                'splines': 'ortho',
                'bgcolor': 'white'
            })
            dot.node_attr.update({
                'shape': 'box',
                'style': 'rounded,filled',
                'fillcolor': 'lightgoldenrod1',
                'fontsize': '14',
                'fontname': 'Segoe UI',
                'fontcolor': 'black',
                'width': '2.8',
                'height': '1.0'
            })
            dot.edge_attr.update({
                'fontsize': '10',
                'fontname': 'Segoe UI'
            })
            dot.attr('graph', **{k: str(v) for k, v in self.graph_style['graph'].items()})
            dot.attr('node', **{k: str(v) for k, v in self.graph_style['node'].items()})
            dot.attr('edge', **{k: str(v) for k, v in self.graph_style['edge'].items()})
                
            # Add title
            dot.attr(label=title, labelloc='t', fontsize='16', fontcolor='#2c3e50')
            
            # Group entries by day
            entries_by_day = {}
            for entry in schedule.entries:
                day = entry.start_time.date()
                if day not in entries_by_day:
                    entries_by_day[day] = []
                entries_by_day[day].append(entry)
            
            # Sort days
            sorted_days = sorted(entries_by_day.keys())
            
            # Add day clusters
            for i, day in enumerate(sorted_days):
                with dot.subgraph(name=f'cluster_{i}') as c:
                    c.attr(
                        label=day.strftime('%Y-%m-%d'),
                        style='filled',
                        color='#e9ecef',
                        fontsize='12',
                        fontcolor='#495057',
                        labelloc='t',
                        rank='same'
                    )
                    
                    # Add tasks for this day
                    for entry in sorted(entries_by_day[day], key=lambda e: e.start_time):
                        task = next((t for t in process.tasks if t.id == entry.task_id), None)
                        resource = next((r for r in process.resources if r.id == entry.resource_id), None)
                        
                        if task and resource:
                            task_type = self._get_task_type(task.name)
                            duration_hours = (entry.end_time - entry.start_time).total_seconds() / 3600
                            
                            # Create node ID
                            node_id = f"{entry.task_id}_{day.strftime('%Y%m%d')}_{entry.resource_id}"
                            
                            # Simple word-wrap for task title to improve readability
                            def _wrap_text(text: str, width: int = 18) -> str:
                                parts = []
                                line = ''
                                for word in text.split(' '):
                                    if len(line) + len(word) + 1 > width:
                                        parts.append(line)
                                        line = word
                                    else:
                                        line = word if not line else f"{line} {word}"
                                if line:
                                    parts.append(line)
                                return '<br/>'.join(parts)

                            wrapped_title = _wrap_text(task.name, 18)

                            # Format node label
                            start_time_str = entry.start_time.strftime('%H:%M')
                            end_time_str = entry.end_time.strftime('%H:%M')
                            label = (
                                f'<<b>{wrapped_title}</b>'
                                f'<br/>Resource: {resource.name}'
                                f'<br/>Time: {start_time_str}-{end_time_str}'
                                f'<br/>Duration: {duration_hours:.1f} hours'
                                f'<br/>Cost: ${entry.cost:,.2f}'
                            )
                            
                            # Add node
                            c.node(
                                node_id,
                                label=label,
                                shape='box',
                                style='filled,rounded',
                                fillcolor=self.task_colors.get(task_type, '#6c757d'),
                                color='#343a40',
                                fontname='Segoe UI',
                                fontsize='12',
                                fontcolor='black',
                                height='0.3',
                                width='0.6',
                                margin='0.15,0.07'
                            )
                            
                            # Add dependencies
                            if hasattr(task, 'dependencies') and task.dependencies:
                                for dep in task.dependencies:
                                    # Find when the dependency was last scheduled
                                    dep_entries = [e for e in schedule.entries if e.task_id == dep]
                                    if dep_entries:
                                        dep_entry = max(dep_entries, key=lambda x: x.end_time)
                                        dep_day = dep_entry.end_time.date()
                                        dep_node_id = f"{dep}_{dep_day.strftime('%Y%m%d')}_{dep_entry.resource_id}"
                                        dot.edge(dep_node_id, node_id, style='dashed', color='#6c757d')
            
            # Determine output file path
            if not output_file:
                output_file = tempfile.mktemp(prefix='schedule_')
            
            # Save and render the graph
            output_path = dot.render(output_file, view=False, cleanup=True)

            # If SVG, post-process safely for responsiveness
            try:
                if output_path and output_path.lower().endswith('.svg') and os.path.exists(output_path):
                    import xml.etree.ElementTree as ET
                    # Parse XML and update root attributes
                    tree = ET.parse(output_path)
                    root = tree.getroot()
                    # Set responsive width, remove fixed height
                    root.set('width', '100%')
                    if 'height' in root.attrib:
                        del root.attrib['height']
                    # Ensure preserveAspectRatio
                    if 'preserveAspectRatio' not in root.attrib:
                        root.set('preserveAspectRatio', 'xMidYMid meet')
                    # Write back
                    tree.write(output_path, encoding='utf-8', xml_declaration=True)
            except Exception as _svg_err:
                # Non-fatal; keep original SVG
                pass

            # Open in default browser if requested
            if show and output_path and os.path.exists(output_path):
                webbrowser.open(f'file://{os.path.abspath(output_path)}')
                
            return output_path
            
        except Exception as e:
            print(f"Error generating Gantt chart: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def plot_summary_comparison(self, before: dict, after: dict, title: str,
                                output_file: str, show: bool = False) -> str:
        """
        Create a before/after summary chart comparing duration, peak people, and total cost.

        Args:
            before: dict with keys: 'duration_hours', 'peak_people', 'total_cost', 'total_resources'
            after: dict with same keys plus 'schedule' (Schedule object)
            title: chart title
            output_file: full path (with or without extension). PNG will be used if no extension
            show: whether to display interactively

        Returns:
            Path to the saved image
        """
        try:
            import os
            import matplotlib
            # Use non-interactive backend for server/headless safety
            try:
                matplotlib.use('Agg')
            except Exception:
                pass
            import matplotlib.pyplot as plt
            import numpy as np

            # Normalize output path and ensure extension
            root, ext = os.path.splitext(output_file)
            if not ext:
                output_file = root + '.png'

            # Get metrics
            dur_before = float(before.get('duration_hours', 0.0) or 0.0)
            dur_after = float(after.get('duration_hours', 0.0) or 0.0)
            
            # Calculate actual resources used in after scenario
            if 'schedule' in after and after['schedule']:
                schedule = after['schedule']
                # Get unique resource count from actual schedule
                ppl_after = len({e.resource_id for e in schedule.entries if hasattr(e, 'resource_id')})
            else:
                ppl_after = int(after.get('peak_people', 0) or 0)
                
            # For before scenario, use total resources if available, otherwise use peak
            ppl_before = int(before.get('total_resources', before.get('peak_people', 0)) or 0)
            
            cost_before = float(before.get('total_cost', 0.0) or 0.0)
            cost_after = float(after.get('total_cost', 0.0) or 0.0)

            fig = plt.figure(figsize=(12, 6))
            fig.suptitle(title, fontsize=18, fontweight='bold')

            # Create subplots with adjusted layout
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(title, fontsize=18, fontweight='bold')
            
            # Duration comparison
            bars1 = ax1.bar(['Before', 'After'],
                          [dur_before, dur_after],
                          color=['#f39c12', '#2980b9'])
            ax1.set_title('Duration (hours)', fontsize=14)
            ax1.set_ylabel('Hours', fontsize=12)
            ax1.grid(axis='y', linestyle='--', alpha=0.3)
            for rect, val in zip(bars1, [dur_before, dur_after]):
                ax1.text(rect.get_x() + rect.get_width()/2, rect.get_height()*0.05,
                       f"{val:.1f} hours", ha='center', va='bottom', 
                       fontsize=11, color='white', fontweight='bold',
                       bbox=dict(facecolor='#33333399', edgecolor='none', boxstyle='round,pad=0.2'))
            
            # Calculate time savings
            if dur_before > 0:
                time_savings = dur_before - dur_after
                time_savings_pct = (time_savings / dur_before) * 100
                ax1.text(1.5, max(dur_before, dur_after) * 0.8,
                       f"Time Saved:\n{time_savings:.1f} hours\n({time_savings_pct:.1f}%)",
                       ha='center', va='center', fontsize=11,
                       bbox=dict(facecolor='#f8f9fa', edgecolor='#dee2e6', boxstyle='round,pad=0.5'))

            # Resource comparison
            bars2 = ax2.bar(['Before', 'After'],
                          [ppl_before, ppl_after],
                          color=['#f39c12', '#2980b9'])
            ax2.set_title('Peak Resource Usage', fontsize=14)
            ax2.set_ylabel('Number of Resources', fontsize=12)
            ax2.grid(axis='y', linestyle='--', alpha=0.3)
            for rect, val in zip(bars2, [ppl_before, ppl_after]):
                ax2.text(rect.get_x() + rect.get_width()/2, rect.get_height()*0.05,
                       f"{val} resources", ha='center', va='bottom', 
                       fontsize=11, color='white', fontweight='bold',
                       bbox=dict(facecolor='#33333399', edgecolor='none', boxstyle='round,pad=0.2'))
            
            # Calculate resource efficiency
            if ppl_before > 0 and dur_before > 0:
                eff_before = dur_before / ppl_before if ppl_before > 0 else 0
                eff_after = dur_after / ppl_after if ppl_after > 0 else 0
                ax2.text(1.5, max(ppl_before, ppl_after) * 0.8,
                       f"Efficiency:\n{eff_before:.1f} → {eff_after:.1f} hours/resource",
                       ha='center', va='center', fontsize=11,
                       bbox=dict(facecolor='#f8f9fa', edgecolor='#dee2e6', boxstyle='round,pad=0.5'))
            
            # Cost comparison (new third panel)
            bars3 = ax3.bar(['Before', 'After'],
                          [cost_before, cost_after],
                          color=['#f39c12', '#2980b9'])
            ax3.set_title('Total Cost ($)', fontsize=14)
            ax3.set_ylabel('Cost (USD)', fontsize=12)
            ax3.grid(axis='y', linestyle='--', alpha=0.3)
            
            # Format cost values with K/M/B suffixes for better readability
            def format_cost(cost):
                if cost >= 1_000_000_000:
                    return f"${cost/1_000_000_000:.1f}B"
                elif cost >= 1_000_000:
                    return f"${cost/1_000_000:.1f}M"
                elif cost >= 1_000:
                    return f"${cost/1_000:.1f}K"
                return f"${cost:,.2f}"
            
            for rect, val in zip(bars3, [cost_before, cost_after]):
                ax3.text(rect.get_x() + rect.get_width()/2, rect.get_height()*0.05,
                       format_cost(val), ha='center', va='bottom', 
                       fontsize=11, color='white', fontweight='bold',
                       bbox=dict(facecolor='#33333399', edgecolor='none', boxstyle='round,pad=0.2'))
            
            # Calculate cost savings
            if cost_before > 0:
                cost_savings = cost_before - cost_after
                cost_savings_pct = (cost_savings / cost_before) * 100
                ax3.text(1.5, max(cost_before, cost_after) * 0.8,
                       f"Cost {'Savings' if cost_savings >= 0 else 'Increase'}:\n{format_cost(abs(cost_savings))} ({abs(cost_savings_pct):.1f}%)",
                       ha='center', va='center', fontsize=11,
                       bbox=dict(facecolor='#f8f9fa', edgecolor='#dee2e6', boxstyle='round,pad=0.5'))
            
            # Add a note about cost calculation
            ax3.text(0.5, -0.2, "Cost definitions\n"
                               "Before: Sequential baseline using cheapest qualified resource rates per task\n"
                               "After: Optimized schedule using actual assigned resource rates",
                   transform=ax3.transAxes, ha='center', va='top', fontsize=9,
                   bbox=dict(facecolor='#f8f9fa', edgecolor='#dee2e6', boxstyle='round,pad=0.5'))

            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for title

            # Summary box
            time_saved = dur_before - dur_after
            pct_time = (time_saved / dur_before * 100.0) if dur_before > 0 else 0.0
            ppl_delta = ppl_after - ppl_before
            cost_delta = cost_after - cost_before
            pct_cost = (cost_delta / cost_before * 100.0) if cost_before > 0 else 0.0

            summary_lines = [
                'Optimization Results Summary:',
                f"• Time Saved: {time_saved:.1f} hours ({pct_time:.1f}% reduction)",
                f"• People Utilization: {'Increased' if ppl_delta >= 0 else 'Decreased'} by {abs(ppl_delta)} people",
                f"• Cost {'Increase' if cost_delta >= 0 else 'Decrease'}: ${abs(cost_delta):,.2f} ({pct_cost:.1f}%)",
                '',
                'Before Optimization:',
                f"• Duration: {dur_before:.1f} hours",
                f"• People: {ppl_before} person",
                f"• Cost: ${cost_before:,.2f}",
                '',
                'After Optimization:',
                f"• Duration: {dur_after:.1f} hours",
                f"• People: {ppl_after} people",
                f"• Cost: ${cost_after:,.2f}",
                '',
                'Total Process Time:',
                f"• Before (sequential sum of all tasks): {dur_before:.1f} hours",
                f"• After (optimized elapsed time): {dur_after:.1f} hours",
            ]

            fig.text(0.5, -0.05, '\n'.join(summary_lines), ha='center', va='top', fontsize=10,
                     bbox=dict(facecolor='#f8f9fa', edgecolor='#dcdcdc', boxstyle='round,pad=0.8'))

            fig.tight_layout(rect=[0, 0.05, 1, 0.95])

            os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
            fig.savefig(output_file, dpi=120, bbox_inches='tight')
            plt.close(fig)

            if show:
                try:
                    webbrowser.open(f'file://{os.path.abspath(output_file)}')
                except Exception:
                    pass
            return output_file
        except Exception as e:
            print(f"Error generating summary comparison: {str(e)}")
            return None

    def plot_whatif_summary(self, wi_results: Dict[str, Any], title: str = 'What-If Scenario Comparison',
                             output_file: Optional[str] = None, show: bool = False) -> Optional[str]:
        """
        Create a PNG summary comparing what-if scenarios and highlight the best.

        Args:
            wi_results: Results dict from WhatIfAnalyzer.analyze_scenarios
            title: Chart title
            output_file: Path without extension or with .png; if None returns None
            show: Whether to display interactively

        Returns:
            Path to saved PNG or None
        """
        # Disabled by request: do not generate What-If summary PNGs
        return None
        try:
            import matplotlib.pyplot as plt
            from matplotlib import gridspec
            import os

            if not wi_results or 'scenarios' not in wi_results:
                print("plot_whatif_summary: no scenarios to plot")
                return None

            scenarios = wi_results.get('scenarios', {})
            best = wi_results.get('best_scenario') or {}
            best_id = best.get('id')

            names = []
            scores = []
            time_pcts = []
            cost_pcts = []
            durations = []
            costs = []
            ids = []

            # Build arrays
            for sid, data in scenarios.items():
                ids.append(sid)
                names.append(data.get('name', sid))
                imp = data.get('improvement', {}) or {}
                scores.append(float(imp.get('score', 0.0) or 0.0))
                time_pcts.append(float(imp.get('time_pct', 0.0) or 0.0))
                cost_pcts.append(float(imp.get('cost_pct', 0.0) or 0.0))
                metrics = data.get('metrics', {}) or {}
                durations.append(float(metrics.get('total_duration', 0.0) or 0.0))
                costs.append(float(metrics.get('total_cost', 0.0) or 0.0))

            if not names:
                print("plot_whatif_summary: empty scenario list")
                return None

            fig = plt.figure(figsize=(12, 7))
            gs = gridspec.GridSpec(2, 2, height_ratios=[3, 2])

            # Bar chart of scores
            ax0 = fig.add_subplot(gs[0, :])
            bars = ax0.bar(range(len(names)), scores, color=['#2ecc71' if ids[i]==best_id else '#3498db' for i in range(len(names))])
            ax0.set_title(title)
            ax0.set_ylabel('Score (weighted)')
            ax0.set_xticks(range(len(names)))
            ax0.set_xticklabels(names, rotation=20, ha='right')
            for i, b in enumerate(bars):
                ax0.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{scores[i]:.3f}", ha='center', va='bottom', fontsize=9)

            # Time/Cost percent change bars
            ax1 = fig.add_subplot(gs[1, 0])
            width = 0.35
            idx = range(len(names))
            ax1.bar([i - width/2 for i in idx], time_pcts, width=width, label='Time %', color='#9b59b6')
            ax1.bar([i + width/2 for i in idx], cost_pcts, width=width, label='Cost %', color='#e67e22')
            ax1.axhline(0, color='gray', linewidth=0.8)
            ax1.set_ylabel('% change vs baseline')
            ax1.set_xticks(idx)
            ax1.set_xticklabels([str(i+1) for i in range(len(names))])
            ax1.legend()

            # Table with absolute metrics
            ax2 = fig.add_subplot(gs[1, 1])
            ax2.axis('off')
            table_data = [["#", "Scenario", "Duration (h)", "Cost ($)", "Score", "Best"]]
            for i, n in enumerate(names):
                mark = '✓' if ids[i] == best_id else ''
                table_data.append([str(i+1), n, f"{durations[i]:.1f}", f"{costs[i]:,.2f}", f"{scores[i]:.3f}", mark])
            t = ax2.table(cellText=table_data, loc='center', cellLoc='center', colLoc='center')
            t.auto_set_font_size(False)
            t.set_fontsize(9)
            t.scale(1, 1.2)

            plt.tight_layout()
            # Save
            if output_file:
                out = output_file if output_file.lower().endswith('.png') else f"{output_file}.png"
                plt.savefig(out, dpi=140, bbox_inches='tight')
                if not show:
                    plt.close(fig)
                return out
            else:
                if show:
                    plt.show()
                return None
        except Exception as e:
            # Disabled; keep silent/return None
            return None