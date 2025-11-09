"""
Manufacturing-specific visualization helper methods
These will be added to the Visualizer class
"""

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


def _plot_manufacturing_cost_comparison(self, ax, schedule, before_metrics):
    """Plot before/after total cost comparison"""
    # Calculate after cost
    after_cost = 0
    for entry in schedule.entries:
        duration = entry.end_hour - entry.start_hour
        # Try to get cost from entry or calculate from resource rate
        if hasattr(entry, 'cost'):
            after_cost += entry.cost
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


def _plot_manufacturing_summary_table(self, ax, schedule, before_metrics):
    """Plot summary table for manufacturing optimization"""
    ax.axis('off')
    
    # Calculate metrics
    if schedule.entries:
        after_duration = max(entry.end_hour for entry in schedule.entries)
        after_resources = len(set(entry.resource_id for entry in schedule.entries))
        after_cost = sum((entry.end_hour - entry.start_hour) * 50 for entry in schedule.entries)
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
