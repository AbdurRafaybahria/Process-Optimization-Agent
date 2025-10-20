# Process Optimization Agent - Test Script Usage

## Overview
The test script (`test_process_detection.py`) is fully dynamic and can test **any process JSON file** without any hardcoding. It automatically detects whether a process is healthcare, manufacturing, or another type and applies the appropriate optimization strategy.

## Usage

### Basic Usage
```bash
# Test any JSON file - the script will automatically detect the process type
python test_process_detection.py <path_to_json_file>

# Examples:
python test_process_detection.py examples/outpatient_consultation.json
python test_process_detection.py examples/patient_registration.json
python test_process_detection.py examples/ecommerce_development.json
python test_process_detection.py path/to/your/custom_process.json
```

### Get Help
```bash
python test_process_detection.py --help
```

## Example Test Processes

### Healthcare Processes
1. **Outpatient Consultation**
   - File: `examples/outpatient_consultation.json`
   - Auto-detected as: HEALTHCARE
   - Features: Patient journey optimization, waiting time minimization
   - Shows: Patient vs Admin timeline separation

2. **Patient Registration**
   - File: `examples/patient_registration.json`
   - Auto-detected as: HEALTHCARE
   - Features: Admin-heavy process, minimal patient involvement
   - Shows: Administrative overhead analysis

### Manufacturing/Development Processes
3. **E-Commerce Platform Development**
   - File: `examples/ecommerce_development.json`
   - Auto-detected as: MANUFACTURING
   - Features: Parallel task execution, cycle time optimization
   - Shows: Parallel execution analysis, throughput metrics

## What the Script Does

### 1. Automatic Process Type Detection
- Analyzes process name, tasks, and resources
- Detects: Healthcare, Manufacturing, Banking, or Generic
- Confidence scoring (0-100%)

### 2. Intelligent Optimization
- **Healthcare**: Minimizes patient waiting time, sequential flow
- **Manufacturing**: Maximizes parallelization, minimizes cycle time
- **Banking**: Optimizes approval workflows

### 3. Dynamic Reporting
- **Healthcare processes** show:
  - Patient journey metrics (arrival, departure, waiting time)
  - Patient-involved vs admin-only task separation
  - Patient efficiency ratio
  
- **Manufacturing processes** show:
  - Cycle time (makespan)
  - Parallel execution analysis
  - Throughput and cost per task
  - Which tasks run in parallel

### 4. Validation
- Shows detected indicators for the process type
- No hardcoded expectations
- Adapts to any process automatically

## Output
Results are saved to `test_results_full.txt` with:
- Process overview
- Detection results
- Optimization metrics
- Recommendations
- Resource tracking

## Examples

### Test Healthcare Process
```bash
python test_process_detection.py examples/outpatient_consultation.json
```
Output shows:
- Patient journey with [DIRECT], [PASSIVE], [ADMIN] labels
- Patient arrival/departure times
- Waiting time analysis

### Test Manufacturing Process
```bash
python test_process_detection.py examples/ecommerce_development.json
```
Output shows:
- Parallel execution analysis
- Cycle time optimization
- No user involvement labels (not applicable)

### Test Any Custom Process
```bash
python test_process_detection.py path/to/your_process.json
```
Automatically detects type and applies appropriate optimization!

## Key Features

✅ **No Hardcoding** - Works with any process
✅ **Automatic Detection** - Identifies process type intelligently
✅ **Adaptive Reporting** - Shows relevant metrics for each type
✅ **Command-Line Interface** - Easy to use and script
✅ **Extensible** - Easy to add new processes

## Notes

- The script automatically determines the best optimization strategy
- Healthcare processes focus on user experience
- Manufacturing processes focus on time and cost efficiency
- All results are logged to `test_results_full.txt`
