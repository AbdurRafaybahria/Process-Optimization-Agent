# Process API Documentation

This document describes the Process management endpoints for creating, updating, and managing processes.

---

## 1. Create Process (Full Example)

**Endpoint:** `POST /process`

### Request JSON

```json
{
  "process_name": "Complex Manufacturing Process",
  "process_code": "MFG-COMPLEX-01",
  "company_id": 1,
  "process_overview": "A detailed process with decision points and parallel execution.",
  "capacity_requirement_minutes": 180,
  "process_status_id": 1,
  "process_category_id": 2,
  "process_version": 1,
  
  "workflow": [
    {
      "item_type": "task",
      "task_id": 101,
      "sequence_number": 1,
      "order": 1
    },
    {
      "item_type": "task",
      "task_id": 102,
      "sequence_number": 2,
      "order": 2
    },
    {
      "item_type": "task",
      "task_id": 103,
      "sequence_number": 3,
      "order": 3
    }
  ],

  "gateways": [
    {
      "gateway_type": "EXCLUSIVE",
      "name": "Quality Check",
      "after_task_id": 101,
      "branches": [
        {
          "condition": "PASSED",
          "target_task_id": 102,
          "is_default": true
        },
        {
          "condition": "FAILED",
          "end_event_name": "Reject & Scrap",
          "is_default": false
        }
      ]
    },
    {
      "gateway_type": "PARALLEL",
      "name": "Start Parallel Assembly",
      "after_task_id": 102,
      "branches": [
        {
          "target_task_id": 103,
          "is_default": false
        },
        {
          "target_task_id": 104,
          "is_default": false
        }
      ]
    }
  ]
}
```

---

## 2. Get All Processes

**Endpoint:** `GET /process`
**Query Params:** `company_id=1`

---

## 3. Get Process by ID

**Endpoint:** `GET /process/:id`

**Response Example:**

```json
{
  "process_id": 10,
  "process_name": "Complex Manufacturing Process",
  "process_code": "MFG-COMPLEX-01",
  "company_id": 1,
  "gateways": [
    {
      "gateway_pk_id": 5,
      "gateway_type": "EXCLUSIVE",
      "name": "Quality Check",
      "after_task_id": 101,
      "branches": [
        {
          "id": 12,
          "condition": "PASSED",
          "target_task_id": 102,
          "is_default": true
        },
        {
          "id": 13,
          "condition": "FAILED",
          "end_event_name": "Reject & Scrap",
          "is_default": false
        }
      ]
    }
  ]
}
```

---

## 4. Updates & Deletes

- **Update:** `PATCH /process/:id`
- **Delete:** `DELETE /process/:id`

---

## 5. Get Optimized Process Version (For CMS Database)

**Endpoint:** `GET /save-optimized-version/{process_id}`

**Description:** This endpoint retrieves the optimized version of a process with gateway suggestions, formatted for saving to the CMS database. It includes:
- Optimized workflow with tasks in execution order
- Parallel gateway suggestions for concurrent task execution
- Exclusive gateway suggestions for decision points
- Process metadata ready for database insertion

**Headers:**
- `Authorization: Bearer <token>` (optional - will authenticate automatically if not provided)

**Response Example:**

```json
{
  "process_name": "Complex Manufacturing Process",
  "process_code": "MFG-COMPLEX-01",
  "company_id": 1,
  "process_overview": "A detailed process with decision points and parallel execution.",
  "capacity_requirement_minutes": 180,
  "process_status_id": 1,
  "process_category_id": 2,
  "process_version": 2,
  
  "workflow": [
    {
      "item_type": "task",
      "task_id": 101,
      "sequence_number": 1,
      "order": 1
    },
    {
      "item_type": "task",
      "task_id": 102,
      "sequence_number": 2,
      "order": 2
    },
    {
      "item_type": "task",
      "task_id": 103,
      "sequence_number": 3,
      "order": 3
    },
    {
      "item_type": "task",
      "task_id": 104,
      "sequence_number": 4,
      "order": 4
    }
  ],

  "gateways": [
    {
      "gateway_type": "EXCLUSIVE",
      "name": "Quality Check",
      "after_task_id": 101,
      "branches": [
        {
          "condition": "PASSED",
          "target_task_id": 102,
          "is_default": true
        },
        {
          "condition": "FAILED",
          "end_event_name": "Reject & Scrap",
          "is_default": false
        }
      ]
    },
    {
      "gateway_type": "PARALLEL",
      "name": "Start Parallel Assembly",
      "after_task_id": 102,
      "branches": [
        {
          "target_task_id": 103,
          "is_default": false
        },
        {
          "target_task_id": 104,
          "is_default": false
        }
      ]
    }
  ]
}
```

**Usage Example:**

```javascript
// Fetch optimized process version
const response = await fetch(`${API_URL}/save-optimized-version/123`, {
  headers: {
    'Authorization': `Bearer ${token}`
  }
});

const optimizedVersion = await response.json();

// Save to CMS database
await saveProcessVersion(optimizedVersion);
```

**Key Features:**
- Automatically increments `process_version` number
- Includes optimized task execution order based on parallel opportunities
- Detects and includes both EXCLUSIVE and PARALLEL gateway suggestions
- Returns data in exact format needed for CMS database insertion
- Preserves all process metadata (company_id, status, category, etc.)

---

## 6. Versioning (Create New Version)

**Quick Example (Create Version):**

```json
{
  "submitted_by": 3,
  "process_name": "Complex Manufacturing Process V2",
  "gateways": [
     {
        "gateway_type": "EXCLUSIVE",
        "name": "New Check",
        "after_task_id": 101,
        "branches": [ ... ]
     }
  ]
}
```
