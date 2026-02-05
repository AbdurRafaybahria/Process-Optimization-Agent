# Process Permission & Version Workflow

This document describes the approval workflow for creating new versions of processes. Non-admin users must submit changes for approval, while SUPER_ADMIN users can create directly.

---

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CREATE NEW VERSION                               │
│                                                                          │
│  User submits new version ──────────────────────────────────────────►   │
│                                                                          │
│     ┌──────────────────────┐       ┌──────────────────────┐             │
│     │   Is SUPER_ADMIN?    │──YES──►  Save to `process`   │             │
│     │                      │       │    table directly    │             │
│     └──────────────────────┘       └──────────────────────┘             │
│              │                                                           │
│              NO                                                          │
│              │                                                           │
│              ▼                                                           │
│     ┌──────────────────────┐                                            │
│     │ Save to `process_    │                                            │
│     │ permission` table    │                                            │
│     │ with PENDING status  │                                            │
│     └──────────────────────┘                                            │
│              │                                                           │
│              ▼                                                           │
│     ┌──────────────────────┐                                            │
│     │ SUPER_ADMIN reviews  │                                            │
│     │                      │                                            │
│     └──────────────────────┘                                            │
│         │             │                                                  │
│     APPROVED      REJECTED                                               │
│         │             │                                                  │
│         ▼             ▼                                                  │
│  ┌─────────────┐ ┌─────────────┐                                        │
│  │ Create in   │ │ Delete from │                                        │
│  │ `process`   │ │ `process_   │                                        │
│  │ table       │ │ permission` │                                        │
│  └─────────────┘ └─────────────┘                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Endpoints

### 1. Create New Version

Creates a new version of an existing process for approval.

| Field | Value |
|-------|-------|
| **Method** | `POST` |
| **URL** | `/process/:id/createNewVersion` |
| **Auth** | **Required** (JWT Cookie) |

#### URL Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | number | Process ID to create a new version from |

#### Request Body

```json
{
  "submitted_by": 3,
  "process_name": "Updated Manufacturing Process v2",
  "process_code": "MFG-002",
  "process_overview": "Updated description for version 2"
}
```

#### Request Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `submitted_by` | number | ✅ Yes | User ID of the submitter (must match logged-in user) |
| `process_name` | string | No | New name (optional, uses existing if not provided) |
| `process_code` | string | No | New code (optional) |
| `process_overview` | string | No | New description (optional) |
| `parent_process_id` | number | No | New parent process (optional) |
| `parent_task_id` | number | No | New parent task (optional) |
| `capacity_requirement_minutes` | number | No | New capacity (optional) |
| `process_status_id` | number | No | New status (optional) |
| `process_category_id` | number | No | New category (optional) |
| `workflow` | array | No | New workflow items (optional) |
| `gateways` | array | No | New gateways (optional) |

#### Response (Non-SUPER_ADMIN - Pending Approval)

```json
{
  "permission_id": 1,
  "process_name": "Updated Manufacturing Process v2",
  "process_code": "MFG-002",
  "company_id": 1,
  "process_version": 2,
  "permission_status": "PENDING",
  "submitter": {
    "user_id": 3,
    "name": "John Doe",
    "email": "john@example.com"
  },
  "processCategory": {
    "id": 1,
    "description": "Manufacturing"
  },
  "savedTo": "process_permission",
  "message": "New version submitted for approval"
}
```

#### Response (SUPER_ADMIN - Direct Creation)

```json
{
  "process_id": 10,
  "process_name": "Updated Manufacturing Process v2",
  "process_code": "MFG-002",
  "process_version": 2,
  "savedTo": "process",
  "message": "New version created directly (SUPER_ADMIN)"
}
```

---

### 2. Review Process Permission (SUPER_ADMIN Only)

Approve or reject a pending process permission request.

| Field | Value |
|-------|-------|
| **Method** | `PATCH` |
| **URL** | `/process/permission/:id/review` |
| **Auth** | **Required** (JWT Cookie + SUPER_ADMIN role) |

#### URL Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | number | Permission ID to review |

#### Request Body

```json
{
  "permission_status": "APPROVED",
  "remarks": "Verified and approved"
}
```

#### Request Fields

| Field | Type | Required | Values | Description |
|-------|------|----------|--------|-------------|
| `permission_status` | enum | ✅ Yes | `APPROVED`, `REJECTED` | Review decision |
| `remarks` | string | No | max 2000 chars | Optional notes |

---

### Approve Example

**Request:**

```bash
curl -X PATCH http://localhost:3000/process/permission/1/review \
  -H "Content-Type: application/json" \
  -H "Cookie: access_token=<super_admin_token>" \
  -d '{
    "permission_status": "APPROVED",
    "remarks": "Verified and approved"
  }'
```

**Response:**

```json
{
  "action": "APPROVED",
  "message": "Process approved and created successfully",
  "process": {
    "process_id": 10,
    "process_name": "Updated Manufacturing Process v2",
    "process_code": "MFG-002",
    "process_version": 2
  },
  "reviewed_by": {
    "user_id": 1,
    "name": "Super Admin",
    "email": "admin@example.com"
  },
  "remarks": "Verified and approved"
}
```

---

### Reject Example

**Request:**

```bash
curl -X PATCH http://localhost:3000/process/permission/1/review \
  -H "Content-Type: application/json" \
  -H "Cookie: access_token=<super_admin_token>" \
  -d '{
    "permission_status": "REJECTED",
    "remarks": "Missing required workflow steps"
  }'
```

**Response:**

```json
{
  "action": "REJECTED",
  "message": "Process permission rejected and deleted",
  "permission_id": 1,
  "process_name": "Updated Manufacturing Process v2",
  "submitted_by": {
    "user_id": 3,
    "name": "John Doe",
    "email": "john@example.com"
  },
  "reviewed_by": {
    "user_id": 1,
    "name": "Super Admin",
    "email": "admin@example.com"
  },
  "remarks": "Missing required workflow steps"
}
```

---

## Error Responses

| Status Code | Error | Description |
|-------------|-------|-------------|
| `400` | Bad Request | Invalid data or already reviewed |
| `401` | Unauthorized | Login required |
| `403` | Forbidden | `submitted_by` doesn't match logged-in user, or not SUPER_ADMIN |
| `404` | Not Found | Process or permission not found |

### Error Examples

**submitted_by mismatch:**
```json
{
  "statusCode": 403,
  "message": "submitted_by (5) does not match the logged-in user (3). You can only submit on behalf of yourself."
}
```

**Not SUPER_ADMIN:**
```json
{
  "statusCode": 403,
  "message": "Only SUPER_ADMIN can review process permissions"
}
```

**Already reviewed:**
```json
{
  "statusCode": 400,
  "message": "This process permission has already been approved"
}
```

---

## Database Tables

### `process_permission` Table

| Column | Type | Description |
|--------|------|-------------|
| `permission_id` | int | Primary key |
| `process_name` | varchar | Proposed process name |
| `process_code` | varchar | Proposed process code |
| `company_id` | int | Company ID |
| `process_version` | int | Version number |
| `permission_status` | enum | `PENDING`, `APPROVED`, `REJECTED` |
| `submitted_by` | int | User ID who submitted |
| `reviewed_by` | int | User ID who reviewed (nullable) |
| `reviewed_at` | datetime | Review timestamp (nullable) |
| `remarks` | text | Review remarks (nullable) |
| `created_at` | datetime | Submission timestamp |

---

## Postman Collection Examples

### Create New Version

```
POST {{baseUrl}}/process/5/createNewVersion
Content-Type: application/json

{
  "submitted_by": 3,
  "process_name": "New Process Version"
}
```

### Review Permission (Approve)

```
PATCH {{baseUrl}}/process/permission/1/review
Content-Type: application/json

{
  "permission_status": "APPROVED",
  "remarks": "Looks good!"
}
```

### Review Permission (Reject)

```
PATCH {{baseUrl}}/process/permission/1/review
Content-Type: application/json

{
  "permission_status": "REJECTED",
  "remarks": "Please add more details"
}
```

---

## Related Documentation

- [Process API](./PROCESS_API.md)
