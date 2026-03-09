# Inclusive OR Gateway (BPMN 2.0) — Complete Developer Guide

> **Audience:** Developers working on the Digital Twin BPMN engine  
> **Last Updated:** March 9, 2026

---

## Table of Contents

1. [What is the Inclusive OR Gateway?](#what-is-the-inclusive-or-gateway)
2. [BPMN Notation & Symbol](#bpmn-notation--symbol)
3. [How It Works — Split & Join](#how-it-works--split--join)
4. [Inclusive OR vs Other Gateways](#inclusive-or-vs-other-gateways)
5. [When to Use the Inclusive OR Gateway](#when-to-use-the-inclusive-or-gateway)
6. [Our API Implementation](#our-api-implementation)
7. [Practical Examples with API Payloads](#practical-examples-with-api-payloads)
8. [Convergence (Join) Behavior In-Depth](#convergence-join-behavior-in-depth)
9. [Default Branch & Edge Cases](#default-branch--edge-cases)
10. [Execution Semantics — Token-Based Explanation](#execution-semantics--token-based-explanation)
11. [Common Mistakes & Pitfalls](#common-mistakes--pitfalls)
12. [Real-World Scenarios](#real-world-scenarios)
13. [FAQ](#faq)

---

## What is the Inclusive OR Gateway?

The **Inclusive OR Gateway** (also called the **OR Gateway** or **Inclusive Gateway**) is one of the four gateway types in BPMN 2.0. It is represented by a **diamond shape with a circle (○) inside**.

> [!IMPORTANT]
> The key distinction of the Inclusive OR is that it allows **one or more** outgoing branches to be activated simultaneously based on conditions, making it a hybrid between Exclusive (XOR) and Parallel (AND) gateways.

### The Core Idea

Think of it like ordering food at a restaurant:
- **Exclusive (XOR):** "Pick ONE dish" — You can only choose one entrée.
- **Parallel (AND):** "You get ALL dishes" — Every dish on the menu is served.
- **Inclusive (OR):** "Pick ONE or MORE dishes" — You choose which dishes you want; you must pick at least one.

---

## BPMN Notation & Symbol

```
     ┌───────┐
     │       │
     │   ○   │   ← Circle inside a diamond
     │       │
     └───────┘
```

| Aspect          | Detail                                    |
|-----------------|-------------------------------------------|
| **Shape**       | Diamond (rhombus)                         |
| **Marker**      | Circle (○) inside                         |
| **Diverging**   | Circle marker — conditions on each branch |
| **Converging**  | Circle marker — waits for active tokens   |
| **BPMN Element**| `<bpmn:inclusiveGateway>`                 |

### Visual Comparison of Gateway Markers

| Gateway Type   | Marker Inside Diamond | Symbol |
|----------------|----------------------|--------|
| Exclusive (XOR)| **✕** (X mark)       | `(X)`  |
| Parallel (AND) | **+** (plus sign)    | `(+)`  |
| **Inclusive (OR)** | **○** (circle)   | `(○)`  |
| Event-Based    | **⬡** (pentagon)     | `(⬡)`  |

---

## How It Works — Split & Join

The Inclusive OR Gateway operates in two modes:

### Diverging (Split) — Forking the Flow

When a token arrives at a **diverging Inclusive OR Gateway**:

1. **Every** outgoing sequence flow's condition is evaluated.
2. **All** branches where the condition evaluates to `true` are activated.
3. A token is sent along **each** activated branch.
4. **At least one** branch must be activated (otherwise it's a runtime error, unless there is a default flow).

```
                    ┌──→ [Branch A] (condition A = true)  ✅ Activated
                    │
Token → (○) ────────┼──→ [Branch B] (condition B = true)  ✅ Activated
         Split      │
                    └──→ [Branch C] (condition C = false) ❌ Not activated
```

> [!NOTE]
> Unlike the Exclusive Gateway where only the **first** matching branch is taken, the Inclusive OR evaluates **all** conditions and activates **every** branch that matches.

### Converging (Join) — Merging the Flow

When used as a **converging** gateway (join), the Inclusive OR:

1. **Waits** for tokens from all **active** incoming branches.
2. Only branches that were actually triggered need to arrive.
3. Once all active tokens have arrived, the flow continues.

```
[Branch A] (active)  ──→ ┐
                         │
[Branch B] (active)  ──→ ├──→ (○) ──→ Continue
                         │    Join
[Branch C] (inactive) ─x ┘         (not waiting for C)
```

> [!WARNING]
> This is where the Inclusive OR gets complex at the engine level. The join must **dynamically determine** which upstream branches were activated to know how many tokens to wait for. Incorrect implementation can cause **deadlocks** or **premature continuation**.

---

## Inclusive OR vs Other Gateways

### Side-by-Side Comparison

| Feature                    | Exclusive (XOR)          | Parallel (AND)            | **Inclusive (OR)**              |
|----------------------------|--------------------------|---------------------------|---------------------------------|
| **Branches Activated**     | Exactly **1**            | **All**                   | **1 or more**                   |
| **Conditions Required?**   | Yes (mandatory)          | No (not used)             | Yes (evaluated per branch)      |
| **Default Branch**         | Yes (recommended)        | N/A                       | Yes (recommended as fallback)   |
| **Join Waits For**         | First token (no sync)    | **All** incoming tokens   | All **active** incoming tokens  |
| **Use Case**               | if-else decisions        | Parallel work             | Conditional parallel work       |
| **Risk of Deadlock**       | None                     | If branch never completes | If join miscounts active tokens |
| **BPMN Symbol**            | ✕                        | +                         | ○                               |

### Decision Flowchart — Which Gateway to Use?

```
                        How many paths should execute?
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
              Exactly ONE     ALL of them    One or MORE
                    │               │               │
              Use EXCLUSIVE    Use PARALLEL   Use INCLUSIVE
                 (XOR)            (AND)           (OR)
```

---

## When to Use the Inclusive OR Gateway

### ✅ Use Inclusive OR When

- **Multiple conditions** can be true at the same time, and you want to execute matching paths in parallel.
- The number of active paths **varies** based on runtime data.
- You need a **flexible fork** — not strictly "one of" or "all of."

### ❌ Don't Use Inclusive OR When

- Only **one** outcome should happen → Use **Exclusive (XOR)**.
- **All** paths should always execute → Use **Parallel (AND)** — it's simpler and more efficient.
- You're waiting for an **external event** → Use **Event-Based Gateway**.

### Quick Heuristic

> Ask: *"Can multiple conditions be true at the same time, but not necessarily all?"*
> - **Yes** → Inclusive OR
> - **No, only one** → Exclusive
> - **Always all** → Parallel

---

## Our API Implementation

In our Digital Twin Server, the Inclusive OR Gateway is defined in the `GatewayType` enum:

### Enum Definition

**File:** `backend/server/src/bpmn-native/dto/gateway.dto.ts`

```typescript
export enum GatewayType {
    EXCLUSIVE = 'EXCLUSIVE',   // XOR - Only ONE branch taken
    PARALLEL  = 'PARALLEL',    // AND - ALL branches taken simultaneously
    INCLUSIVE  = 'INCLUSIVE',   // OR  - One or MORE branches taken
    EVENT_BASED = 'EVENT_BASED', // Waits for first event
}
```

### Gateway DTO Structure

```typescript
export class GatewayDto {
    type: GatewayType;           // Set to 'INCLUSIVE'
    name?: string;                // e.g., "Check Requirements"
    afterTaskId?: number | null;  // Task ID before the gateway
    convergeAtTaskId?: number;    // Task ID where branches merge (JOIN)
    branches: GatewayBranchDto[]; // Array of branch paths
}
```

### Branch DTO Structure

```typescript
export class GatewayBranchDto {
    targetTaskId?: number;   // Task this branch leads to
    endTaskId?: number;      // Task after which branch ends
    endEventName?: string;   // Name for branch-specific end event
    condition?: string;      // Condition label for the branch
    isDefault?: boolean;     // Marks as default/fallback path
}
```

### API Endpoint

```
POST /bpmn-native/generate/:processId?format=png
```

| Parameter | Type   | Default | Description                       |
|-----------|--------|---------|-----------------------------------|
| `format`  | string | `png`   | Output: `png`, `svg`, `jpeg`, `xml` |
| `width`   | number | auto    | Image width (for png/jpeg)        |

---

## Practical Examples with API Payloads

### Example 1: Basic Inclusive OR — Patient Preparation

A hospital discharge process where some preparation steps are conditional.

```json
{
  "gateways": [
    {
      "type": "INCLUSIVE",
      "name": "Required Preparations",
      "afterTaskId": 174,
      "convergeAtTaskId": 717,
      "branches": [
        {
          "targetTaskId": 715,
          "condition": "needsLabWork == true"
        },
        {
          "targetTaskId": 59,
          "condition": "needsConsent == true"
        }
      ]
    }
  ]
}
```

**Visual Result:**

```
                    ┌──→ Task 715 (Lab Work) ──────────┐
                    │   [if needsLabWork == true]       │
Task 174 ───→ (○) ──┤                                   ├──→ (○) ──→ Task 717
         Split      │                                   │    Join
                    └──→ Task 59 (Get Consent) ────────┘
                        [if needsConsent == true]
```

**Possible runtime execution paths:**
| needsLabWork | needsConsent | Branches Activated                |
|-------------|-------------|-----------------------------------|
| `true`      | `true`      | Both: Lab Work + Consent          |
| `true`      | `false`     | Only: Lab Work                    |
| `false`     | `true`      | Only: Consent                     |
| `false`     | `false`     | ⚠️ ERROR — no branch activated!  |

> [!CAUTION]
> If **no** condition evaluates to `true` and there is no default branch, the process instance will fail. Always add a default branch or ensure at least one condition will always be true.

---

### Example 2: Three-Way Inclusive OR — Order Fulfillment

An order may require different handling depending on the items ordered.

```json
{
  "gateways": [
    {
      "type": "INCLUSIVE",
      "name": "Fulfillment Steps",
      "afterTaskId": 100,
      "convergeAtTaskId": 104,
      "branches": [
        {
          "targetTaskId": 101,
          "condition": "hasPhysicalItems == true"
        },
        {
          "targetTaskId": 102,
          "condition": "hasDigitalItems == true"
        },
        {
          "targetTaskId": 103,
          "condition": "requiresGiftWrapping == true"
        }
      ]
    }
  ]
}
```

**Visual Result:**

```
                    ┌──→ Task 101 (Ship Physical) ──┐
                    │                                │
Task 100 ───→ (○) ──┼──→ Task 102 (Send Digital) ───┼──→ (○) ──→ Task 104
         Split      │                                │    Join    (Invoice)
                    └──→ Task 103 (Gift Wrap) ───────┘
```

If a customer orders a physical item with gift wrapping (but no digital items): Tasks 101 and 103 execute concurrently, while Task 102 is skipped. The join gateway waits for both 101 and 103 before proceeding to 104.

---

### Example 3: Inclusive OR with a Default Branch

Adding a default branch as a safety net.

```json
{
  "gateways": [
    {
      "type": "INCLUSIVE",
      "name": "Notification Channels",
      "afterTaskId": 200,
      "convergeAtTaskId": 205,
      "branches": [
        {
          "targetTaskId": 201,
          "condition": "emailEnabled == true"
        },
        {
          "targetTaskId": 202,
          "condition": "smsEnabled == true"
        },
        {
          "targetTaskId": 203,
          "condition": "pushEnabled == true"
        },
        {
          "targetTaskId": 204,
          "isDefault": true,
          "condition": "Log Only (default)"
        }
      ]
    }
  ]
}
```

The **default branch** (Task 204) activates **only** if no other condition evaluates to `true`. This ensures the process never stalls.

---

### Example 4: Inclusive OR Without Join (Branches Go to End)

Sometimes branches don't need to converge — each terminates independently.

```json
{
  "gateways": [
    {
      "type": "INCLUSIVE",
      "name": "Notification Type",
      "afterTaskId": 300,
      "branches": [
        {
          "targetTaskId": 301,
          "condition": "notifyManager == true"
        },
        {
          "endEventName": "Email Sent",
          "condition": "sendEmail == true"
        },
        {
          "endEventName": "Audit Logged",
          "condition": "logAudit == true"
        }
      ]
    }
  ]
}
```

**Visual Result:**

```
                    ┌──→ Task 301 (Notify Manager) ──→ End
                    │
Task 300 ───→ (○) ──┼──→ ● Email Sent
         Split      │
                    └──→ ● Audit Logged
```

> [!NOTE]
> When `convergeAtTaskId` is **omitted**, branches do not merge — each runs to its own end event or continues the main sequence independently.

---

## Convergence (Join) Behavior In-Depth

The `convergeAtTaskId` field on our `GatewayDto` controls where branches merge.

### How the Join Gateway Works

```
                    ┌──→ Task A ──┐        Task A token arrives → wait...
                    │             │
     (○) Split ─────┤             ├──→ (○) Join ──→ Next Task
                    │             │        Task B token arrives → continue!
                    └──→ Task B ──┘
```

### Rules for `convergeAtTaskId`

| Rule | Description |
|------|-------------|
| **Must be a valid task** | The task ID must exist in the process workflow |
| **Should come after branches** | The convergence task should logically occur after all branch targets |
| **Only for PARALLEL & INCLUSIVE** | Exclusive (XOR) gateways don't need convergence |
| **Omit for separate endpoints** | Don't use `convergeAtTaskId` if branches should end independently |

### Token Synchronization

The join gateway's behavior is what makes the Inclusive OR complex:

```
Scenario: 3 branches exist → only 2 are activated at runtime

Branch A: ✅ activated  →  token arrives at join  →  ⏳ waiting for B
Branch B: ✅ activated  →  token arrives at join  →  ✅ all active arrived → CONTINUE
Branch C: ❌ not activated → join does NOT wait for this token
```

> [!IMPORTANT]
> The join must know which branches were activated upstream. In BPMN execution engines this is tracked via **token state**. Our BPMN generation creates the correct XML structure; the actual runtime synchronization depends on the BPMN engine (e.g., Camunda, Flowable) used to execute the process.

---

## Default Branch & Edge Cases

### What Happens If No Condition Is True?

| Has Default Branch? | Behavior |
|---------------------|----------|
| **Yes** | The default branch is activated (just like XOR) |
| **No**  | ⚠️ **Runtime error** — the BPMN engine will throw an exception |

### Setting a Default Branch

```json
{
  "targetTaskId": 999,
  "isDefault": true,
  "condition": "Fallback Path"
}
```

> [!TIP]
> **Best Practice:** Always include a default branch on your Inclusive OR gateway, even if you believe one condition will always be true. This prevents unexpected process failures in edge cases.

### Edge Case: All Conditions Are True

If **all** conditions evaluate to `true`, then **all** branches execute — behaving identically to a Parallel (AND) gateway. This is perfectly valid.

### Edge Case: Only One Condition Is True

If only **one** condition evaluates to `true`, then only that single branch executes — behaving identically to an Exclusive (XOR) gateway. This is also valid.

---

## Execution Semantics — Token-Based Explanation

BPMN processes are executed using a **token-based** model. Here's how tokens flow through an Inclusive OR:

### Step-by-Step Token Flow

```
Step 1: A single token arrives at the Inclusive OR (Split)
        ┌──────────────┐
        │   (○) Split  │ ← 1 incoming token
        └──────────────┘

Step 2: Conditions are evaluated
        Branch A: amount > 100      → true  ✅
        Branch B: isPriority == true → true  ✅
        Branch C: needsAudit == true → false ❌

Step 3: Tokens are created for each activated branch
        Token 1 → Branch A → Task A
        Token 2 → Branch B → Task B
        (No token for Branch C)

Step 4: Branch tasks execute in parallel
        Task A: running...
        Task B: running...

Step 5: Tokens arrive at the Inclusive OR (Join)
        Token 1 from Task A → arrives at Join → ⏳ waiting
        Token 2 from Task B → arrives at Join → ✅ all active tokens received

Step 6: Single merged token continues the flow
        ┌──────────────┐
        │   (○) Join   │ → 1 outgoing token → Next Task
        └──────────────┘
```

---

## Common Mistakes & Pitfalls

### ❌ Mistake 1: Forgetting `convergeAtTaskId`

```json
{
  "type": "INCLUSIVE",
  "afterTaskId": 10,
  "branches": [
    { "targetTaskId": 11, "condition": "A" },
    { "targetTaskId": 12, "condition": "B" }
  ]
}
```

**Problem:** Without `convergeAtTaskId`, branches will not merge. Each branch will flow independently to the end. If you need synchronization, you **must** specify `convergeAtTaskId`.

### ❌ Mistake 2: No Default Branch, No Guaranteed Condition

```json
{
  "type": "INCLUSIVE",
  "afterTaskId": 10,
  "branches": [
    { "targetTaskId": 11, "condition": "x > 100" },
    { "targetTaskId": 12, "condition": "y > 200" }
  ]
}
```

**Problem:** If `x <= 100` AND `y <= 200`, no branch is activated → **runtime error**.

**Fix:** Add a default branch:
```json
{ "targetTaskId": 13, "isDefault": true, "condition": "Default" }
```

### ❌ Mistake 3: Confusing Inclusive OR with Exclusive OR

| If you write…                       | It means…                                   |
|-------------------------------------|---------------------------------------------|
| `"type": "EXCLUSIVE"` with 3 branches | **Only 1** branch executes                  |
| `"type": "INCLUSIVE"` with 3 branches | **1, 2, or all 3** branches may execute     |

### ❌ Mistake 4: Using Inclusive OR When Parallel Would Suffice

If **all** branches should **always** execute, use `PARALLEL` instead. It's simpler and more semantically clear.

---

## Real-World Scenarios

### Scenario 1: Insurance Claim Processing

An insurance claim may require different assessments depending on the type and severity.

```json
{
  "gateways": [
    {
      "type": "INCLUSIVE",
      "name": "Required Assessments",
      "afterTaskId": 500,
      "convergeAtTaskId": 505,
      "branches": [
        {
          "targetTaskId": 501,
          "condition": "claimAmount > 10000"
        },
        {
          "targetTaskId": 502,
          "condition": "involvesMedical == true"
        },
        {
          "targetTaskId": 503,
          "condition": "isThirdParty == true"
        },
        {
          "targetTaskId": 504,
          "isDefault": true,
          "condition": "Standard Review"
        }
      ]
    }
  ]
}
```

```
                           ┌──→ Task 501 (Senior Review) ────────┐
                           │   [claimAmount > 10000]              │
                           │                                      │
Task 500 ───→ (○) ─────────┼──→ Task 502 (Medical Assessment) ───┼──→ (○) ──→ Task 505
         Split             │   [involvesMedical]                  │    Join   (Decision)
                           │                                      │
                           ├──→ Task 503 (3rd Party Eval) ────────┤
                           │   [isThirdParty]                     │
                           │                                      │
                           └──→ Task 504 (Standard Review) ───────┘
                               [default fallback]
```

### Scenario 2: Employee Onboarding

Different onboarding tasks apply depending on the employee's role and location.

```json
{
  "gateways": [
    {
      "type": "INCLUSIVE",
      "name": "Onboarding Requirements",
      "afterTaskId": 600,
      "convergeAtTaskId": 606,
      "branches": [
        {
          "targetTaskId": 601,
          "condition": "requiresHardware == true"
        },
        {
          "targetTaskId": 602,
          "condition": "requiresSecurityClearance == true"
        },
        {
          "targetTaskId": 603,
          "condition": "isRemote == true"
        },
        {
          "targetTaskId": 604,
          "condition": "needsTraining == true"
        },
        {
          "targetTaskId": 605,
          "isDefault": true,
          "condition": "Basic Setup Only"
        }
      ]
    }
  ]
}
```

### Scenario 3: E-Commerce Order Processing

Different fulfillment steps based on what the customer ordered.

```json
{
  "gateways": [
    {
      "type": "INCLUSIVE",
      "name": "Order Fulfillment",
      "afterTaskId": 700,
      "convergeAtTaskId": 705,
      "branches": [
        {
          "targetTaskId": 701,
          "condition": "hasPhysicalProducts == true"
        },
        {
          "targetTaskId": 702,
          "condition": "hasSubscription == true"
        },
        {
          "targetTaskId": 703,
          "condition": "hasGiftCard == true"
        },
        {
          "targetTaskId": 704,
          "condition": "requiresCustomization == true"
        }
      ]
    }
  ]
}
```

---

## FAQ

### Q: What's the difference between Inclusive OR and Parallel?

**Parallel (AND)** always activates **all** branches — no conditions are evaluated. **Inclusive OR** evaluates conditions and activates only the matching branches (which could be all, some, or just one).

### Q: Can I nest Inclusive OR gateways?

Yes. You can have an Inclusive OR inside a branch of another Inclusive OR (or any other gateway type). Each gateway is independently configured in the `gateways` array.

### Q: How does the join know which branches were activated?

The BPMN execution engine tracks which outgoing sequence flows were taken at the split gateway. The converging (join) gateway uses this information to know which tokens to expect. Our `convergeAtTaskId` field generates the correct BPMN XML structure for this.

### Q: What if I have conditions that overlap?

That's **perfectly fine** — in fact, that's the whole point of the Inclusive OR. If multiple conditions are true, multiple branches run. Example:

```
Branch A condition: orderTotal > 50    → true if order is $150
Branch B condition: orderTotal > 100   → true if order is $150
```

Both A and B execute for a $150 order.

### Q: How is this represented in BPMN XML?

The Inclusive OR generates a `<bpmn:inclusiveGateway>` element in the XML:

```xml
<!-- Diverging (Split) -->
<bpmn:inclusiveGateway id="inclusive_split_1"
                        name="Check Requirements"
                        gatewayDirection="Diverging">
  <bpmn:incoming>flow_from_task</bpmn:incoming>
  <bpmn:outgoing>flow_to_branch_a</bpmn:outgoing>
  <bpmn:outgoing>flow_to_branch_b</bpmn:outgoing>
</bpmn:inclusiveGateway>

<!-- Converging (Join) -->
<bpmn:inclusiveGateway id="inclusive_join_1"
                        name=""
                        gatewayDirection="Converging">
  <bpmn:incoming>flow_from_branch_a</bpmn:incoming>
  <bpmn:incoming>flow_from_branch_b</bpmn:incoming>
  <bpmn:outgoing>flow_to_next_task</bpmn:outgoing>
</bpmn:inclusiveGateway>
```

### Q: Do I need both a split and a join?

- **If branches should merge:** Yes — use `convergeAtTaskId` to create the join.
- **If branches end independently:** No — omit `convergeAtTaskId` and each branch flows to its own end event.

---

## Quick Reference Card

```
┌──────────────────────────────────────────────────────────┐
│                  INCLUSIVE OR GATEWAY (○)                 │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Type:          "INCLUSIVE"                               │
│  Symbol:        Diamond with Circle (○)                  │
│  Split:         Activates 1+ branches where              │
│                 condition = true                         │
│  Join:          Waits for all ACTIVE tokens               │
│  Default:       Recommended (isDefault: true)            │
│  convergeAtTaskId: Required if branches should merge     │
│                                                          │
│  API Payload:                                            │
│  {                                                       │
│    "type": "INCLUSIVE",                                   │
│    "name": "Gateway Label",                              │
│    "afterTaskId": <taskId>,                              │
│    "convergeAtTaskId": <taskId>,  // optional            │
│    "branches": [                                         │
│      { "targetTaskId": <id>, "condition": "..." },       │
│      { "targetTaskId": <id>, "isDefault": true }         │
│    ]                                                     │
│  }                                                       │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Related Documentation

- [BPMN Gateway API Documentation](./BPMN_GATEWAY_API.md) — Full gateway API reference
- [Parallel Gateway endTaskId Examples](./PARALLEL_GATEWAY_ENDTASKID_EXAMPLES.md) — Parallel gateway specifics
- [BPMN 2.0 Specification](https://www.omg.org/spec/BPMN/2.0/) — Official OMG BPMN 2.0 spec
