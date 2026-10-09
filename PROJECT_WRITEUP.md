# ORDIX: Orders Simplified — Project Technical Writeup & Design Document

---

## 1. Executive Summary

**ORDIX** (*Orders Simplified*) is an enterprise-grade, deterministic AI order assistant engineered to deliver instantaneous, zero-hallucination order intelligence and commercial analytics from tabular datasets. Powered by a high-performance, single-process stack comprising **FastAPI**, **vanilla JavaScript (ES6+)**, **Pandas**, and the official **Google Gemini GenAI SDK**, ORDIX eliminates the arithmetic inaccuracies, hallucinated records, and rate-limit fragility typical of naive LLM chatbot wrappers.

By coupling Gemini's natural language comprehension with an autonomous, deterministic **Structured Query Plan (SQP)** engine and a local pandas fast-path classifier, ORDIX answers operational questions (order lookups, status distributions, monthly revenue variance, cancellations, customer spending rankings) in sub-millisecond local execution time with 100% mathematical precision.

---

## 2. Problem Statement & Commercial Motivation

### The Operational Challenge
In modern retail, e-commerce, and logistics environments, operational data resides in spreadsheets and transactional exports (`orders.csv`). Extracting actionable insights presents significant friction:
- **Operational Latency:** Customer support agents and dispatchers must repeatedly filter and cross-tabulate spreadsheets to check shipment statuses or payment modes.
- **Reporting Bottlenecks:** Management inquiries regarding monthly trends, regional sales distribution, or cancelled order losses require custom formulas or ad-hoc assistance from data engineering teams.
- **The "LLM Math Failure" Trap:** Generic LLMs deployed directly on tabular datasets routinely hallucinate numbers, miscalculate sums, fail at chronological group-bys, and invent orders. Because language models predict tokens rather than execute arithmetic, they cannot be trusted with commercial financials without deterministic backend execution.

### The ORDIX Value Proposition
ORDIX was conceived to solve this dilemma through strict **architectural grounding**:
1. **Never Calculate in Prompts:** Calculations are executed exclusively by verified Python/pandas code.
2. **Never Fabricate Data:** Order details are retrieved verbatim from the underlying transactional record.
3. **High Availability:** If the external LLM provider encounters rate limits (`429`) or network downtime, the system fails over transparently to local deterministic query execution without user disruption.

---

## 3. Architecture & Core Innovations

```
                                    ┌────────────────────────────────────────────────────────┐
                                    │                     ORDIX Web Client                   │
                                    │      (Semantic HTML5 / Vanilla CSS / ES6+ JS)          │
                                    └──────────────────────────┬─────────────────────────────┘
                                                               │  POST /api/chat
                                                               ▼
                                    ┌────────────────────────────────────────────────────────┐
                                    │                    FastAPI Backend                     │
                                    │         (Pydantic Validation / Input Sanitation)       │
                                    └──────────────┬──────────────────────────┬──────────────┘
                                                   │                          │
                        Fast-Path Match (Sub-5ms)  │                          │ Complex Natural Language
                                                   ▼                          ▼
                    ┌──────────────────────────────────────┐  ┌───────────────────────────────┐
                    │      Deterministic Fast-Path Router  │  │    Google Gemini GenAI Agent  │
                    │       (Regex / Intent Classifier)    │  │    (Structured Tool Calling)  │
                    └──────────────────┬───────────────────┘  └───────────────┬───────────────┘
                                       │                                      │ Tool Invocations
                                       ▼                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                             Deterministic Execution Layer                                   │
│  ┌────────────────────────┐  ┌───────────────────────────┐  ┌────────────────────────────┐  │
│  │ Local Pandas Engine    │  │ Structured Query Planner  │  │ Backend Python Tools       │  │
│  │ (query_engine.py)      │  │ (query_planner/executor)  │  │ (search / statistics)      │  │
│  └───────────┬────────────┘  └─────────────┬─────────────┘  └─────────────┬──────────────┘  │
└──────────────┼─────────────────────────────┼──────────────────────────────┼─────────────────┘
               │                             │                              │
               └─────────────────────────────┼──────────────────────────────┘
                                             ▼
                               ┌───────────────────────────┐
                               │ Verified CSV Data Store   │
                               │        (orders.csv)       │
                               └───────────────────────────┘
```

ORDIX is structured around five core engineering innovations:

### 3.1 Dynamic Schema Introspection (`schema_inspector.py`)
Unlike brittle chatbots hardcoded to specific columns, ORDIX dynamically introspects `orders.csv`:
- Inspects column names, inferred data types (strings, numeric, ISO dates), missing value ratios, and cardinality.
- Extracts and caches distinct categorical values (cities, order statuses, payment methods, product categories) with automatic invalidation based on file modification timestamp (`mtime`).
- Generates compact, token-efficient schema context for query planning without loading raw rows into LLM context windows.

### 3.2 Declarative Structured Query Planning (`query_plan.py` & `query_planner.py`)
Natural language inquiries are parsed into strongly validated Pydantic `QueryPlan` specifications:
- **Filters:** Equality, inclusion, case-insensitive substring search, and datetime range boundaries (`start_date`, `end_date`).
- **Aggregations:** Supported operators include `sum`, `count`, `mean`, `min`, `max`, `distinct_values`, and `percentage`.
- **Grouping & Pivoting:** Arbitrary column grouping with automatic computation of grand totals.
- **Sorting & Slicing:** Order direction (`asc`, `desc`) and row limits (e.g., top 5 spenders, bottom 3 customers).

### 3.3 Deterministic Execution Engine (`plan_executor.py` & `query_engine.py`)
The query plans and mathematical operations execute deterministically via Pandas:
- **Zero Hallucination:** Computations rely on verified vector and aggregation operations.
- **Financial Clarity:** Explicitly distinguishes **gross order value**, **net active revenue**, and **delivered revenue**, ensuring cancelled order amounts (₹72,634.00) are never erroneously counted toward delivered revenue.
- **Safe Execution:** Implements strict operator whitelisting with zero `eval()`, zero `exec()`, and no arbitrary code execution.

### 3.4 Hybrid Intelligent Routing (`router.py`)
To minimize external API costs, latency, and token consumption, incoming queries pass through a high-performance fast-path classifier:
- Detects recurring intent patterns (e.g., specific Order ID lookups, monthly revenue requests, cancelled order queries, city customer inquiries, top customer requests).
- Directly executes matched queries via the local engine in **under 5 milliseconds** with **0 Gemini API calls**.
- Seamlessly hands off ambiguous, conversational, or unstructured queries to Gemini with tool definitions.

### 3.5 Quota-Exhaustion Resilience & Graceful Fallback (`agent.py`)
If Gemini returns a `429 Too Many Requests` (quota limit) or `503 Service Unavailable`, ORDIX catches the provider exception and automatically runs the local query planner. The user receives an accurate, grounded answer rather than a fatal error screen.

---

## 4. Backend Capabilities & Tool Specifications

| Tool / Subsystem | Primary Function | Input Arguments | Output Representation |
| :--- | :--- | :--- | :--- |
| `search_orders` | Multi-field lookup & filtering across order ID, customer, status, category, city, payment, date range | `order_id`, `customer_name`, `status`, `category`, `city`, `payment_method`, `date_from`, `date_to`, `limit` | JSON array of matched order records, total count, and structured card metadata |
| `calculate_order_statistics` | Pre-aggregated verified mathematical summaries | `operation` (`summary`, `order_counts_by_status`, `cancelled_and_pending`, `revenue_total`, `monthly_revenue`, `revenue_by_category`, `top_customers`, `revenue_by_city`), `filters` | Statistical summaries, percentages, currency aggregates, and chronological month breakdowns |
| `query_engine.py` | Chronological variances, online payment breakdowns, least spender ranking, and context follow-ups | Analytical queries, contextual history | Grounded markdown tables, variance calculations, and direct metric summaries |
| `main.py` REST API | Single-process web endpoints (`/api/chat`, `/api/dashboard`, `/api/suggestions`, `/health`) | JSON payloads with Pydantic schemas | JSON responses, HTTP error codes, telemetry |

---

## 5. Dataset Profile & Ground Truth Verification

The solution is grounded on the verified `orders.csv` transactional dataset comprising **60 orders** spanning June 1, 2026 to September 28, 2026:

| Dimension / Metric | Ground Truth Value | Verification Method |
| :--- | :--- | :--- |
| **Total Order Records** | 60 | Verified via `len(orders)` and Pandas |
| **Gross Order Value** | ₹4,70,312.00 | Sum of all `total_inr` entries |
| **Delivered Orders** | 48 orders (₹3,71,040.00) | `status == 'delivered'` |
| **Cancelled Orders** | 7 orders (₹72,634.00) | `status == 'cancelled'` |
| **Returned Orders** | 3 orders (₹24,292.00) | `status == 'returned'` |
| **Processing Orders** | 1 order (₹1,299.00) | `status == 'processing'` |
| **Shipped Orders** | 1 order (₹1,047.00) | `status == 'shipped'` |
| **Net Revenue (Active)** | ₹3,97,678.00 | Gross minus cancelled revenue |
| **Highest-Spending Customer** | Rohan Das (₹1,03,460.00) | Grouped aggregation by `customer_name` |
| **Lowest-Spending Customer** | Swati Nair (₹1,047.00) | Grouped aggregation by `customer_name` |
| **Top Product Category** | Electronics (₹3,39,715.00) | Grouped aggregation by `category` |
| **Active Cities** | 6 cities (Bengaluru, Chennai, Hyderabad, Kochi, Pune, Thiruvananthapuram) | Distinct categorical scan |

---

## 6. Enterprise Security & Privacy Posture

1. **Zero Secret Leakage:**
   - Production secrets and local API keys are strictly loaded via `.env` and kept in runtime memory.
   - Comprehensive `.gitignore` excludes `.env`, `*.env`, `.env.*`, certificates (`*.pem`, `*.key`), and IDE configurations.
   - Only `.env.example` containing descriptive placeholder tokens is tracked in version control.
2. **API Isolation:**
   - Client applications never access or receive the Gemini API key. All AI invocations are orchestrated server-side.
3. **Payload Sanitization & Boundary Limits:**
   - Inbound chat inquiries are validated through Pydantic `ChatRequest` with a strict 1,000-character upper bound, preventing denial-of-service, token stuffing, and memory exhaustion.
   - Conversation histories are capped at 20 message turns to ensure predictable memory usage.
4. **Execution Safety:**
   - Pure Python and Pandas calculations avoid string interpolation into SQL or Python `eval()`, mitigating injection vectors entirely.

---

## 7. Testing Strategy & Quality Assurance

ORDIX includes an automated test suite of **105 passing tests** across 8 specialized test modules:

```
tests/
├── test_accuracy_and_reliability.py   (10 tests) — Ground truth calculations, quota fallbacks, least spender
├── test_general_analyst.py            (20 tests) — Schema introspection, query plans, variances, groupings
├── test_tools.py                      (21 tests) — Exact search, multi-field filters, 9 statistical operations
├── test_router.py                     (19 tests) — Fast-path matching, fuzzy patterns, quota mock fallbacks
├── test_query_engine.py               (14 tests) — Monthly aggregations, online payments, follow-ups
├── test_api.py                        (11 tests) — FastAPI endpoints, health check, status code assertions
├── test_agent.py                      (5 tests)  — Tool dispatching, iteration bounds, agent turn mocks
└── test_data_loader.py                (5 tests)  — CSV parsing, corrupt header detection, schema validation
────────────────────────────────────────────────────────────────────────────────
TOTAL: 105 passed, 0 failed, 100% test pass rate
```

Every test runs deterministically offline without network calls or external API token consumption, ensuring fast execution and reproducible CI/CD pipelines.

---

## 8. User Experience & Design Philosophy

The ORDIX frontend was developed without heavyweight frontend frameworks to maintain lightning-fast page loads and zero dependency rot:
- **Curated Executive Palette:** Deep Forest Green (`#25483D`), Warm Ivory (`#F5F0E8`), Muted Amber (`#B77943`), Charcoal (`#292F2A`), and Soft Stone (`#D9D0C3`).
- **Typography:** Google Fonts **Inter** for optimal legibility of dense financial numbers.
- **Interactive Feedback:** Quick suggestion chips for common executive queries (*Track an Order*, *Cancelled Orders*, *Monthly Revenue*, *Top Customer*).
- **Responsive Layout:** Adaptive cards and tables formatted with native currency glyphs (`₹`) and dynamic tool execution badges.

---

## 9. Setup & Local Reproduction Guide

### Quick Start (PowerShell / Windows)
```powershell
# 1. Clone repository
git clone <repository-url>
cd order-assistant

# 2. Set up virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install requirements
pip install -r requirements.txt

# 4. Copy environment configuration
Copy-Item .env.example .env
# Edit .env to supply your GEMINI_API_KEY from https://aistudio.google.com/app/apikey

# 5. Run test suite
pytest -v

# 6. Start server
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

---

## 10. Scope & Future Roadmap

### Current Scope & Boundaries
- **Historical Horizon:** Inquiries focus on the June 1, 2026 – September 28, 2026 dataset timeframe.
- **Read-Only Analytics:** Designed for decision support and order discovery; writes/updates to the CSV are not performed.

### Planned Enhancements
1. **Multi-File / SQL Ingestion:** Expand schema introspection to ingest PostgreSQL, SQLite, and multiple relational tables with foreign-key joins.
2. **Streaming AI Responses:** Integrate Server-Sent Events (SSE) for token-by-token streaming responses during complex multi-step inquiries.
3. **Automated Export Capabilities:** Enable instant CSV and PDF downloads of generated summary tables directly from the chat interface.

---

*Author: ORDIX Development Team*  
*License: MIT*
