# DEVELOPMENT_RULES.md — CineMind AI

Binding rules for any human or AI agent working on this codebase. These override convenience/speed whenever they conflict.

## Scope Discipline
1. Build incrementally, in the milestone order defined in `ROADMAP.md`. Do not jump ahead to later-tier (V2/V3) features while an earlier tier is incomplete.
2. Do not build the entire application in one pass. One milestone at a time, verified before moving on.
3. Do not create files, folders, or dependencies that aren't needed for the current milestone.
4. Do not over-engineer the current version (no premature microservices, no unused abstraction layers, no speculative config).

## Decision-Making
5. Do not make architectural decisions silently. If a milestone requires a decision not already covered in `ARCHITECTURE.md`, state the options and reasoning before proceeding.
6. Before any major architectural change (framework swap, schema redesign, new external service), explain the reason first and get confirmation.
7. Never delete or rewrite working code unless the task requires it — prefer additive, backward-compatible changes.
8. Preserve backward compatibility when adding features, unless a breaking change is explicitly required and called out.

## Code Structure
9. Keep API routes, business/service logic, database models, ML/NLP logic, and utilities in separate modules — never mix concerns in one file.
10. Keep all ML/NLP code in its own package, isolated from FastAPI route code, so it can be tested and reasoned about independently.
11. Use Python type hints throughout.
12. Use Pydantic schemas for all API request/response validation — no raw untyped dicts across the API boundary.

## Data & Secrets
13. Never hardcode API keys, passwords, tokens, or secrets anywhere in code or committed files.
14. All secrets/config come from environment variables (`.env`, not committed; `.env.example` committed as a template).
15. Do not use mock/placeholder data for anything that will eventually use real data (e.g., TMDB movies, real user reviews) beyond the earliest scaffolding stage — replace mocks with real integrations as soon as the relevant milestone is reached.

## Quality & Security
16. Write tests for meaningful functionality (auth, data mutations, recommendation correctness) — not necessarily 100% coverage, but core flows must be tested.
17. Follow security best practices: hashed passwords, JWT handled correctly, input validation, no secrets in logs, sensible CORS config.
18. Keep the project deployable at all times — avoid states where the app can't be built/run/deployed for extended periods.

## Understandability
19. The project must stay explainable by a BTech student in a viva/interview — favor clear, conventional approaches over clever-but-opaque ones. If a technique needs a research paper to justify, document *why* it was chosen in plain language in the relevant `*_PLAN.md` file.
20. Any new planning/architecture decision should be reflected in the relevant `.md` file (`ARCHITECTURE.md`, `DATABASE_DESIGN.md`, `API_PLAN.md`, `ML_NLP_PLAN.md`) so documentation never drifts from reality.
