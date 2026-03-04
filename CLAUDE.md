# DINO Project Rules

## Git Workflow

- Integration branch: `develop` (never commit directly to `develop` or `main`)
- Stable branch: `main` (promoted from `develop` at milestones)
- Always use `--base develop` with `gh pr create`
- Branch naming: `feat/<name>`, `fix/<name>`, `refactor/<name>`, `chore/<name>`
- Create a GitHub issue before coding (provides traceability)
- Link PRs to issues with `Closes #N` in the PR body

## Review Triangle (Mandatory Before Merge)

Every PR with code changes must complete the full review triangle before merging:

1. **Implement** — write code + tests on a feature branch
2. **Review** — run ALL 4 review agents in parallel:
   - `pr-review-toolkit:code-reviewer` — bugs, logic errors, conventions
   - `pr-review-toolkit:silent-failure-hunter` — error handling, fallback behavior
   - `pr-review-toolkit:type-design-analyzer` — type design quality
   - `pr-review-toolkit:pr-test-analyzer` — test coverage completeness
3. **Triage** — wait for ALL agents to complete, then categorize findings as FIX or ACCEPT
4. **Fix** — apply fixes for FIX items
5. **Re-review** — run review agents again on the fixed code
6. **Repeat** — continue until a round produces no new actionable findings
7. **Merge** — only after a clean review round

Never merge on partial review results. Never skip the re-review after fixes.

**Exempt from review triangle:** config-only, docs-only, and CI-only PRs (no production code changes).

## Development

- Use `uv` for all Python tooling (dependencies, virtualenv, running tests)
- Run tests: `uv run pytest tests/ -v`
- TDD: write tests first where applicable
- Keep `TableStateMachine` alongside new zone system (no deletion)

## Project Structure

- `src/dino/` — main package
- `src/dino/spatial/` — spatial projection layer (localizers, registry, projection math)
- `src/dino/zones/` — domain-agnostic zone system (zone manager, event rules)
- `tests/` — pytest tests (class-based, mirrors source structure)
