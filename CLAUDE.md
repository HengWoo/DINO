# DINO Project Rules

## Git

- Integration branch: `develop`
- Stable branch: `main` (promoted from `develop` at milestones)
- Branch naming: `feat/<name>`, `fix/<name>`, `refactor/<name>`, `chore/<name>`

## Project-Specific

- Keep `TableStateMachine` alongside new zone system (no deletion)
- Tests: pytest, class-based, mirrors source structure

## Project Structure

- `src/dino/` — main package
- `src/dino/spatial/` — spatial projection layer (localizers, registry, projection math)
- `src/dino/zones/` — domain-agnostic zone system (zone manager, event rules)
- `tests/` — pytest tests
