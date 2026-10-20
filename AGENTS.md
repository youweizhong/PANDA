# PANDA Agent Instructions

The one contract for every coding agent in this repository.

## Quick Start

Prioritize implementation. During routine development run only the smallest tests and checks needed for the changed behavior. Do not run broad tests, or benchmark suites after every edit or commit. Qualify the complete result at the end of each major phase and before merge or release.

## Code Style

- **Warnings are errors.** Fix every compiler/linter warning before considering code done. Never leave warnings.
- **Reuse over reinvention.** Check `src` first. Extend existing abstractions.
- **Minimize code.** Delete dead code, inline single-use helpers. Every line must earn its place.
- **One way to do things.** Don't introduce a second pattern when one exists.

## Commits

1. Update CHANGELOG.md in the same commit **when the change is user-visible**. Refactors, test-only changes and internal cleanups do not need an entry.
2. Stage files explicitly (no `git add -A`)
3. Conventional subject, `type(scope): summary`. In use: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `chore`, `style`, `security`, `merge`.
4. Author: Youwei Zhong youwei.zhong@yale.edu
5. No `Co-Authored-By` trailers
