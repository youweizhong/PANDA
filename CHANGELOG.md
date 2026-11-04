# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added the NeurIPS 2026 acceptance note and paper link to the README.
- Added `CITATION.cff` with preferred citation metadata for the PANDA paper accepted at NeurIPS 2026.

- Added third-party credits and generated Rust and Python dependency-license reports.
- Added a security policy covering supported versions, private vulnerability reporting, scope, and coordinated disclosure.
- Added shared repository instructions for coding agents in `AGENTS.md`.

### Changed

- Removed `.gitignore` exclusions for agent instructions and local tooling configuration so they can be tracked in the repository.

### Fixed

- Corrected the fixture property documentation to describe `C f(x) + d` and the non-strict inequalities selected by `side`.

### Security

- Hyrax openings now bind the claimed evaluation to the commitment. The proof carries the opening randomness `r_eval` of `com_eval`, and the verifier checks `com_eval == value * G0 + r_eval * h` for single and batched openings. Before this, a prover that absorbed a false value into the transcript could pass verification. This changes the proof format (one more field element per opening), so proofs made by earlier versions are not compatible.
- The sigmoid and tanh critical-point verifier now rejects a failed Hyrax opening of the envelope multiplicities (`envelope_mult_open`). It used to discard that result, so the table side of the envelope LogUp was not bound to the committed multiplicities.
