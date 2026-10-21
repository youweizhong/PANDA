# Security Policy

## Supported versions

Security fixes land on the latest release on main. Older versions are not supported. Upgrade to the latest release to receive fixes.

## Reporting a vulnerability

Please report security issues **privately** via GitHub Security Advisories:

[https://github.com/google/capsem/security/advisories/new](https://github.com/youweizhong/PANDA/security/advisories/new)

You can expect:
- An acknowledgement within 7 days.
- An initial assessment and severity classification within 14 days.
- A fix or mitigation plan within 90 days for high-severity issues.
- Public disclosure coordinated with the reporter after a fix is available.

Do **not** file public GitHub issues for security reports.

## Scope

PANDA is a ZKP system whose job is to certifying robustness properties of neural networks.

- Only **soundness** issues are in scope -- if a malicious prover can forge a proof to a verifier and convince the verifier that a (quantized) neural network satisfies some property that it should not satisfy, then soundness is broken.

## Out of scope

- Completeness and zero-knowledge issues.
- Bugs in unreleased code on development branches.

## Disclosure

Once a fix is released, we will publish an advisory describing the issue, affected versions, and mitigation. Credit is given to reporters unless they request anonymity.
