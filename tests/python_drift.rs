//! Regenerates the drift-check fixture corpus, then shells out to
//! `tests/python_drift_check.py` to compare the Rust quantized
//! bound against the Python float reference. Skipped if `python3` or
//! `numpy` aren't on PATH, or if `CARGO_MANIFEST_DIR` is unset.

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> Option<PathBuf> {
    let manifest = std::env::var_os("CARGO_MANIFEST_DIR")?;
    Some(PathBuf::from(manifest))
}

#[test]
fn python_float_drift_check() {
    let Some(root) = repo_root() else {
        eprintln!("CARGO_MANIFEST_DIR is not set; skipping python drift check");
        return;
    };
    // Skip on workers without python3 + numpy so this test stays
    // optional for minimal CI environments.
    let probe = Command::new("python3")
        .arg("-c")
        .arg("import numpy")
        .status();
    if probe.map(|s| !s.success()).unwrap_or(true) {
        eprintln!("python3 + numpy not available; skipping python drift check");
        return;
    }

    // Regenerate fixtures from the current Rust engine first, so the
    // Python checker compares against fresh quantized bounds.
    // Test-local precision for the regenerated drift corpus.
    let dump = Command::new(env!("CARGO"))
        .args(["run", "--quiet", "--bin", "panda_fixture_check", "--", "14"])
        .current_dir(&root)
        .status()
        .expect("cargo run panda_fixture_check");
    assert!(dump.success(), "panda_fixture_check binary failed");

    let out = Command::new("python3")
        .args(["-m", "tests.python_drift_check"])
        .arg(root.join("evaluation").join("benchmarks").join("drift"))
        .arg("--tolerance")
        .arg("0.05")
        .current_dir(&root)
        .output()
        .expect("python drift check");
    if !out.status.success() {
        eprintln!("STDOUT:\n{}", String::from_utf8_lossy(&out.stdout));
        eprintln!("STDERR:\n{}", String::from_utf8_lossy(&out.stderr));
        panic!("python drift check exceeded tolerance");
    }
}
