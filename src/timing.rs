//! Process-global prover timing registry.
//!
//! Named RAII scopes accumulate wall-clock nanoseconds into a shared
//! map; [`snapshot_json`] serialises it as the one-line
//! `prover components v1: {...}` payload the evaluation harness
//! parses. Global (not thread-local) because the prover is logically
//! sequential — one FS transcript — so a scope stack shared behind an
//! uncontended [`Mutex`] is sound and effectively free.

use std::collections::BTreeMap;
use std::sync::{LazyLock, Mutex, MutexGuard};
use std::time::Instant;

struct Registry {
    nanos: BTreeMap<String, u128>,
    stack: Vec<&'static str>,
}

static REGISTRY: LazyLock<Mutex<Registry>> = LazyLock::new(|| {
    Mutex::new(Registry {
        nanos: BTreeMap::new(),
        stack: Vec::new(),
    })
});

// Timing must never poison-panic the prover.
fn lock() -> MutexGuard<'static, Registry> {
    REGISTRY.lock().unwrap_or_else(|e| e.into_inner())
}

enum GuardKind {
    /// Named scope: pushed on the stack; elapsed accumulates under its
    /// own name.
    Scope(&'static str),
    /// Lookup memo: elapsed accumulates under `<innermost scope>_lu`
    /// (or `orphan_lu` with an empty stack); no stack push.
    Lookup,
    /// Plain counter: elapsed accumulates under its name, no stack
    /// interaction (cross-cutting PCS totals).
    Counter(&'static str),
    /// `nc_tangent`: accumulates only while `nc_total` is on the stack.
    Tangent,
}

pub(crate) struct Guard {
    kind: GuardKind,
    start: Instant,
}

pub(crate) fn scope(name: &'static str) -> Guard {
    lock().stack.push(name);
    Guard {
        kind: GuardKind::Scope(name),
        start: Instant::now(),
    }
}

pub(crate) fn lu_scope() -> Guard {
    Guard {
        kind: GuardKind::Lookup,
        start: Instant::now(),
    }
}

pub(crate) fn counter(name: &'static str) -> Guard {
    Guard {
        kind: GuardKind::Counter(name),
        start: Instant::now(),
    }
}

pub(crate) fn tangent_scope() -> Guard {
    Guard {
        kind: GuardKind::Tangent,
        start: Instant::now(),
    }
}

pub(crate) fn add(name: &str, nanos: u128) {
    let mut r = lock();
    *r.nanos.entry(name.to_string()).or_insert(0) += nanos;
}

pub(crate) fn add_tangent(nanos: u128) {
    let mut r = lock();
    // Guards against the standalone cert calls outside the prover
    // (benchmark bound recompute, crown_float_eval binary).
    if r.stack.contains(&"nc_total") {
        *r.nanos.entry("nc_tangent".to_string()).or_insert(0) += nanos;
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed().as_nanos();
        match self.kind {
            GuardKind::Scope(name) => {
                let mut r = lock();
                *r.nanos.entry(name.to_string()).or_insert(0) += elapsed;
                let _ = r.stack.pop();
            }
            GuardKind::Lookup => {
                let mut r = lock();
                let key = match r.stack.last() {
                    Some(inner) => format!("{inner}_lu"),
                    None => "orphan_lu".to_string(),
                };
                *r.nanos.entry(key).or_insert(0) += elapsed;
            }
            GuardKind::Counter(name) => add(name, elapsed),
            GuardKind::Tangent => add_tangent(elapsed),
        }
    }
}

/// Clear all accumulated timings and the scope stack. The harness
/// calls this immediately before each `prove_final_pass`.
pub fn reset() {
    let mut r = lock();
    r.nanos.clear();
    r.stack.clear();
}

/// Serialise the registry as the contract JSON object: sorted keys,
/// values in seconds with 6 decimals, zero-valued keys omitted except
/// `nc_total` / `zk_total` (always present).
pub fn snapshot_json() -> String {
    let r = lock();
    let mut out = String::from("{");
    let mut first = true;
    let always = |k: &str| k == "nc_total" || k == "zk_total";
    let mut entries: BTreeMap<&str, u128> = r.nanos.iter().map(|(k, &v)| (k.as_str(), v)).collect();
    entries.entry("nc_total").or_insert(0);
    entries.entry("zk_total").or_insert(0);
    for (k, v) in entries {
        if v == 0 && !always(k) {
            continue;
        }
        if !first {
            out.push(',');
        }
        first = false;
        out.push_str(&format!("\"{}\":{:.6}", k, v as f64 / 1e9));
    }
    out.push('}');
    out
}
