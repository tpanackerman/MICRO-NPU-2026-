//! Cheap wall-clock accounting for the serving path, printed to stderr (server.log).

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

thread_local! {
    static TABLE: RefCell<BTreeMap<&'static str, (u64, Duration)>> = const { RefCell::new(BTreeMap::new()) };
}

pub fn add(name: &'static str, elapsed: Duration) {
    TABLE.with(|table| {
        let mut table = table.borrow_mut();
        let entry = table.entry(name).or_insert((0, Duration::ZERO));
        entry.0 += 1;
        entry.1 += elapsed;
    });
}

/// Prints and clears the table: `name=count/total_ms`.
pub fn flush(tag: &str) {
    TABLE.with(|table| {
        let mut table = table.borrow_mut();
        let mut line = String::from(tag);
        for (name, (count, total)) in table.iter() {
            line.push_str(&format!(" {name}={count}/{:.1}", total.as_secs_f64() * 1e3));
        }
        eprintln!("{line}");
        table.clear();
    });
}

/// Times an awaited expression under `name`.
macro_rules! timed {
    ($name:expr, $body:expr) => {{
        let started = std::time::Instant::now();
        let value = $body;
        $crate::host::prof::add($name, started.elapsed());
        value
    }};
}
pub(crate) use timed;

pub fn since(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1e3
}
