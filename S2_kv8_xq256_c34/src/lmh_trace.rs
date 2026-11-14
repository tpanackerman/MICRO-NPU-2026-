//! On-device cycle collection for the lmhead test binaries (armed only when FURIOSA_OPT_PROFILE is info|debug|trace).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

const TRACING_TARGET_NPU: &str = "span::npu";

#[derive(Clone, Copy)]
struct Span {
    cluster: u64,
    begin: u64,
    end: u64,
}

#[derive(Clone, Default)]
pub struct Collector {
    spans: Arc<Mutex<Vec<Span>>>,
    next_id: Arc<AtomicU64>,
}

impl Collector {
    pub fn install() -> Option<Self> {
        let level = std::env::var("FURIOSA_OPT_PROFILE").unwrap_or_default().to_ascii_lowercase();
        if !matches!(level.as_str(), "info" | "debug" | "trace") {
            return None;
        }
        let collector = Self::default();
        tracing::subscriber::set_global_default(collector.clone()).expect("set global tracing subscriber");
        Some(collector)
    }

    pub fn clear(&self) {
        self.spans.lock().unwrap().clear();
    }

    /// Longest per-cluster window (cycles) of whatever ran since the last `clear`.
    pub fn window_cycles(&self) -> Option<u64> {
        let spans = self.spans.lock().unwrap();
        let mut per_cluster: HashMap<u64, (u64, u64)> = HashMap::new();
        for span in spans.iter() {
            let window = per_cluster.entry(span.cluster).or_insert((span.begin, span.end));
            window.0 = window.0.min(span.begin);
            window.1 = window.1.max(span.end);
        }
        per_cluster.values().map(|(begin, end)| end.saturating_sub(*begin)).max()
    }
}

#[derive(Default)]
struct FieldExtractor {
    begin: Option<u64>,
    end: Option<u64>,
    cluster: Option<u64>,
}

impl tracing::field::Visit for FieldExtractor {
    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        match field.name() {
            "begin_cycle" => self.begin = Some(value),
            "end_cycle" => self.end = Some(value),
            "tid" => self.cluster = Some(value),
            _ => {}
        }
    }

    fn record_debug(&mut self, _field: &tracing::field::Field, _value: &dyn std::fmt::Debug) {}
}

impl tracing::Subscriber for Collector {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        metadata.target() == TRACING_TARGET_NPU
    }

    fn new_span(&self, attrs: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        if attrs.metadata().target() == TRACING_TARGET_NPU {
            let mut extractor = FieldExtractor::default();
            attrs.record(&mut extractor);
            if let (Some(cluster), Some(begin), Some(end)) = (extractor.cluster, extractor.begin, extractor.end) {
                self.spans.lock().unwrap().push(Span { cluster, begin, end });
            }
        }
        tracing::span::Id::from_u64(self.next_id.fetch_add(1, Ordering::Relaxed) + 1)
    }

    fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}
    fn event(&self, _event: &tracing::Event<'_>) {}
    fn enter(&self, _span: &tracing::span::Id) {}
    fn exit(&self, _span: &tracing::span::Id) {}
}
