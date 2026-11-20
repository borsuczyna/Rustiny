//! Low-overhead CPU timing with percentile statistics.
//!
//! The profiler keeps fixed-size windows of recent samples (no unbounded growth, no allocation
//! after warm-up on the hot path) and can report the median, P95, P99 and extremes that the
//! performance targets in the design document are expressed in.

#![cfg_attr(test, allow(clippy::unwrap_used))]

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Summary statistics over a [`SampleWindow`], in the unit of the samples (milliseconds for the
/// frame profiler).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Stats {
    /// Number of samples summarised.
    pub count: usize,
    /// Smallest sample.
    pub min: f64,
    /// Largest sample.
    pub max: f64,
    /// Arithmetic mean.
    pub mean: f64,
    /// Median (P50).
    pub median: f64,
    /// 95th percentile (nearest-rank).
    pub p95: f64,
    /// 99th percentile (nearest-rank).
    pub p99: f64,
}

/// Fixed-capacity ring of the most recent samples.
#[derive(Debug, Clone)]
pub struct SampleWindow {
    samples: VecDeque<f64>,
    capacity: usize,
}

impl SampleWindow {
    /// Create a window holding at most `capacity` samples (minimum 1).
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            samples: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Add a sample, discarding the oldest one if the window is full. Non-finite values are
    /// ignored so that one bad measurement cannot poison every statistic.
    pub fn push(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        if self.samples.len() == self.capacity {
            self.samples.pop_front();
        }
        self.samples.push_back(value);
    }

    /// Number of samples currently held.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// `true` if no sample has been recorded.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Most recent sample.
    pub fn last(&self) -> Option<f64> {
        self.samples.back().copied()
    }

    /// Iterate oldest → newest.
    pub fn iter(&self) -> impl Iterator<Item = f64> + '_ {
        self.samples.iter().copied()
    }

    /// Copy samples oldest → newest into `out` as `f32` (for plotting widgets), reusing its
    /// allocation.
    pub fn copy_f32_into(&self, out: &mut Vec<f32>) {
        out.clear();
        out.extend(self.samples.iter().map(|&v| v as f32));
    }

    /// Remove all samples.
    pub fn clear(&mut self) {
        self.samples.clear();
    }

    /// Compute summary statistics. Returns the all-zero default for an empty window.
    pub fn stats(&self) -> Stats {
        if self.samples.is_empty() {
            return Stats::default();
        }
        let mut sorted: Vec<f64> = self.samples.iter().copied().collect();
        sorted.sort_by(f64::total_cmp);
        let n = sorted.len();
        let sum: f64 = sorted.iter().sum();
        Stats {
            count: n,
            min: sorted[0],
            max: sorted[n - 1],
            mean: sum / n as f64,
            median: percentile_sorted(&sorted, 50.0),
            p95: percentile_sorted(&sorted, 95.0),
            p99: percentile_sorted(&sorted, 99.0),
        }
    }
}

/// Nearest-rank percentile of an ascending-sorted, non-empty slice.
fn percentile_sorted(sorted: &[f64], pct: f64) -> f64 {
    debug_assert!(!sorted.is_empty());
    let rank = ((pct / 100.0) * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// Per-frame timing recorder with named scopes.
#[derive(Debug)]
pub struct FrameProfiler {
    window: usize,
    frame_start: Option<Instant>,
    last_frame_start: Option<Instant>,
    /// Time between consecutive `begin_frame` calls (includes idle/wait time).
    frame_interval_ms: SampleWindow,
    /// Time spent between `begin_frame` and `end_frame` (CPU work for the frame).
    cpu_ms: SampleWindow,
    scopes: HashMap<&'static str, SampleWindow>,
    frames: u64,
}

impl FrameProfiler {
    /// Create a profiler remembering the last `window` frames.
    pub fn new(window: usize) -> Self {
        Self {
            window,
            frame_start: None,
            last_frame_start: None,
            frame_interval_ms: SampleWindow::new(window),
            cpu_ms: SampleWindow::new(window),
            scopes: HashMap::new(),
            frames: 0,
        }
    }

    /// Mark the start of a frame.
    pub fn begin_frame(&mut self) {
        let now = Instant::now();
        if let Some(prev) = self.last_frame_start.replace(now) {
            self.frame_interval_ms.push(ms(now - prev));
        }
        self.frame_start = Some(now);
    }

    /// Mark the end of a frame; returns its CPU time.
    pub fn end_frame(&mut self) -> Option<Duration> {
        let start = self.frame_start.take()?;
        let elapsed = start.elapsed();
        self.cpu_ms.push(ms(elapsed));
        self.frames += 1;
        Some(elapsed)
    }

    /// Record the duration of a named scope for the current frame.
    pub fn record_scope(&mut self, name: &'static str, duration: Duration) {
        let window = self.window;
        self.scopes
            .entry(name)
            .or_insert_with(|| SampleWindow::new(window))
            .push(ms(duration));
    }

    /// Time `f`, record it under `name` and return its result.
    pub fn measure<R>(&mut self, name: &'static str, f: impl FnOnce() -> R) -> R {
        let start = Instant::now();
        let result = f();
        self.record_scope(name, start.elapsed());
        result
    }

    /// Frames completed so far.
    pub fn frame_count(&self) -> u64 {
        self.frames
    }

    /// Frame-to-frame interval samples in milliseconds.
    pub fn frame_interval(&self) -> &SampleWindow {
        &self.frame_interval_ms
    }

    /// Per-frame CPU time samples in milliseconds.
    pub fn cpu_time(&self) -> &SampleWindow {
        &self.cpu_ms
    }

    /// Samples of a named scope, if it has ever been recorded.
    pub fn scope(&self, name: &str) -> Option<&SampleWindow> {
        self.scopes.get(name)
    }

    /// All recorded scopes, sorted by name for stable display.
    pub fn scopes_sorted(&self) -> Vec<(&'static str, &SampleWindow)> {
        let mut v: Vec<_> = self.scopes.iter().map(|(k, w)| (*k, w)).collect();
        v.sort_by_key(|(k, _)| *k);
        v
    }

    /// Forget all samples (e.g. when switching between idle and continuous redraw).
    pub fn reset(&mut self) {
        self.frame_interval_ms.clear();
        self.cpu_ms.clear();
        self.scopes.clear();
        self.last_frame_start = None;
        self.frame_start = None;
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_window_has_zero_stats() {
        assert_eq!(SampleWindow::new(4).stats(), Stats::default());
    }

    #[test]
    fn nearest_rank_percentiles() {
        let mut w = SampleWindow::new(100);
        for v in 1..=100 {
            w.push(f64::from(v));
        }
        let s = w.stats();
        assert_eq!(s.count, 100);
        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 100.0);
        assert_eq!(s.mean, 50.5);
        assert_eq!(s.median, 50.0);
        assert_eq!(s.p95, 95.0);
        assert_eq!(s.p99, 99.0);
    }

    #[test]
    fn single_sample_is_every_percentile() {
        let mut w = SampleWindow::new(4);
        w.push(7.5);
        let s = w.stats();
        assert_eq!(
            (s.min, s.median, s.p95, s.p99, s.max),
            (7.5, 7.5, 7.5, 7.5, 7.5)
        );
    }

    #[test]
    fn window_evicts_oldest_and_ignores_non_finite() {
        let mut w = SampleWindow::new(3);
        for v in [1.0, 2.0, 3.0, f64::NAN, f64::INFINITY, 4.0] {
            w.push(v);
        }
        assert_eq!(w.iter().collect::<Vec<_>>(), [2.0, 3.0, 4.0]);
        assert_eq!(w.last(), Some(4.0));
    }

    #[test]
    fn copy_f32_reuses_buffer() {
        let mut w = SampleWindow::new(4);
        w.push(1.0);
        w.push(2.0);
        let mut out = vec![9.0; 10];
        w.copy_f32_into(&mut out);
        assert_eq!(out, [1.0, 2.0]);
    }

    #[test]
    fn frame_profiler_tracks_frames_and_scopes() {
        let mut p = FrameProfiler::new(16);
        for _ in 0..3 {
            p.begin_frame();
            p.measure("work", || std::thread::sleep(Duration::from_millis(2)));
            assert!(p.end_frame().unwrap() >= Duration::from_millis(2));
        }
        assert_eq!(p.frame_count(), 3);
        assert_eq!(p.cpu_time().len(), 3);
        assert_eq!(
            p.frame_interval().len(),
            2,
            "interval needs two frame starts"
        );
        let work = p.scope("work").unwrap().stats();
        assert_eq!(work.count, 3);
        assert!(work.min >= 2.0);
        assert!(p.scope("missing").is_none());
    }

    #[test]
    fn end_frame_without_begin_is_none() {
        assert!(FrameProfiler::new(4).end_frame().is_none());
    }
}
