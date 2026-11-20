//! Console panel: live, filterable, virtualised log viewer.

use super::style::{COLOR_DEBUG, COLOR_ERROR, COLOR_INFO, COLOR_TRACE, COLOR_WARN};
use dear_imgui_rs::{ListClipper, Ui, WindowKey};
use rustiny_core::logging::{LogBuffer, LogEntry};
use tracing::Level;

/// Console filtering state.
#[derive(Debug)]
pub struct State {
    /// Show `ERROR` records.
    pub show_error: bool,
    /// Show `WARN` records.
    pub show_warn: bool,
    /// Show `INFO` records.
    pub show_info: bool,
    /// Show `DEBUG` records.
    pub show_debug: bool,
    /// Show `TRACE` records.
    pub show_trace: bool,
    /// Keep the view pinned to the newest record.
    pub autoscroll: bool,
    /// Case-insensitive substring filter.
    pub filter: String,
    visible: Vec<usize>,
    last_seq: u64,
    last_filter: String,
    last_mask: u8,
    last_len: usize,
}

impl Default for State {
    fn default() -> Self {
        Self {
            show_error: true,
            show_warn: true,
            show_info: true,
            show_debug: false,
            show_trace: false,
            autoscroll: true,
            filter: String::new(),
            visible: Vec::new(),
            last_seq: u64::MAX,
            last_filter: String::new(),
            last_mask: 0,
            last_len: 0,
        }
    }
}

impl State {
    fn mask(&self) -> u8 {
        u8::from(self.show_error)
            | u8::from(self.show_warn) << 1
            | u8::from(self.show_info) << 2
            | u8::from(self.show_debug) << 3
            | u8::from(self.show_trace) << 4
    }

    /// `true` if `entry` passes the level and text filters.
    pub fn accepts(&self, entry: &LogEntry, needle_lower: &str) -> bool {
        let level_ok = match entry.level {
            Level::ERROR => self.show_error,
            Level::WARN => self.show_warn,
            Level::INFO => self.show_info,
            Level::DEBUG => self.show_debug,
            Level::TRACE => self.show_trace,
        };
        level_ok
            && (needle_lower.is_empty()
                || entry.message.to_lowercase().contains(needle_lower)
                || entry.target.to_lowercase().contains(needle_lower))
    }

    /// Recompute the visible index list only when the log, filter or level mask changed.
    fn refresh(&mut self, log: &LogBuffer) {
        let seq = log.next_seq();
        let mask = self.mask();
        let len = log.len();
        if seq == self.last_seq
            && mask == self.last_mask
            && len == self.last_len
            && self.filter == self.last_filter
        {
            return;
        }
        let needle = self.filter.to_lowercase();
        let mut visible = std::mem::take(&mut self.visible);
        visible.clear();
        log.with_entries(|entries| {
            visible.extend(
                entries
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| self.accepts(e, &needle))
                    .map(|(i, _)| i),
            );
        });
        self.visible = visible;
        self.last_seq = seq;
        self.last_mask = mask;
        self.last_len = len;
        self.last_filter.clone_from(&self.filter);
    }
}

fn level_color(level: Level) -> [f32; 4] {
    match level {
        Level::ERROR => COLOR_ERROR,
        Level::WARN => COLOR_WARN,
        Level::INFO => COLOR_INFO,
        Level::DEBUG => COLOR_DEBUG,
        Level::TRACE => COLOR_TRACE,
    }
}

/// Live log viewer. The visible list is virtualised, so cost is proportional to what is on
/// screen, not to the (up to thousands of) retained records.
pub fn draw(ui: &Ui, key: &WindowKey, open: &mut bool, log: &LogBuffer, state: &mut State) {
    if !*open {
        return;
    }
    ui.window(key).opened(open).build(|| {
        if ui.button("Clear") {
            log.clear();
        }
        ui.same_line();
        ui.checkbox("Errors", &mut state.show_error);
        ui.same_line();
        ui.checkbox("Warnings", &mut state.show_warn);
        ui.same_line();
        ui.checkbox("Info", &mut state.show_info);
        ui.same_line();
        ui.checkbox("Debug", &mut state.show_debug);
        ui.same_line();
        ui.checkbox("Trace", &mut state.show_trace);
        ui.same_line();
        ui.checkbox("Auto-scroll", &mut state.autoscroll);
        ui.input_text("Filter", &mut state.filter).build();
        ui.separator();

        state.refresh(log);
        ui.child_window("console-scroll").build(ui, || {
            log.with_entries(|entries| {
                let clipper = ListClipper::new(state.visible.len()).begin(ui);
                for row in clipper.iter() {
                    let Some(entry) = state.visible.get(row).and_then(|&i| entries.get(i)) else {
                        continue;
                    };
                    let line = format!(
                        "[{:>8.3}] {:<5} {}: {}",
                        entry.elapsed.as_secs_f64(),
                        entry.level,
                        entry.target,
                        entry.message
                    );
                    ui.text_colored(level_color(entry.level), line);
                }
            });
            if state.autoscroll && ui.scroll_y() >= ui.scroll_max_y() - 1.0 {
                ui.set_scroll_here_y(1.0);
            }
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn entry(level: Level, msg: &str) -> LogEntry {
        LogEntry {
            seq: 0,
            level,
            target: "t".into(),
            message: msg.into(),
            elapsed: Duration::ZERO,
        }
    }

    #[test]
    fn level_and_text_filters_combine() {
        let mut s = State::default();
        assert!(s.accepts(&entry(Level::ERROR, "boom"), ""));
        assert!(
            !s.accepts(&entry(Level::DEBUG, "boom"), ""),
            "debug hidden by default"
        );
        s.show_debug = true;
        assert!(s.accepts(&entry(Level::DEBUG, "boom"), ""));
        assert!(s.accepts(&entry(Level::INFO, "Hello World"), "world"));
        assert!(!s.accepts(&entry(Level::INFO, "Hello World"), "mars"));
    }

    #[test]
    fn refresh_only_recomputes_on_change() {
        let log = LogBuffer::new(16);
        let mut s = State::default();
        log.push(Level::INFO, "t", "one".into());
        log.push(Level::DEBUG, "t", "hidden".into());
        s.refresh(&log);
        assert_eq!(s.visible, [0]);
        // Sentinel proves the cached list is reused when nothing changed.
        s.visible.push(99);
        s.refresh(&log);
        assert_eq!(s.visible, [0, 99]);
        log.push(Level::WARN, "t", "two".into());
        s.refresh(&log);
        assert_eq!(s.visible, [0, 2]);
        s.show_debug = true;
        s.refresh(&log);
        assert_eq!(s.visible, [0, 1, 2]);
    }
}
