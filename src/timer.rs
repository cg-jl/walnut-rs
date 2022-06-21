//! Clone of walnut's Timer utilities.

use std::{io, time};

/// Simple timer structure that encapsulates the `start` [`Instant`](time::Instant).
///
/// To instantiate a [`Timer`] use its [`Default`] instance:
/// ```
/// # use walnut::timer::Timer;
/// let timer = Timer::default();
/// assert_eq!(timer.elapsed().as_millis(), 0);
/// ```
/// This implementation differs from the original in the [`elapsed`](Timer::elapsed) method,
/// because a richer structure for time differences [`Duration`](time::Duration) exists.
#[derive(Debug)]
pub struct Timer {
    start: time::Instant,
}

impl Default for Timer {
    fn default() -> Self {
        Self {
            start: time::Instant::now(),
        }
    }
}

impl Timer {
    pub fn reset(&mut self) {
        self.start = time::Instant::now();
    }
    pub fn elapsed(&self) -> time::Duration {
        time::Instant::now() - self.start
    }
}

/// Have a timer for a specific thing. It prints its elapsed time in milliseconds when dropping.
#[derive(Debug)]
pub struct ScopedTimer {
    name: String,
    timer: Timer,
}

impl ScopedTimer {
    pub fn new(name: String) -> Self {
        Self {
            name,
            timer: Timer::default(),
        }
    }
}

impl Drop for ScopedTimer {
    fn drop(&mut self) {
        let time = self.timer.elapsed().as_millis();
        let name = &self.name;
        let mut stdout = io::stdout().lock();
        use io::Write;
        let _ = writeln!(&mut stdout, "[TIMER] {name} - {time}ms");
    }
}
