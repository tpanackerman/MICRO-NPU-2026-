//! Sleep through the long part of a long device launch instead of spinning on it.
//!
//! The runtime waits for a launch by polling its completion ring: a 200 us spin, then a `yield_now` loop on our single-task runtime, which never
//! parks the thread (the launch thread shows about 0.94 core of CPU in a 2-CPU Arena container, where the runtime's SQPOLL poller already burns
//! another core and the whole container is CFS-throttled in 9 of 10 periods). The long launches (`layer_*_t128` about 0.6 ms, the LM head about 2 ms)
//! spend most of their wait in that loop. After the runtime's own spin has expired, this wrapper sleeps for the part of the launch that is certain
//! to still be running (the shortest duration seen for the same launch kind, minus the elapsed time, minus a margin) and then lets the runtime
//! poll as before. Nothing about the launch, its arguments or its result changes; only the host thread's busy time.
//!
//! The estimate is learned from the launches themselves (no table of device times) and cannot run away: it is the all-time minimum of the observed
//! durations, relaxed upwards by one microsecond per 256 launches, so a late wake-up (which lengthens an observed duration) never raises it. A wake-up
//! that finds the launch already finished widens the margin of that launch kind (additive increase, slow decrease), so a loaded host backs off by itself.
//! `S2_SLEEPWAIT=0` disables it.

use std::cell::RefCell;
use std::future::{Future, IntoFuture};
use std::pin::pin;
use std::task::Poll;
use std::time::{Duration, Instant};

/// Which launch kind a wait belongs to (each kind has its own duration estimate and margin).
#[derive(Clone, Copy)]
pub enum Slot {
    LayerS = 0,
    LayerF = 1,
    Head = 2,
}
const SLOTS: usize = 3;
/// Samples needed before a kind may sleep.
const WARM: u32 = 8;
/// Never sleep for less than this.
const MIN_SLEEP_US: u32 = 60;
/// Default safety margin before the estimated end of a launch (a sleeping thread wakes late by about 60-100 us).
const MARGIN_US: u32 = 150;
/// A wake-up that finds the launch finished widens the margin by this much (at most `EXTRA_MAX_US`); every wake-up that does not narrows it by `EXTRA_DECAY_US`.
const EXTRA_STEP_US: u32 = 50;
const EXTRA_MAX_US: u32 = 800;
const EXTRA_DECAY_US: u32 = 2;
/// The minimum estimate rises by one microsecond per this many launches.
const RELAX_EVERY: u32 = 256;

#[derive(Clone, Copy)]
struct Kind {
    min_us: u32,
    seen: u32,
    relax: u32,
    extra_us: u32,
}

thread_local! {
    static KINDS: RefCell<[Kind; SLOTS]> = const { RefCell::new([Kind { min_us: u32::MAX, seen: 0, relax: 0, extra_us: 0 }; SLOTS]) };
}

/// The safety margin in microseconds, or `None` when sleeping is off.
fn margin_us() -> Option<u32> {
    static MARGIN: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *MARGIN.get_or_init(|| {
        let on = cfg!(backend = "npu") && std::env::var("S2_SLEEPWAIT").map(|value| value != "0").unwrap_or(true);
        on.then_some(MARGIN_US)
    })
}

/// The (estimated end, extra margin) of a launch kind, or `None` while it is still being learned.
fn estimate(slot: Slot) -> Option<(u32, u32)> {
    KINDS.with(|kinds| {
        let kind = kinds.borrow()[slot as usize];
        (kind.seen >= WARM).then_some((kind.min_us, kind.extra_us))
    })
}

fn record(slot: Slot, elapsed: Duration) {
    let us = elapsed.as_micros().min(u128::from(u32::MAX - 1)) as u32;
    KINDS.with(|kinds| {
        let kind = &mut kinds.borrow_mut()[slot as usize];
        kind.seen = kind.seen.saturating_add(1);
        if us < kind.min_us {
            kind.min_us = us;
        } else {
            kind.relax += 1;
            if kind.relax >= RELAX_EVERY {
                kind.relax = 0;
                kind.min_us = kind.min_us.saturating_add(1);
            }
        }
    });
}

/// `late`: the wake-up found the launch already finished.
fn feedback(slot: Slot, late: bool) {
    KINDS.with(|kinds| {
        let kind = &mut kinds.borrow_mut()[slot as usize];
        kind.extra_us = if late { (kind.extra_us + EXTRA_STEP_US).min(EXTRA_MAX_US) } else { kind.extra_us.saturating_sub(EXTRA_DECAY_US) };
    });
}

/// Awaits `launch` like `.await` would; after the runtime's own 200 us spin it sleeps until the margin before the launch's estimated end.
pub async fn sleepy<L: IntoFuture>(slot: Slot, launch: L) -> L::Output {
    let started = Instant::now();
    let mut future = pin!(launch.into_future());
    let mut slept = false;
    let mut after_nap = false;
    let out = std::future::poll_fn(|context| {
        loop {
            let polled = future.as_mut().poll(context);
            if after_nap {
                after_nap = false;
                feedback(slot, polled.is_ready());
            }
            match polled {
                Poll::Pending if !slept => {
                    slept = true;
                    if let (Some(margin), Some((end, extra))) = (margin_us(), estimate(slot)) {
                        let used = started.elapsed().as_micros().min(u128::from(u32::MAX)) as u32;
                        if let Some(nap) = end.checked_sub(used).and_then(|left| left.checked_sub(margin + extra)) {
                            if nap >= MIN_SLEEP_US {
                                std::thread::sleep(Duration::from_micros(u64::from(nap)));
                                after_nap = true;
                            }
                        }
                    }
                }
                other => return other,
            }
        }
    })
    .await;
    record(slot, started.elapsed());
    out
}
