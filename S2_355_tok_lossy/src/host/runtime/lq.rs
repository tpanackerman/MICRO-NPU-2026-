//! R15 LQ: a deep launch queue. The synchronous path (`launch(..).await`) submits a launch only after the previous one answered, so the device idles for the host turnaround of
//! every launch (completion detection, function lookup, binding, ring write). furiosa-opt 0.9.0 lets a launch be SUBMITTED without waiting for it (`backend::npu::{function,
//! Function::run}`: the first poll of `run` submits; the device runs launches in submission order, 16 in flight per cluster), so the host can stay `depth` launches ahead.
//!
//! Exactness: the same kernels, arguments and order; the device executes them exactly as before. What changes is only WHEN the host touches device memory: every host read of
//! a buffer a queued launch writes, and every host write of a buffer a queued launch reads or writes, must come after those launches answered. Callers either `drain()` (wait
//! for every queued launch) or keep the `mark()` of the last launch that may touch the buffer and check `completed(mark)` / `wait(mark)` before the transfer.
//! A launch submitted through the ordinary `launch(..).await` while launches are queued is submitted behind them and answers after them (one in-order ring).
//!
//! `S2_LQ_DEPTH` = launches in flight (default 8, measured best with 6..12; 0 = off: every `lq_launch!` is the old synchronous launch). Waiting for the front launch spins like the runtime does, except
//! for the long launch kinds (`sleepwait::Slot`) whose learned minimum duration allows a nap (same estimator and margins as `sleepwait::sleepy`).

use furiosa_opt_std::prelude::*;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::time::Instant;

use crate::host::prof;
use crate::host::sleepwait::Slot;

/// Launches kept in flight (0 = off).
pub fn depth() -> usize {
    static DEPTH: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *DEPTH.get_or_init(|| {
        if !cfg!(backend = "npu") {
            return 0;
        }
        std::env::var("S2_LQ_DEPTH").ok().and_then(|v| v.parse().ok()).unwrap_or(8).min(12)
    })
}

/// Whether `lq_launch!` queues (NPU backend and `S2_LQ_DEPTH` > 0).
pub fn on() -> bool {
    depth() > 0
}

type Fut = std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), Error>>>>;

struct Entry {
    seq: u64,
    fut: Fut,
    slot: Option<Slot>,
    /// a time the launch had certainly not started yet (its submission, later the last poll that found the launch before it unanswered)
    start_lb: Instant,
    napped: bool,
}

struct State {
    q: VecDeque<Entry>,
    /// sequence number of the last launch pushed / of the last launch known answered (every launch <= `done` answered: they answer in order)
    pushed: u64,
    done: u64,
    /// the first failure of a queued launch, reported by the next push / wait / drain
    err: Option<Error>,
}

thread_local! {
    static STATE: RefCell<State> = const { RefCell::new(State { q: VecDeque::new(), pushed: 0, done: 0, err: None }) };
}

fn noop_cx() -> std::task::Context<'static> {
    std::task::Context::from_waker(std::task::Waker::noop())
}

/// Polls the front launch once; `true` when it answered (it is removed).
fn poll_front(st: &mut State) -> bool {
    let Some(front) = st.q.front_mut() else { return false };
    let polled = front.fut.as_mut().poll(&mut noop_cx());
    let now = Instant::now();
    match polled {
        std::task::Poll::Pending => {
            // the launch behind it cannot have started before now
            if let Some(next) = st.q.get_mut(1) {
                next.start_lb = next.start_lb.max(now);
            }
            false
        }
        std::task::Poll::Ready(result) => {
            let entry = st.q.pop_front().expect("front");
            if let Some(slot) = entry.slot {
                // an upper bound of the launch's duration (start_lb <= start, now >= end): the estimator keeps a minimum, so a loose sample never lengthens a nap
                crate::host::sleepwait::lq_learn(slot, now.duration_since(entry.start_lb));
            }
            st.done = st.done.max(entry.seq);
            if let Err(error) = result {
                st.err.get_or_insert(error);
            }
            true
        }
    }
}

/// Waits until the front launch answered: one nap for a long launch kind whose minimum duration is known, else spinning polls.
fn wait_front(st: &mut State) {
    let mut after_nap: Option<Slot> = None;
    loop {
        if st.q.is_empty() {
            return;
        }
        let answered = poll_front(st);
        if let Some(slot) = after_nap.take() {
            crate::host::sleepwait::lq_feedback(slot, answered);
        }
        if answered {
            return;
        }
        let front = st.q.front_mut().expect("non-empty");
        if let (false, Some(slot)) = (front.napped, front.slot) {
            front.napped = true;
            if let Some(nap) = crate::host::sleepwait::lq_nap(slot, front.start_lb.elapsed()) {
                std::thread::sleep(nap);
                after_nap = Some(slot);
                continue;
            }
        }
        std::hint::spin_loop();
    }
}

fn take_err(st: &mut State) -> Result<(), Error> {
    match st.err.take() {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// Sequence number of the last launch pushed: a later `completed(mark)` / `wait(mark)` covers it and every launch before it.
pub fn mark() -> u64 {
    STATE.with(|st| st.borrow().pushed)
}

/// Non-blocking: whether every launch up to `mark` answered (polls the queue front without waiting).
pub fn completed(mark: u64) -> bool {
    STATE.with(|st| {
        let mut st = st.borrow_mut();
        while st.done < mark && poll_front(&mut st) {}
        st.done >= mark || st.q.is_empty()
    })
}

/// Blocks until every launch up to `mark` answered.
pub fn wait(mark: u64) -> Result<(), Error> {
    settle(mark);
    STATE.with(|st| take_err(&mut st.borrow_mut()))
}

/// `wait` that leaves a launch failure for the next `wait` / `drain` / push to report (for callers that cannot return it).
pub fn settle(mark: u64) {
    STATE.with(|st| {
        let mut st = st.borrow_mut();
        if st.done >= mark || st.q.is_empty() {
            return;
        }
        let t = Instant::now();
        while st.done < mark && !st.q.is_empty() {
            wait_front(&mut st);
        }
        prof::add("lq_wait", t.elapsed());
    })
}

/// Blocks until every queued launch answered. Required before any host transfer that touches a buffer a queued launch may use.
pub fn drain() -> Result<(), Error> {
    STATE.with(|st| {
        let mut st = st.borrow_mut();
        if st.q.is_empty() {
            return take_err(&mut st);
        }
        let t = Instant::now();
        while !st.q.is_empty() {
            wait_front(&mut st);
        }
        prof::add("lq_drain", t.elapsed());
        take_err(&mut st)
    })
}

/// Submits `F` (registry path `path`, non-generic: key `&[]`) with `args` bound exactly as the `#[device]` macro binds them (the argument tuple without the device), after
/// waiting for room in the queue. Returns once the launch is submitted (the runtime's first poll spins up to 200 us for an early answer).
#[cfg(backend = "npu")]
pub async fn push<A: furiosa_opt_std::runtime::DeviceSend>(device: &Device, path: &'static str, args: &A, slot: Option<Slot>) -> Result<(), Error> {
    let function = furiosa_opt_std::backend::npu::function(device, path, &[]).await?;
    let mut inputs = furiosa_opt_std::runtime::Buffers::new();
    furiosa_opt_std::runtime::DeviceSend::bind(args, &mut inputs)?;
    let outputs = furiosa_opt_std::runtime::Buffers::new();
    let mut fut: Fut = Box::pin(async move { function.run(&inputs, &outputs).await });
    STATE.with(|st| {
        let mut st = st.borrow_mut();
        let depth = depth().max(1);
        if st.q.len() >= depth {
            let t = Instant::now();
            while st.q.len() >= depth {
                wait_front(&mut st);
            }
            prof::add("lq_room", t.elapsed());
        }
        st.pushed += 1;
        let seq = st.pushed;
        let start_lb = Instant::now();
        // the first poll submits the launch
        match fut.as_mut().poll(&mut noop_cx()) {
            std::task::Poll::Ready(result) => {
                // answered within the runtime's spin: only possible when nothing was queued before it
                st.done = st.done.max(seq);
                if let Err(error) = result {
                    st.err.get_or_insert(error);
                }
            }
            std::task::Poll::Pending => st.q.push_back(Entry { seq, fut, slot, start_lb, napped: false }),
        }
        prof::add("lq_push", start_lb.elapsed());
        take_err(&mut st)
    })
}

#[cfg(not(backend = "npu"))]
pub async fn push<A: furiosa_opt_std::runtime::DeviceSend>(_device: &Device, _path: &'static str, _args: &A, _slot: Option<Slot>) -> Result<(), Error> {
    unreachable!("the launch queue is NPU-only (lq::on() is false on the CPU backend)")
}

/// A device launch through the queue (`lq::on()`), else exactly the old synchronous form.
/// - `lq_launch!(device, F, (a, b, ..))` == `launch(F, (&mut *device, a, b, ..)).await`
/// - `lq_launch!(device, F, (a, b, ..), slot)` == `sleepy(slot, launch(..)).await`
/// - `lq_launch!(device, F, (a, b, ..), slot, work)` == `sleepy_with(slot, launch(..), work).await` (`work` runs once, after the submission)
/// The queued form keeps the compile-time argument check of `launch` (an unexecuted `launch(..)` of the same tuple).
macro_rules! lq_launch {
    ($dev:expr, $f:path, ($($a:expr),* $(,)?)) => {{
        if $crate::host::runtime::lq::on() {
            if false {
                let _ = launch($f, (&mut *$dev, $($a),*));
            }
            $crate::host::runtime::lq::push(&*$dev, $f.path(), &($($a,)*), None).await
        } else {
            launch($f, (&mut *$dev, $($a),*)).await
        }
    }};
    ($dev:expr, $f:path, ($($a:expr),* $(,)?), $slot:expr) => {{
        if $crate::host::runtime::lq::on() {
            if false {
                let _ = launch($f, (&mut *$dev, $($a),*));
            }
            $crate::host::runtime::lq::push(&*$dev, $f.path(), &($($a,)*), Some($slot)).await
        } else {
            $crate::host::sleepwait::sleepy($slot, launch($f, (&mut *$dev, $($a),*))).await
        }
    }};
    ($dev:expr, $f:path, ($($a:expr),* $(,)?), $slot:expr, $work:expr) => {{
        if $crate::host::runtime::lq::on() {
            if false {
                let _ = launch($f, (&mut *$dev, $($a),*));
            }
            let pushed = $crate::host::runtime::lq::push(&*$dev, $f.path(), &($($a,)*), Some($slot)).await;
            if pushed.is_ok() {
                ($work)();
            }
            pushed
        } else {
            $crate::host::sleepwait::sleepy_with($slot, launch($f, (&mut *$dev, $($a),*)), $work).await
        }
    }};
}
pub(crate) use lq_launch;
