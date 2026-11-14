//! Where the runtime's kernel submission poller runs, inside a CPU-quota container.
//!
//! The runtime (`furiosa-opt-rt`) hands every host-to-device transfer to an io_uring whose submission queue is polled by a kernel thread (`iou-sqp-*`, SQPOLL). That poller keeps spinning
//! for 100 ms after each submission, and a transfer happens every pass, so it never sleeps: it burns about one full core doing nothing between transfers. In an Arena container the
//! cgroup quota is 2 CPUs (`cpu.max` = 200000 100000); poller (0.95 core) + launch thread (0.9 core, spinning on launches) + DMA thread (0.06) exceed it, and the container is throttled
//! in about two thirds of its 100 ms periods (hostdiet R5/R8: `nr_throttled / nr_periods`, per-thread census).
//!
//! The launch thread pins itself to the CPU it runs on and the poller is pinned to the same CPU at nice 19: the poller then only runs while the launch thread does not (it naps in
//! `sleepwait`, it parks while it awaits a transfer), which is exactly when it has work, and the container needs about 1.3 cores instead of 1.95. The DMA thread (it spins only while a
//! transfer is in flight) is moved to the other CPUs. Nothing of the model, the requests or the launches is touched; this only places our own threads and the runtime's poller.
//! It is applied only when the cgroup quota is at most 2.5 CPUs (`cpu.max`): elsewhere (no quota, a wide quota) nothing changes. Every system call that fails is skipped and the
//! ones before it are undone where that matters. `S2_CPUPLACE=0` disables it.

use std::os::raw::c_int;
use std::sync::OnceLock;

unsafe extern "C" {
    fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u8) -> c_int;
    fn sched_getaffinity(pid: i32, cpusetsize: usize, mask: *mut u8) -> c_int;
    fn sched_getcpu() -> c_int;
    fn setpriority(which: c_int, who: u32, prio: c_int) -> c_int;
    fn getpriority(which: c_int, who: u32) -> c_int;
}

const MASK_BYTES: usize = 128;
/// Placement only inside a container whose CPU quota is at most this many CPUs.
const QUOTA_CPUS_MAX: f64 = 2.5;

/// The CPUs other than the launch thread's, once the launch thread is pinned (unset = nothing was placed).
static OTHERS: OnceLock<Vec<usize>> = OnceLock::new();
static PARALLELISM: OnceLock<usize> = OnceLock::new();

fn mask_of(cpus: &[usize]) -> [u8; MASK_BYTES] {
    let mut mask = [0u8; MASK_BYTES];
    for &cpu in cpus {
        if cpu < MASK_BYTES * 8 {
            mask[cpu / 8] |= 1 << (cpu % 8);
        }
    }
    mask
}

fn affinity(tid: i32) -> Vec<usize> {
    let mut mask = [0u8; MASK_BYTES];
    // SAFETY: the kernel writes at most MASK_BYTES bytes into the buffer we own.
    if unsafe { sched_getaffinity(tid, MASK_BYTES, mask.as_mut_ptr()) } < 0 {
        return Vec::new();
    }
    (0..MASK_BYTES * 8).filter(|cpu| mask[cpu / 8] & (1 << (cpu % 8)) != 0).collect()
}

fn pin(tid: i32, cpus: &[usize]) -> bool {
    let mask = mask_of(cpus);
    // SAFETY: the mask buffer lives for the call.
    unsafe { sched_setaffinity(tid, MASK_BYTES, mask.as_ptr()) >= 0 }
}

/// The container's CPU quota in CPUs (`cpu.max`: "quota period"), `None` when unlimited or unreadable.
fn quota_cpus() -> Option<f64> {
    let text = std::fs::read_to_string("/sys/fs/cgroup/cpu.max").ok()?;
    let mut parts = text.split_whitespace();
    let quota: f64 = parts.next()?.parse().ok()?;
    let period: f64 = parts.next()?.parse().ok()?;
    (quota > 0.0 && period > 0.0).then(|| quota / period)
}

fn my_tid() -> Option<i32> {
    std::fs::read_link("/proc/thread-self").ok()?.file_name()?.to_str()?.parse().ok()
}

fn enabled() -> bool {
    cfg!(backend = "npu") && std::env::var("S2_CPUPLACE").map(|value| value != "0").unwrap_or(true)
}

/// `available_parallelism` as it was before the launch thread was pinned (a pinned thread would report 1).
pub fn parallelism() -> usize {
    *PARALLELISM.get_or_init(|| std::thread::available_parallelism().map(|value| value.get()).unwrap_or(1))
}

/// Called once by the worker thread after the model is loaded (the runtime's poller exists since the device was opened).
pub fn place() {
    parallelism();
    if !enabled() || !quota_cpus().is_some_and(|cpus| cpus <= QUOTA_CPUS_MAX) {
        return;
    }
    let Some(me) = my_tid() else { return };
    let (mut poller, mut dma) = (None, None);
    if let Ok(dir) = std::fs::read_dir("/proc/self/task") {
        for entry in dir.flatten() {
            let Some(tid) = entry.file_name().to_str().and_then(|name| name.parse::<i32>().ok()) else { continue };
            let comm = std::fs::read_to_string(entry.path().join("comm")).unwrap_or_default();
            if comm.starts_with("iou-sqp") {
                poller = Some(tid);
            } else if comm.starts_with("gemma4-worker") && tid != me {
                dma = dma.max(Some(tid));
            }
        }
    }
    let Some(poller) = poller else { return };
    // SAFETY: plain system call.
    let x = unsafe { sched_getcpu() };
    let allowed = affinity(0);
    let others: Vec<usize> = allowed.iter().copied().filter(|&cpu| x >= 0 && cpu != x as usize).collect();
    if x < 0 || others.is_empty() || !allowed.contains(&(x as usize)) {
        return;
    }
    let x = x as usize;
    // SAFETY: plain system calls (lowering the priority of one of our own threads is always permitted; getpriority returns the nice value, and -1 may also mean an error: a negative value is never restored).
    let nice = unsafe { getpriority(0, poller as u32) };
    if unsafe { setpriority(0, poller as u32, 19) } < 0 {
        return;
    }
    let undo_nice = || {
        if nice >= 0 {
            // SAFETY: plain system call.
            unsafe { setpriority(0, poller as u32, nice) };
        }
    };
    if !pin(poller, &[x]) {
        undo_nice();
        return;
    }
    if !pin(me, &[x]) {
        pin(poller, &allowed);
        undo_nice();
        return;
    }
    if let Some(dma) = dma {
        pin(dma, &others);
    }
    let _ = OTHERS.set(others.clone());
    eprintln!("S2 cpuplace: launch thread and poller on CPU {x} (poller nice 19), DMA thread on {} other CPUs", others.len());
}

/// First call of a helper thread spawned by the launch thread (they inherit its CPU): leave the launch thread's CPU.
pub fn leave_launch_cpu() {
    if let Some(cpus) = OTHERS.get() {
        pin(0, cpus);
    }
}
