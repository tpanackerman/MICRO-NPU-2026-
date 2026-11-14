//! R14 D13e (design: stage2/d13e/DESIGN.md): the next-token boundary of the 256-row hybrid pass on the device. Switch `S2_D13E`:
//! - 0 (default): b11 byte for byte;
//! - 1 = P0 "fin2bucket": the rows that finish a prompt are copied on the device into the free rows of the 16-row decode bucket (`ops_d13e::d13e_rowcopy8`), so the riders and the
//!   finishing rows are served by ONE `final_norm_and_argmax_res2` launch with no row D2H and no host-group upload;
//! - 2 = P0 + P1 "device feed": after that LM-head launch the device reduces the keys to the token of each row (`ops_d13e::d13e_reduce`, the exact host rule) and gathers the embedding rows
//!   of the NEXT pass's decode rows into its pre-uploaded packed buffer (`ops_d13e::d13e_feed16`); the host reads 2 x 32 B of tokens and skips the decode-row patch when the next pass is
//!   the predicted one.
//! - 3 = P0 + P1 + P2 "late read": when every pending row is `Generation::blind_ok`, the next pass launches before the tokens are read; they are read during it.
//! The plan of every pass stays b11's (the worker applies the tokens and plans exactly as before; the fed rows are used only on a pre-upload hit of the same pass description).

use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Bn2, Fl1, H, Kc, Ke, Rw8, W};
use crate::ops_d13e::RowTable4k;

/// Rows of the decode bucket the boundary works on.
pub const LANES: usize = 16;
/// Byte pitch of one residual / row-table row (4096 bf16 columns).
pub const PITCH: usize = 8192;

/// `S2_D13E` (default 1 = P0 finishing rows copied on the device; 0 = b11; 2 = P0 + the P1 device feed, lab only; NPU builds only, 0 on the CPU emulator, which cannot alias views).
pub fn level() -> u8 {
    static LEVEL: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(|| {
        let level = std::env::var("S2_D13E").ok().and_then(|v| v.trim().parse::<u8>().ok()).unwrap_or(3).min(3);
        if cfg!(backend = "npu") { level } else { 0 }
    })
}



/// One lane of the transition k -> k+1: lane `j` = row `240 + j` of pass k (its rider, its finishing-row copy, or a padding row).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lane {
    /// Ban variant (index into `Ban::ALL`) of the session whose row sits in this lane in pass k (0 for a padding lane).
    pub variant: u8,
    /// `Some(token)`: the row this lane feeds takes the host-known `token` (Clip override); `None`: the lane's own device token.
    pub host: Option<usize>,
    /// Row of the next pass's decode bucket this lane's gathered embedding row goes to (0..16).
    pub dst: usize,
}

/// The reduce / feed plan of one transition k -> k+1 (16 lanes, a bijection lanes -> next-pass rows).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeedPlan {
    pub lanes: [Lane; LANES],
    /// The request slots of the occupied lanes of pass k, in lane order (the engine feeds only if they equal the `DevRows::slots` the pass actually left on the device).
    pub k_slots: Vec<usize>,
}

/// What the next pass's rows need. `k_lanes[j]` = the session key (request slot) occupying lane j of pass k with its ban variant; `next_rows[b]` = the session key of row b of the next pass's
/// bucket (`None` = padding row); `host_token(key)` = the host-known pending token of a session that is NOT in pass k's lanes. Returns `None` when a row of the next pass needs a token the
/// device does not produce and the host does not know.
pub fn plan_lanes(k_lanes: &[Option<(usize, u8)>; LANES], next_rows: &[Option<usize>; LANES], mut host_token: impl FnMut(usize) -> Option<usize>) -> Option<FeedPlan> {
    let mut lanes = [Lane { variant: 0, host: Some(0), dst: usize::MAX }; LANES];
    let mut row_taken = [false; LANES];
    // device-fed rows: a session that sits in lane j of pass k and in row b of the next pass
    for (j, lane) in k_lanes.iter().enumerate() {
        lanes[j].variant = lane.map_or(0, |(_, v)| v);
        let Some((key, _)) = lane else { continue };
        if let Some(b) = next_rows.iter().position(|r| *r == Some(*key)) {
            lanes[j].host = None;
            lanes[j].dst = b;
            row_taken[b] = true;
        }
    }
    // the padding rows of the next pass take token 0's row (b11 writes a clone of the last rider there; padding-row values never reach a real row)
    let spare_lanes: Vec<usize> = (0..LANES).filter(|&j| lanes[j].dst == usize::MAX).collect();
    let mut spare = spare_lanes.into_iter();
    for b in 0..LANES {
        if row_taken[b] {
            continue;
        }
        let token = match next_rows[b] {
            Some(key) => host_token(key)?,
            None => 0,
        };
        let j = spare.next().expect("16 lanes for 16 rows");
        lanes[j].host = Some(token);
        lanes[j].dst = b;
    }
    let k_slots = k_lanes.iter().flatten().map(|&(key, _)| key).collect();
    Some(FeedPlan { lanes, k_slots })
}

/// Little-endian bytes of `v`.
pub fn i32_le(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

impl FeedPlan {
    /// pen `[Bn2, Rw8]` of group g: 0 at the lane's variant, `i32::MAX` at the other.
    pub fn pen(&self, g: usize) -> Vec<i32> {
        let mut out = vec![0i32; 16];
        for r in 0..8 {
            let v = self.lanes[8 * g + r].variant as usize;
            out[(1 - v) * 8 + r] = i32::MAX;
        }
        out
    }
    pub fn lo(&self, g: usize) -> Vec<i32> {
        (0..8).map(|r| self.lanes[8 * g + r].host.map_or(i32::MIN, |t| (t << 13) as i32)).collect()
    }
    pub fn hi(&self, g: usize) -> Vec<i32> {
        (0..8).map(|r| self.lanes[8 * g + r].host.map_or(i32::MAX, |t| (t << 13) as i32)).collect()
    }
    pub fn dst(&self, g: usize) -> Vec<i32> {
        (0..8).map(|r| (self.lanes[8 * g + r].dst * PITCH) as i32).collect()
    }
    /// Whether group g has a Clip override (else its raw reduce output is the gather index).
    pub fn overridden(&self, g: usize) -> bool {
        (0..8).any(|r| self.lanes[8 * g + r].host.is_some())
    }
}

/// The P0 copy of the finishing rows: sources = the last row of every finishing run (residual row numbers), lanes `d ..` of the bucket. `None` when it does not apply (bucket of 8, no
/// finishing row, or riders + finishing rows > 16).
pub fn copy_rows(finishing_rows: &[usize], d: usize, bd: usize) -> Option<(Vec<i32>, Vec<i32>)> {
    let f = finishing_rows.len();
    if bd != 16 || f == 0 || d + f > LANES || f > 8 {
        return None;
    }
    let mut s: Vec<i32> = finishing_rows.iter().map(|&r| (r * PITCH) as i32).collect();
    let mut t: Vec<i32> = (0..f).map(|i| ((d + i) * PITCH) as i32).collect();
    // unused entries repeat the first pair (the same bytes written twice)
    s.resize(8, s[0]);
    t.resize(8, t[0]);
    Some((s, t))
}

/// Device tensors of P1: the 2 GiB pre-scaled row table, the reduce constants, the no-override Clip bounds and the reduce outputs (two `[Rw8]` pieces 8 KB apart each).
pub struct FeedDev {
    pub table: RowTable4k,
    pub g3: HbmTensor<i32, Chip, m![Fl1]>,
    pub g6: HbmTensor<i32, Chip, m![Fl1]>,
    pub lo_free: HbmTensor<i32, Chip, m![Rw8]>,
    pub hi_free: HbmTensor<i32, Chip, m![Rw8]>,
    /// raw (pre-Clip) tokens of both groups: what the host reads (one 16 KB D2H)
    pub raw: HbmTensor<i32, Chip, m![Bn2, Rw8 # 2048]>,
    /// clipped offsets of the groups that have an override
    pub off: HbmTensor<i32, Chip, m![Bn2, Rw8 # 2048]>,
}

impl FeedDev {
    /// Builds the row table (`bf16(e * EMBED_SCALE)` with the production `scale_row`, 4096 columns, pad 0) and uploads it with the constants. Call AFTER the layer weights.
    pub async fn new(device: &mut Device, embed: &crate::host::load::EmbedHost) -> Result<Self, Error> {
        let started = std::time::Instant::now();
        let width = H::SIZE * 2;
        let mut scaled = vec![0u8; W::SIZE * PITCH];
        let per = W::SIZE.div_ceil(8);
        std::thread::scope(|s| {
            for (c, out) in scaled.chunks_mut(per * PITCH).enumerate() {
                let rows = out.len() / PITCH;
                let first = c * per;
                s.spawn(move || {
                    for r in 0..rows {
                        super::chunk::scale_row(&mut out[r * PITCH..r * PITCH + width], embed.row(first + r));
                    }
                });
            }
        });
        let built = started.elapsed();
        let table: RowTable4k = HostTensor::<bf16, m![W, Kc, Ke]>::from_buf(scaled).to_hbm(&mut device.pdma).await?;
        let g3 = HostTensor::<i32, m![Fl1]>::from_vec((0..512).map(|g| ((511 - g) << 3) as i32).collect::<Vec<i32>>()).to_hbm(&mut device.pdma).await?;
        let g6 = HostTensor::<i32, m![Fl1]>::from_vec((0..512).map(|g| (g << 6) as i32).collect::<Vec<i32>>()).to_hbm(&mut device.pdma).await?;
        let lo_free = HostTensor::<i32, m![Rw8]>::from_vec(vec![i32::MIN; 8]).to_hbm(&mut device.pdma).await?;
        let hi_free = HostTensor::<i32, m![Rw8]>::from_vec(vec![i32::MAX; 8]).to_hbm(&mut device.pdma).await?;
        let raw = HostTensor::<i32, m![Bn2, Rw8 # 2048]>::zero().to_hbm(&mut device.pdma).await?;
        let off = HostTensor::<i32, m![Bn2, Rw8 # 2048]>::zero().to_hbm(&mut device.pdma).await?;
        eprintln!("S2 d13e: row table built in {built:.2?}, uploaded in {:.2?}", started.elapsed() - built);
        Ok(Self { table, g3, g6, lo_free, hi_free, raw, off })
    }

    /// Group g's piece (`[Rw8]`, 8 KB aligned) of a two-group output.
    pub fn piece(t: &HbmTensor<i32, Chip, m![Bn2, Rw8 # 2048]>, g: usize) -> Result<HbmTensor<i32, Chip, m![Rw8]>, Error> {
        crate::host::views::piece_at::<i32, m![Bn2, Rw8 # 2048], i32, m![Rw8]>(t, g * PITCH)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(plan: &FeedPlan, k: &[Option<(usize, u8)>; LANES], next: &[Option<usize>; LANES], host: &dyn Fn(usize) -> Option<usize>) {
        let mut seen = [false; LANES];
        for (j, lane) in plan.lanes.iter().enumerate() {
            assert!(lane.dst < LANES && !seen[lane.dst], "bijection");
            seen[lane.dst] = true;
            match next[lane.dst] {
                Some(key) => match lane.host {
                    None => assert_eq!(k[j].map(|(kk, _)| kk), Some(key), "device lane feeds its own session"),
                    Some(t) => {
                        assert!(!k.iter().flatten().any(|&(kk, _)| kk == key), "a session in pass k is fed from the device");
                        assert_eq!(Some(t), host(key));
                    }
                },
                None => assert!(lane.host.is_some(), "padding rows are host-fed"),
            }
        }
    }

    #[test]
    fn lanes_are_a_bijection() {
        let host = |key: usize| Some(1000 + key);
        // 15 riders + 1 finishing row in pass k; the next pass: the 15 riders minus slot 3 (max_new), the finishing session (slot 40), and slot 50 that skipped pass k
        let mut k = [None; LANES];
        for j in 0..15 {
            k[j] = Some((j, (j % 2) as u8));
        }
        k[15] = Some((40, 1));
        let mut next = [None; LANES];
        let mut b = 0;
        for j in 0..15 {
            if j != 3 {
                next[b] = Some(j);
                b += 1;
            }
        }
        next[b] = Some(40);
        next[b + 1] = Some(50);
        let plan = plan_lanes(&k, &next, host).expect("feedable");
        check(&plan, &k, &next, &host);
        assert_eq!(plan.lanes[3].host, Some(1050));
        // padding rows in both passes
        let mut k2 = [None; LANES];
        k2[0] = Some((7, 0));
        k2[1] = Some((8, 1));
        let mut next2 = [None; LANES];
        next2[0] = Some(8);
        next2[1] = Some(9);
        let plan = plan_lanes(&k2, &next2, host).expect("feedable");
        check(&plan, &k2, &next2, &host);
        // a row whose token nobody knows
        assert!(plan_lanes(&k2, &next2, |_| None).is_none());
    }

    #[test]
    fn tables() {
        let host = |key: usize| Some(key);
        let mut k = [None; LANES];
        k[0] = Some((5, 1));
        let mut next = [None; LANES];
        next[2] = Some(5);
        let plan = plan_lanes(&k, &next, host).unwrap();
        assert_eq!(plan.dst(0)[0], 2 * PITCH as i32);
        let pen = plan.pen(0);
        assert_eq!((pen[0], pen[8]), (i32::MAX, 0));
        assert_eq!((plan.lo(0)[0], plan.hi(0)[0]), (i32::MIN, i32::MAX));
        assert!(plan.overridden(0) && plan.overridden(1));
        assert_eq!(copy_rows(&[63, 127], 14, 16), Some((vec![63 * 8192, 127 * 8192, 63 * 8192, 63 * 8192, 63 * 8192, 63 * 8192, 63 * 8192, 63 * 8192], vec![14 * 8192, 15 * 8192, 14 * 8192, 14 * 8192, 14 * 8192, 14 * 8192, 14 * 8192, 14 * 8192])));
        assert_eq!(copy_rows(&[63], 16, 16), None);
        assert_eq!(copy_rows(&[63], 4, 8), None);
    }
}
