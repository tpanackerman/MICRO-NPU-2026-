use furiosa_opt_std::prelude::*;

axes![
    H = 3840,
    L = 15360,
    W = 262144,
    Ns = 8,
    Gs = 2,
    Ds = 256,
    Qs = 4096,
    Ps = 2048,
    Ts = 1024,
    Gf = 16,
    Df = 512,
    Qf = 8192,
    Pf = 512,
    Tf = 512,
    C = 64,
    E = 32768,
    Rv = 6912,
    Mv = 3840,
    Ov = 3840,
    Pv = 1120,
    Aa = 640,
    Dummy2 = 2,
    Dummy8 = 8,
    Dummy256 = 256,
];

// Device argmax (src/device/lmhead): 8 hidden rows, 16-wide dummy, per-slice vocab ids (512) and 8 rows of them (4096).
// Chunk QKV (src/device/qkvc): output groups, K chunks, chunk width, row tile, norm/rope relabels, row-tile counts, T = 8 bucket.
axes![Gg = 16, Kc = 32, Ke = 128, Rt = 64, Hq = 3, Hr = 1280, Iz = 2, Rq4 = 4, Rq2 = 2, Nl = 4, Dh = 2, Dq = 128, Rk = 32, Rq8 = 8, Tb = 8];
// Batched decode attention (src/device/dattn, tree dattn): Bd = decode rows per launch is the qkvc T = 8 bucket axis (alias of Tb), Rn request slots of the
// flat sliding cache [Rf = (Rn + 1) * 1280 rows (ring of 1280 per slot, see rpa_host), Ns, Ds] (slot Rn = scratch), Pn pool pages of the flat full-layer pool
// [Pr = (Pn + 2) * 512 rows, Df] (page Pn = zero page, page Pn + 1 = trash page).
axes![
    Rn = 22, Pn = 110, Np = 5, Rf = 29440, Tx = 1024, Bc = 4, Bd16 = 16, Bc8 = 8, Sk = 32, Dm32 = 32, Dm64 = 64, Pr = 57344,
    Kx2 = 1024, Kx3 = 1536, Kx4 = 2048, Kx5 = 2560, Sf2 = 16, Sf3 = 24, Sf4 = 32, Sf5 = 40, Sg2 = 32, Sg3 = 48, Sg4 = 64, Sg5 = 80,
    Kq1 = 512, Kq6 = 3072, Kq7 = 3584, Kq8 = 4096, Kq10 = 5120, Sf1 = 8, Sf6 = 48, Sf7 = 56, Sf8 = 64, Sf10 = 80, Dc = 8, Dw = 64,
    Tq = 256, Pq = 7168, R8 = 8, Kb2 = 128, Kb5 = 320, Hg = 8, Bk = 64
];
pub use self::Tb as Bd;

axes![Rh = 2];
// Ub = 64-row blocks of the 256-row hybrid pass (device/qkvc/quad.rs, R13 qkv256).
axes![Ub = 4];
axes![Rw8 = 8, Dummy16 = 16, Fl1 = 512, Fl8 = 4096];
// Bn2 = leading ban-variant axis of the argmax keys output `[Bn2, c, v, Rw8]`: ONE contiguous 16 KB block per variant (the order of `host::generate::Ban::ALL`); the kernels compute ALL
// variants for every row and the host decodes them all (see `host::lmhead`).
axes![Bn2 = 2];
// Gb4 = leading block axis of the two-group argmax keys output `[Gb4, c, v, Rw8]`: ONE contiguous 16 KB block per (group of 8 rows, ban variant), block `2 * group + variant` (the same store pattern
// as the verified 8-row kernel; interleaving the groups inside the row axis (`[c, v, Rw16]`, 32 B per group in a 64 B line) lost most lanes of group 1 on HW, t256b job 150254).
axes![Gb4 = 4];

// Chunk attention (src/device/chunk): Tc = tokens per chunk (compile-time bucket), fresh broadcast axes, in-slice key slots
// (Kk = [Seg(2) = ring | chunk] x 32 keys; Kf = 16 per page x pages, Kf64 for 4 pages).
axes![Tc = 64, Dummy4 = 4, Dummy32 = 32, Dummy64 = 64, Seg = 2, Kk = 64, Kf = 32, Seg4 = 4, Kf64 = 64];

// Batched image / audio encoder (src/device/mm): row groups, K chunks, padded chunk widths, row tiles, token tiles.
axes![Mg = 16, Mc = 32, Ev = 224, Ep = 128, Ea = 32];
axes![Rt120 = 120, Rt240 = 240, Rt80 = 80, Rt48 = 48];
axes![Tm16 = 16, Tm32 = 32, Tm64 = 64, Tm128 = 128, Tm256 = 256];
// Chunk axes of the token-per-slice encoder norms (src/device/mm/norm.rs): Rv = 8 x 864, Mv = 8 x 480.
axes![Lq8 = 8, Lr864 = 864, Mq8 = 8, Mr480 = 480];

// Ragged attention (rpa) and the hybrid pass glue (hyb): Pw = prefill units per wave (one per cluster), Cx = chunk-key rows per unit (a run of <= 4 segments),
// P128 / P256 = staged prefill rows of the S = 2 / 4 segment buckets, Pb2 / Pb4 = the same as 8-row blocks of the full layers, Rg8 = 8-row groups of a 64-row unit.
axes![Pw = 2, Cx = 256, P128 = 128, P256 = 256, Pb2 = 16, Pb4 = 32, Rg8 = 8, Rg4 = 4];
// Tpk = bf16 elements of the packed per-pass upload of the hybrid pass (embeddings + every table: ONE host-to-device transfer, the tensors are views of it).
// Worst case (4-page full prefill + 16 decode rows + the MASK_PEN key penalties) = 2916416 bytes (job 148076 showed 2097152 is too small), rounded up to 2924544 = 357 x 8 KB; + 4 x 8 KB for the KVD tables (window index + two cache row index blocks, each at an 8 KB boundary) = 2957312 = 361 x 8 KB.
axes![Tpk = 1478656];
// Tpk256 = the same for the 256-row hybrid pass (runtime/hybrid256.rs): 256 embedding rows + 4 units of rope rows + two waves of masks. 6004736 bytes = 733 x 8 KB (upper bound; the first run prints the bytes used).
axes![Tpk256 = 3026944];
// Kw = key window rows of one prefill unit in the KVD variant (ring window 1024 + chunk keys 128, all read back from the flat ring cache).
axes![Kw = 1152];
// Kw5 = key window rows of one prefill unit in the 256-row KVD variant (ring window 1024 + chunk keys 256 = the WHOLE 1280-row ring of the request slot).
axes![Kw5 = 1280];

// R13 fusedlayer: ring cache as 32-row blocks (Rf = Rb32 * Rw32 = 920 * 32), Rw32 = row inside a block (= key lane), Sb = block slots of a decode window row, Sb40 = padded mask slots.
axes![Rb32 = 920, Rw32 = 32, Sb = 34, Sb40 = 40];

axes![Hq6 = 6, Hr640 = 640];


// R14 d13b (S2_DECTIER): tiered decode attention of the 16 rider rows (rows typed `Bd16 = 8` / `Bd16 = 4`): tiers of 8 rows x Sb = 34 slots, 4 x Sb26, 4 x Sb18;
// Sp32 / Sp24 = mask slots padded to 8; Bc2 = rows per cluster of a 4-row tier.
axes![Bc2 = 2, Sb26 = 26, Sb18 = 18, Sp32 = 32, Sp24 = 24];
// R14 attnre item E: hg4 full-layer prefill layout (4 query-head groups x 64 key lanes; 32 key slots per lane in the 4-page bucket).
axes![Hg4 = 4, Kq32 = 32];
