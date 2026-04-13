"""
TD-gGA クエンチダイナミクス メインスクリプト

ga_mainfin.py (静的 gGA 計算) で平衡状態を求め、
td_gga_solver.py (時間発展) で U クエンチ後のダイナミクスを計算する。
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

import ga_mainfin as ga
import td_gga_solver as td

# ============================================================
# 1. 平衡状態の計算 (U_initial)
# ============================================================
U_initial = 2.0
nphysorb  = 2
nghost    = 4   # ga_mainfin.py スタンドアロンと同じ設定

print(f"[1] 静的 gGA 計算 (U = {U_initial}) を開始...")
ga_obj = ga.GA(U=U_initial, nghost=nghost, nphysorb=nphysorb, n=0.5)

# 初期値: 金属相 (metallic guess)
nqspo = (nphysorb + nghost) // 2
rinit_0      = np.zeros(nqspo); rinit_0[0] = 1.0
lambdainit_0 = np.zeros(nqspo * (nqspo + 1) // 2)
muinit_0     = 0.0

ga_obj.optimize_selfc_new(rinit=rinit_0, lambdainit=lambdainit_0, muinit=muinit_0)

if not ga_obj.lconv:
    print("  警告: 静的計算が収束していません。結果を確認してください。")
else:
    print("  収束完了。")

# ============================================================
# 2. 初期状態の抽出
# ============================================================
B = ga_obj.nqspo              # quasi-spatial orbital 数 (= (nphysorb + nghost) // 2)
ed = ga_obj.imp_solver        # edSolver インスタンス

Phi_0   = ed.eig_vec.copy()                            # 多体埋め込み基底状態 (hsize_half,)
n_ab_0  = ga_obj.Delta[:B, :B].copy()                  # 局所準粒子密度行列 (B, B)
dim_Phi = ed.hsize_half                                 # 多体 Hilbert 空間次元

# D: ga_obj.D は (B, 1) のため、(B, B) にゼロパディングして Frozen-D 初期値とする
D_guess_0 = np.zeros((B, B), dtype=complex)
D_guess_0[:, :ga_obj.D.shape[1]] = ga_obj.D            # 最初の列だけに hybridization を格納

print(f"  B = {B}, dim_Phi = {dim_Phi}")
print(f"  ||Phi_0|| = {np.linalg.norm(Phi_0):.6f}  (≈ 1 が正常)")

# ============================================================
# 3. フォック空間演算子 op_cb, op_bb の構築
# ============================================================
print("[3] フォック空間演算子を構築中...")

FH_list = ed.build_creation_ops()  # 全軌道の生成演算子リスト (全Fock空間)

# 半充填セクターのオフセット計算 (ed_solver の build_op_prods と同様)
ioff = sum(int(comb(ed.n_tot_orb, i, exact=True)) for i in range(ed.n_half))
iend = ioff + ed.hsize_half

# 軌道インデックスの規約:
#   物理軌道 alpha = 0 .. B-1  → FH_list[alpha]
#   浴/ゴースト軌道 a = B .. 2B-1 → FH_list[B + a_loop]
# （FH_list の先頭 B 個が物理、次の B 個が浴）

op_cb = np.zeros((B, B, dim_Phi, dim_Phi), dtype=complex)  # c†_al b_a
op_bb = np.zeros((B, B, dim_Phi, dim_Phi), dtype=complex)  # b†_b  b_a

for a_loop in range(B):      # 浴軌道ループインデックス (0 .. B-1)
    # 浴軌道の消滅演算子: b_a = (b†_a)† = FH_list[B + a_loop].conj().T
    b_ann_a = FH_list[B + a_loop].getH()   # 全Fock空間での消滅演算子 (sparse)

    for al in range(B):      # 物理軌道インデックス alpha (0 .. B-1)
        # op_cb[a_loop, al] = c†_al b_a
        op_full = FH_list[al].dot(b_ann_a)
        op_cb[a_loop, al] = op_full[ioff:iend, ioff:iend].toarray()

    for b_loop in range(B):  # 浴軌道インデックス b (0 .. B-1)
        # op_bb[a_loop, b_loop] = b†_b b_a
        op_full = FH_list[B + b_loop].dot(b_ann_a)
        op_bb[a_loop, b_loop] = op_full[ioff:iend, ioff:iend].toarray()

print("  op_cb, op_bb の構築完了。")

# ============================================================
# 4. H_emb_0_fock の構築 (Fock 空間, U_final へのクエンチ)
# ============================================================
U_final = 4.0
print(f"[4] クエンチ後ハミルトニアン (U_f = {U_final}) の固定部分を構築中...")

# 1体行列 h_1body (2B × 2B) を構築
h_1body = np.zeros((2 * B, 2 * B), dtype=complex)

# (1) 物理ブロック (0:B, 0:B): on-site エネルギー -U_final/2
np.fill_diagonal(h_1body[:B, :B], -U_final / 2.0)

# (2) 浴ブロック (B:2B, B:2B):
# ※注意: Λ^c はここに入れない（ソルバー内で動的に足すため 0 のまま）

# (3) D ブロック (物理-浴の混成): Frozen-D 近似のため初期値のまま固定
h_1body[:B, B:] = D_guess_0
h_1body[B:, :B] = D_guess_0.conj().T

# 一体行列を Fock 空間表現に変換
# H_fock = Σ_{ij} h_1body[i,j] * c†_j c_i  (半充填セクター)
import scipy.sparse as sp
H_loc_fock = sp.csr_matrix((dim_Phi, dim_Phi), dtype=complex)
for i in range(2 * B):
    ann_i = FH_list[i].getH()
    for j in range(2 * B):
        if abs(h_1body[i, j]) < 1e-14:
            continue
        op_ij = FH_list[j].dot(ann_i)
        H_loc_fock += h_1body[i, j] * op_ij[ioff:iend, ioff:iend]

# (4) 2体相互作用 (Hubbard U) の追加
# U2loc は build_Hemb 時に U_initial スケールで保存済み → U_final にスケール変換
# ファイル名: "U2loc-imp{nr}{type}.npz" (ed_solver.py の save_npz 命名規則に従う)
U2loc_sparse = sp.load_npz(f"U2loc-imp{ed.impurity_nr}{ed.impurity_type}.npz")
H_loc_fock += (U_final / U_initial) * U2loc_sparse   # 既に hsize_half × hsize_half

H_emb_0_fock = H_loc_fock.toarray()   # (dim_Phi, dim_Phi) dense 行列
H_phys_1body = h_1body[:B, :B].copy() # (B, B) 物理ブロック（h_qp の 0 次項用）

print(f"  H_emb_0_fock の形状: {H_emb_0_fock.shape}")

# ============================================================
# 5. TD-gGA ダイナミクスの実行
# ============================================================
print("[5] TD-gGA ダイナミクスを実行中...")

physics_params = {
    'dim_Phi'       : dim_Phi,
    'B'             : B,
    'H_emb_0_fock'  : H_emb_0_fock,   # Λ^c 以外を含む多体ハミルトニアン (dim_Phi, dim_Phi)
    'H_phys_1body'  : H_phys_1body,   # h_qp の 0 次項用 1 体物理ブロック (B, B)
    'op_cb'         : op_cb,           # (B, B, dim_Phi, dim_Phi) c†_al b_a
    'op_bb'         : op_bb,           # (B, B, dim_Phi, dim_Phi) b†_b  b_a
}

t_max = 10.0
dt    = 0.05

sol = td.run_quench_dynamics(Phi_0, n_ab_0, t_max, dt, physics_params, rtol=1e-8, atol=1e-10)

if sol.success:
    print(f"  ODE 求解成功。ステップ数: {len(sol.t)}")
else:
    print(f"  警告: ODE 求解に問題が発生しました。メッセージ: {sol.message}")

# ============================================================
# 6. 物理量の計算とプロット
# ============================================================
print("[6] 物理量を計算中...")

nsteps = len(sol.t)
norms       = np.zeros(nsteps)   # ||Phi(t)||²  (保存則の確認)
occupations = np.zeros((B, nsteps))  # 準粒子占有数 n_aa(t) の対角成分

for k in range(nsteps):
    Phi_t, n_ab_t = td.unpack_state(sol.y[:, k], dim_Phi, B)
    norms[k]         = np.real(Phi_t.conj() @ Phi_t)
    occupations[:, k] = np.real(np.diag(n_ab_t))

fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# --- プロット 1: ノルムの保存 (数値安定性の確認) ---
axes[0].plot(sol.t, norms, 'k-', linewidth=1.5)
axes[0].axhline(1.0, color='gray', linestyle='--', linewidth=0.8, label='理想値 = 1')
axes[0].set_ylabel(r'$\langle\Phi(t)|\Phi(t)\rangle$')
axes[0].set_title(f'TD-gGA クエンチダイナミクス  '
                  f'$U_i = {U_initial}$  →  $U_f = {U_final}$')
axes[0].legend()
axes[0].set_ylim([0.8, 1.2])

# --- プロット 2: 準粒子占有数の時間変化 ---
for orb in range(B):
    axes[1].plot(sol.t, occupations[orb, :], label=f'軌道 {orb}')
axes[1].set_xlabel('時刻 $t$')
axes[1].set_ylabel(r'$n_{aa}(t)$')
axes[1].legend()

plt.tight_layout()
plt.savefig('quench_dynamics.png', dpi=150)
plt.show()
print("  プロットを quench_dynamics.png に保存しました。")
