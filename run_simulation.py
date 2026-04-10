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
nghost    = 2

print(f"[1] 静的 gGA 計算 (U = {U_initial}) を開始...")
ga_obj = ga.GA(U=U_initial, nghost=nghost, nphysorb=nphysorb, n=0.5)
ga_obj.optimize_selfc_new(rinit=None, lambdainit=None, muinit=None)

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
# 4. H_loc_final の構築 (Fock 空間, U_final へのクエンチ)
# ============================================================
# NOTE: compute_derivatives 内で dPhi_dt = -1j * (H_emb @ Phi) を実行するため、
#       H_emb の次元は Phi と同じ dim_Phi × dim_Phi でなければならない。
#       ここでは一体行列 (2B × 2B) を第二量子化で Fock 空間へ変換する:
#         H_fock = Σ_{ij} h_1body[i,j] * c†_i c_j  (半充填セクター)
#
U_final = 4.0
print(f"[4] クエンチ後ハミルトニアン (U = {U_final}) を構築中...")

# 一体行列 h_1body (2B × 2B) を構築
#   物理ブロック (0:B, 0:B): on-site エネルギー -U_final/2
#   浴ブロック  (B:2B, B:2B): 平衡状態の Lambda (浴軌道エネルギー)
h_1body = np.zeros((2 * B, 2 * B), dtype=complex)
np.fill_diagonal(h_1body[:B, :B], -U_final / 2.0)          # 物理ブロック
h_1body[B:, B:] = np.diag(np.diag(ga_obj.Lmbdac))          # 浴ブロック (平衡 Lambda 対角)

# 一体行列を Fock 空間表現に変換
import scipy.sparse as sp
H_loc_fock = sp.csr_matrix((dim_Phi, dim_Phi), dtype=complex)
for i in range(2 * B):
    ann_i = FH_list[i].getH()          # c_i (全Fock空間, 消滅)
    for j in range(2 * B):
        if abs(h_1body[i, j]) < 1e-14:
            continue
        # c†_j c_i の半充填セクター行列
        op_ij = FH_list[j].dot(ann_i)
        H_loc_fock += h_1body[i, j] * op_ij[ioff:iend, ioff:iend]

H_loc_final = H_loc_fock.toarray()  # (dim_Phi, dim_Phi) dense 行列
print(f"  H_loc_final の形状: {H_loc_final.shape}")

# ============================================================
# 5. TD-gGA ダイナミクスの実行
# ============================================================
print("[5] TD-gGA ダイナミクスを実行中...")

physics_params = {
    'dim_Phi'       : dim_Phi,
    'B'             : B,
    'H_loc_final'   : H_loc_final,    # (dim_Phi, dim_Phi) Fock 空間表現
    'D_guess_0'     : D_guess_0,       # (B, B) Frozen-D 初期値
    'op_cb'         : op_cb,           # (B, B, dim_Phi, dim_Phi) c†_al b_a
    'op_bb'         : op_bb,           # (B, B, dim_Phi, dim_Phi) b†_b  b_a
}

t_max = 10.0
dt    = 0.05

sol = td.run_quench_dynamics(Phi_0, n_ab_0, t_max, dt, physics_params)

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
