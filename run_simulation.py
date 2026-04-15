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

FH_list = ed.build_creation_ops()
ioff = sum(int(comb(ed.n_tot_orb, i, exact=True)) for i in range(ed.n_half))
iend = ioff + ed.hsize_half

op_cb = np.zeros((B, 1, dim_Phi, dim_Phi), dtype=complex)
op_bb = np.zeros((B, B, dim_Phi, dim_Phi), dtype=complex)
nphys = ed.n_phys_orb  # = 2 (spin-up=0, spin-dn=1)
c_up_dag = FH_list[0]

# fix_gauge により bath 軌道は逆順に並んでいる:
# Delta の a 番軌道 ↔ FH_list[nphys + 2*(B-1-a)]  (spin-up bath)
#                    ↔ FH_list[nphys + 2*(B-1-a)+1] (spin-dn bath)
for a in range(B):
    bath_up_idx = nphys + 2 * (B - 1 - a)
    b_ann_a = FH_list[bath_up_idx].getH()
    op_cb[a, 0] = c_up_dag.dot(b_ann_a)[ioff:iend, ioff:iend].toarray()
    for b in range(B):
        bath_up_b = nphys + 2 * (B - 1 - b)
        b_dag_b = FH_list[bath_up_b]
        op_bb[a, b] = b_dag_b.dot(b_ann_a)[ioff:iend, ioff:iend].toarray()

print("  op_cb, op_bb の構築完了。")

# ============================================================
# 4. H_emb_0_fock の構築 (Fock 空間, U_final へのクエンチ)
# ============================================================
U_final = 4.0
print(f"[4] クエンチ後ハミルトニアン (U_f = {U_final}) の固定部分を構築中...")

import scipy.sparse as sp
H_loc_fock = sp.csr_matrix((dim_Phi, dim_Phi), dtype=complex)

n_up_phys = FH_list[0].dot(FH_list[0].getH())
n_dn_phys = FH_list[1].dot(FH_list[1].getH())
op_n_phys = (n_up_phys + n_dn_phys)[ioff:iend, ioff:iend]
H_loc_fock += (-U_final / 2.0) * op_n_phys

D_vec = ga_obj.D
for a in range(B):
    bath_up_idx = nphys + 2 * (B - 1 - a)
    bath_dn_idx = nphys + 2 * (B - 1 - a) + 1
    term_up = FH_list[0].dot(FH_list[bath_up_idx].getH()) + FH_list[bath_up_idx].dot(FH_list[0].getH())
    term_dn = FH_list[1].dot(FH_list[bath_dn_idx].getH()) + FH_list[bath_dn_idx].dot(FH_list[1].getH())
    H_loc_fock += D_vec[a, 0] * term_up[ioff:iend, ioff:iend]
    H_loc_fock += D_vec[a, 0] * term_dn[ioff:iend, ioff:iend]

n_up_full = FH_list[0].dot(FH_list[0].getH())
n_dn_full = FH_list[1].dot(FH_list[1].getH())
op_docc = n_up_full.dot(n_dn_full)[ioff:iend, ioff:iend].toarray()
H_loc_fock += U_final * sp.csr_matrix(op_docc)

H_emb_0_fock = H_loc_fock.toarray()
H_phys_1body = np.array([[-U_final / 2.0]], dtype=complex)

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
norms       = np.zeros(nsteps)
occupations = np.zeros((B, nsteps))
docc        = np.zeros(nsteps)   # 二重占有数 d(t) = ⟨Φ|n_↑ n_↓|Φ⟩
Z_t         = np.zeros(nsteps)   # 準粒子重み Z(t) = (R R†)_{00}

I = np.eye(B)
from scipy import linalg as LA

for k in range(nsteps):
    Phi_t, n_ab_t = td.unpack_state(sol.y[:, k], dim_Phi, B)
    norms[k]          = np.real(Phi_t.conj() @ Phi_t)
    occupations[:, k] = np.real(np.diag(n_ab_t))
    docc[k]           = np.real(Phi_t.conj() @ op_docc @ Phi_t)

    # R 行列の計算（固有値クリッピングで数値安定化）
    # op_cb は (B, 1, ...) — スピン上向き1本のみ → fdaggerc は (B, 1)
    Delta    = (n_ab_t + n_ab_t.conj().T) / 2.0
    fdaggerc = np.array([[Phi_t.conj() @ op_cb[a, 0] @ Phi_t] for a in range(B)])
    eigs_D, U_D = LA.eigh(Delta)
    eigs_D = np.clip(np.real(eigs_D), 1e-10, 1.0 - 1e-10)
    inv_sqrt_eigs = 1.0 / np.sqrt(eigs_D * (1.0 - eigs_D))
    R = (U_D * inv_sqrt_eigs[np.newaxis, :]) @ U_D.conj().T @ fdaggerc
    # R は (B, 1) → Z = (R R†)[0,0]
    Z_t[k] = np.real((R @ R.conj().T)[0, 0])

# 数値を npz に保存（replot.py で即座に再プロット可能）
np.savez('td_gga_result.npz',
         t=sol.t, norms=norms, occupations=occupations, docc=docc, Z=Z_t,
         U_initial=U_initial, U_final=U_final)

fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

# --- プロット 1: ノルムの保存 ---
axes[0].plot(sol.t, norms, 'k-', linewidth=1.5)
axes[0].axhline(1.0, color='gray', linestyle='--', linewidth=0.8, label='理想値 = 1')
axes[0].set_ylabel(r'$\langle\Phi(t)|\Phi(t)\rangle$')
axes[0].set_title(f'TD-gGA クエンチダイナミクス  $U_i={U_initial}$ → $U_f={U_final}$')
axes[0].legend()
norm_dev = max(abs(norms - 1.0).max() * 1.5, 1e-4)
axes[0].set_ylim([1.0 - norm_dev, 1.0 + norm_dev])

# --- プロット 2: 準粒子占有数 ---
for orb in range(B):
    axes[1].plot(sol.t, occupations[orb, :], label=f'軌道 {orb}')
axes[1].set_ylabel(r'$n_{aa}(t)$')
axes[1].legend()

# --- プロット 3: 二重占有数 ---
axes[2].plot(sol.t, docc, 'g-', linewidth=1.5)
axes[2].set_ylabel(r'$d(t) = \langle n_\uparrow n_\downarrow \rangle$')
axes[2].set_ylim(bottom=0)

# --- プロット 4: 準粒子重み ---
axes[3].plot(sol.t, Z_t, 'm-', linewidth=1.5)
axes[3].set_ylabel(r'$Z(t)$')
axes[3].set_ylim([0.0, 1.0])

plt.tight_layout()
plt.savefig('quench_dynamics.png', dpi=150)
plt.show()
print("  プロットを quench_dynamics.png に保存しました。")

# ============================================================
# 7. 平衡状態（初期状態）の結果サマリー
# ============================================================
print("\n" + "="*50)
print(f"  平衡状態サマリー (U_i = {U_initial})")
print("="*50)
print(f"  Z        = {ga_obj.Z:.6f}")
print(f"  d (docc) = {docc[0]:.6f}")
print(f"  n_ab (対角) = {[f'{np.real(n_ab_0[i,i]):.4f}' for i in range(B)]}")
print(f"  ||Phi_0|| = {np.linalg.norm(Phi_0):.10f}")
print(f"  dim_Phi   = {dim_Phi},  B = {B}")
print("="*50)
print(f"  t=0 の d(t) = {docc[0]:.6f}  (平衡値と一致するか確認)")
print(f"  t=0 の Z(t) = {Z_t[0]:.6f}  (平衡値 Z={ga_obj.Z:.6f} と比較)")
