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
dim_Phi = ed.hsize_half                                 # 多体 Hilbert 空間次元

# fix_gauge の変換行列を取得し、ゲージ固定基底で全量を統一する。
# Phi_0 は H_emb(D_gf, Lmbdac_gf) の固有状態なので FH_list の順序 = ゲージ基底。
# ga_obj.Delta は元の基底にあるため U_trans で回転させる。
D_gf, _, phasemat_fix, permmat_fix, transmat_fix = \
    ga_obj.fix_gauge(ga_obj.D.copy(), ga_obj.Lmbdac.copy(), lfor_D=True, lreturn_mats=True)
# U_trans: 元の基底 → ゲージ固定基底 (= T_back^T)
U_trans = permmat_fix @ phasemat_fix @ transmat_fix.T  # (B, B) 実数直交行列
Delta_orig = ga_obj.Delta[:B, :B].copy()
n_ab_0 = U_trans @ Delta_orig @ U_trans.T              # ゲージ固定基底の密度行列

print(f"  B = {B}, dim_Phi = {dim_Phi}")
print(f"  ||Phi_0|| = {np.linalg.norm(Phi_0):.6f}  (≈ 1 が正常)")
print(f"  U_trans:\n{np.round(U_trans, 4)}")
print(f"  D_gf: {D_gf.flatten().round(6)}")

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

# ゲージ固定基底で統一: FH_list[nphys + 2*a] が a 番目の bath (spin-up) に対応
for a in range(B):
    bath_up_idx = nphys + 2 * a
    b_ann_a = FH_list[bath_up_idx].getH()
    op_cb[a, 0] = c_up_dag.dot(b_ann_a)[ioff:iend, ioff:iend].toarray()
    for b in range(B):
        bath_up_b = nphys + 2 * b
        b_dag_b = FH_list[bath_up_b]
        op_bb[a, b] = b_dag_b.dot(b_ann_a)[ioff:iend, ioff:iend].toarray()

print("  op_cb, op_bb の構築完了。")

# ============================================================
# 4. H_emb_0_fock の構築 (Fock 空間, U_final へのクエンチ)
# ============================================================
U_final = 2.5
print(f"[4] クエンチ後ハミルトニアン (U_f = {U_final}) の固定部分を構築中...")

import scipy.sparse as sp
H_loc_fock = sp.csr_matrix((dim_Phi, dim_Phi), dtype=complex)

n_up_phys = FH_list[0].dot(FH_list[0].getH())
n_dn_phys = FH_list[1].dot(FH_list[1].getH())
op_n_phys = (n_up_phys + n_dn_phys)[ioff:iend, ioff:iend]
H_loc_fock += (-U_final / 2.0) * op_n_phys

for a in range(B):
    bath_up_idx = nphys + 2 * a       # ゲージ固定基底: forward indexing
    bath_dn_idx = nphys + 2 * a + 1
    term_up = FH_list[0].dot(FH_list[bath_up_idx].getH()) + FH_list[bath_up_idx].dot(FH_list[0].getH())
    term_dn = FH_list[1].dot(FH_list[bath_dn_idx].getH()) + FH_list[bath_dn_idx].dot(FH_list[1].getH())
    H_loc_fock += D_gf[a, 0] * term_up[ioff:iend, ioff:iend]
    H_loc_fock += D_gf[a, 0] * term_dn[ioff:iend, ioff:iend]

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
dt    = 0.005

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
Z_bare_t    = np.zeros(nsteps)   # 裸の重み Z_bare(t) = |R[0,0]|²
Z_dressed_t = np.zeros(nsteps)   # 着衣の重み Z_dressed(t): Frozen-Λ 多バンド公式
E_emb_t     = np.zeros(nsteps)   # 埋め込みハミルトニアン期待値 ⟨Φ|H_emb_0|Φ⟩
E_tot_t     = np.zeros(nsteps)   # TDVP 保存エネルギー ⟨Φ|H_emb_0|Φ⟩ - Tr(n_ab R H_phys R†)

I = np.eye(B)
from scipy import linalg as LA

for k in range(nsteps):
    Phi_t, _ = td.unpack_state(sol.y[:, k], dim_Phi, B)
    norms[k] = np.real(Phi_t.conj() @ Phi_t)
    docc[k]  = np.real(Phi_t.conj() @ op_docc @ Phi_t)

    # n_ab を Phi から直接計算（拘束条件を厳密に満たす）
    n_ab_t = np.array([[Phi_t.conj() @ op_bb[a, b] @ Phi_t
                        for b in range(B)] for a in range(B)])
    occupations[:, k] = np.real(np.diag(n_ab_t))

    # --- Z(t) と E_tot の計算（ゲージ固定基底のまま計算）---
    # ODE もゲージ固定基底で動くため、U_trans 回転は不要
    fdaggerc_gf = np.array([[Phi_t.conj() @ op_cb[a, 0] @ Phi_t] for a in range(B)])

    Delta_gf_t = (n_ab_t + n_ab_t.conj().T) / 2.0
    eigs_D, U_D = LA.eigh(Delta_gf_t)
    eigs_D = np.clip(np.real(eigs_D), 1e-4, 1.0 - 1e-4)
    inv_sqrt_eigs = 1.0 / np.sqrt(eigs_D * (1.0 - eigs_D))
    R_gf = (U_D * inv_sqrt_eigs[np.newaxis, :]) @ U_D.conj().T @ fdaggerc_gf

    # Z_bare: R_gf はシミュレーション基底（fix_gauge(D,Λ^c) ゲージ）で計算
    Z_bare_t[k]    = np.real((R_gf @ R_gf.conj().T)[0, 0])
    # Z_dressed: U_trans.T で fix_gauge(R,Λ_new) ゲージに変換してから多バンド公式
    R_dressed      = U_trans.T @ R_gf
    Z_dressed_t[k] = ga_obj.calc_Z(ga_obj.Lmbda_new, R_dressed)
    E_emb_t[k] = np.real(Phi_t.conj() @ H_emb_0_fock @ Phi_t)

    # 全エネルギー E_tot の計算 (Eq. 79 に基づく修正)
    # 1. 世界Bの運動エネルギー E_kin = Tr(n_ab * h_qp^0)
    h_qp_0 = R_gf @ H_phys_1body @ R_gf.conj().T
    E_kin = np.real(np.trace(n_ab_t @ h_qp_0.T))
    # 2. 世界Aの純粋な局所エネルギー: H_emb の期待値から混成項を引く
    # E_hyb = 2 Re Tr(D_gf^T fdaggerc_gf)  (Eq. 82)
    E_hyb = 4.0 * np.real(np.trace(D_gf.T @ fdaggerc_gf))
    E_loc = E_emb_t[k] - E_hyb
    # 3. 全エネルギー (保存量)
    E_tot_t[k] = E_kin + E_loc

# 数値を npz に保存（replot.py で即座に再プロット可能）
np.savez('td_gga_result.npz',
         t=sol.t, norms=norms, occupations=occupations, docc=docc,
         Z_bare=Z_bare_t, Z_dressed=Z_dressed_t,
         E_emb=E_emb_t, E_tot=E_tot_t, U_initial=U_initial, U_final=U_final)

# ============================================================
# 7. 平衡状態（初期状態）の結果サマリー（ターミナル出力）
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
print(f"  t=0 の Z_bare    = {Z_bare_t[0]:.6f}  (裸の重み |R[0,0]|²)")
print(f"  t=0 の Z_dressed = {Z_dressed_t[0]:.6f}  (着衣の重み, 平衡値 Z={ga_obj.Z:.6f} と比較)")

# ============================================================
# 8. プロット（2×2 レイアウト）
# ============================================================
import os, glob

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
ax_norm = axes[0, 0]
ax_occ  = axes[0, 1]
ax_docc = axes[1, 0]
ax_Z    = axes[1, 1]

# タイトルに平衡状態の物理量を記載
eq_text = (f"平衡状態 ($U_i={U_initial}$):  "
           f"$Z_0={ga_obj.Z:.4f}$,  "
           f"$d_0={docc[0]:.4f}$,  "
           f"$n_{{ab}}$={[f'{np.real(n_ab_0[i,i]):.3f}' for i in range(B)]}")
fig.suptitle(
    f'TD-gGA クエンチダイナミクス  $U_i={U_initial}$ → $U_f={U_final}$\n{eq_text}',
    fontsize=11)

# --- プロット 1: TDVP 保存エネルギー ---
ax_norm.plot(sol.t, E_tot_t, color='black', linewidth=1.5)
ax_norm.axhline(E_tot_t[0], color='gray', linestyle='--', linewidth=0.8,
                label=f'初期値 $E_0={E_tot_t[0]:.4f}$')
ax_norm.set_ylabel(r'$E_{\rm tot}=E_{\rm kin}+E_{\rm loc}$  (Eq. 79)')
ax_norm.set_xlabel(r'時刻 $t$')
ax_norm.legend()
E_margin = max(np.ptp(E_tot_t) * 0.15, 0.005)
ax_norm.set_ylim([np.min(E_tot_t) - E_margin, np.max(E_tot_t) + E_margin])

# --- プロット 2: 準粒子占有数 ---
for orb in range(B):
    ax_occ.plot(sol.t, occupations[orb, :], label=f'軌道 {orb}')
ax_occ.set_ylabel(r'$n_{aa}(t)$')
ax_occ.set_xlabel(r'時刻 $t$')
ax_occ.legend()

# --- プロット 3: 二重占有数 ---
ax_docc.plot(sol.t, docc, 'g-', linewidth=1.5)
ax_docc.axhline(docc[0], color='gray', linestyle='--', linewidth=0.8,
                label=f'初期値 $d_0={docc[0]:.4f}$')
ax_docc.set_ylabel(r'$d(t) = \langle n_\uparrow n_\downarrow \rangle$')
ax_docc.set_xlabel(r'時刻 $t$')
ax_docc.set_ylim(bottom=0)
ax_docc.legend()

# --- プロット 4: 準粒子重み（裸 vs 着衣）---
ax_Z.plot(sol.t, np.clip(Z_dressed_t, 0, 1), 'm-', linewidth=1.5,
          label=rf'$Z_{{\rm dressed}}$ (Frozen-$\Lambda$), $Z_0={Z_dressed_t[0]:.4f}$')
ax_Z.plot(sol.t, np.clip(Z_bare_t, 0, 1), 'b--', linewidth=1.2,
          label=rf'$Z_{{\rm bare}}=|R_{{00}}|^2$, $Z_0={Z_bare_t[0]:.4f}$')
ax_Z.axhline(ga_obj.Z, color='gray', linestyle=':', linewidth=0.8,
             label=f'静的 $Z={ga_obj.Z:.4f}$')
ax_Z.set_ylabel(r'$Z(t)$')
ax_Z.set_xlabel(r'時刻 $t$')
ax_Z.set_ylim([0.0, 1.0])
ax_Z.legend(fontsize=8)

plt.tight_layout()

# 連番でファイル名を自動生成（quench_1.png, quench_2.png, ...）
existing = glob.glob('quench_*.png')
nums = []
for f in existing:
    base = os.path.splitext(os.path.basename(f))[0]
    try:
        nums.append(int(base.split('_')[1]))
    except (IndexError, ValueError):
        pass
next_num = max(nums, default=0) + 1
save_name = f'quench_{next_num}.png'

plt.savefig(save_name, dpi=150)
plt.show()
print(f"  プロットを {save_name} に保存しました。")
