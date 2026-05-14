"""
run_diagnostic.py
エネルギー保存・拘束条件・各種量の包括診断スクリプト

出力:
  diagnostic_U{Ui}_to_{Uf}_T{tmax}.npz   — 数値データ
  diagnostic_U{Ui}_to_{Uf}_T{tmax}.png   — 診断プロット (3×3)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ga_mainfin import GA
import td_gGA_solver_full as td

# ==========================================
# 設定
# ==========================================
U_initial = 1.0
U_final   = 2.5
t_max     = 3.0
dt        = 0.01
N_freq    = 50

# ==========================================
# 静的計算
# ==========================================
print("="*50)
print(f"静的計算 U={U_initial}")
print("="*50)
ga_obj = GA(U=U_initial, nghost=4, nphysorb=2, n=0.5, T=0.003, eks=-99)
calc_nqspo = 3
rinit = np.zeros(calc_nqspo); rinit[0] = 1.0
lambdainit = np.zeros(calc_nqspo * (calc_nqspo + 1) // 2)
ga_obj.optimize_selfc_new(rinit, lambdainit, muinit=0.0)

Z_static   = ga_obj.Z
d_static   = float(np.real(ga_obj.imp_solver.docc))
Eqp_static = ga_obj.Eqp
Etot_static = ga_obj.Etot
print(f"\n静的値: Z={Z_static:.5f}, d={d_static:.5f}")
print(f"         Eqp={Eqp_static:.6f}, Etot={Etot_static:.6f}")

# ==========================================
# TD-gGA クエンチ
# ==========================================
print("\n" + "="*50)
print(f"TD-gGA クエンチ U={U_initial} → {U_final}, t_max={t_max}")
print("="*50)
res = td.run_full_simulation(ga_obj, U_final, t_max, dt, N_freq=N_freq)

t       = res['t']
docc    = res['docc']
E_emb   = res['E_emb']
E_qp    = res['E_qp']
E_lmbda = res['E_lmbda']
E_B     = res['E_B']
E_phys  = res['E_phys']
F2      = res['F2']
TrDelta = res['TrDelta']
Z       = res['Z']
eig_Lc  = res['eig_Lc']
sqrtDtD = res['sqrtDtD']

# ==========================================
# 数値サマリー (コンソール出力)
# ==========================================
def rel_err(arr):
    return (arr - arr[0]) / max(abs(arr[0]), 1e-12)

print("\n" + "="*50)
print("診断サマリー")
print("="*50)
print(f"{'量':<20} {'t=0 値':>12} {'最大相対誤差':>14}")
print("-"*50)
for name, arr in [("E_emb",   E_emb),
                  ("E_qp",    E_qp),
                  ("E_lmbda", E_lmbda),
                  ("E_B",     E_B),
                  ("E_phys",  E_phys)]:
    re = rel_err(arr)
    print(f"{name:<20} {arr[0]:>12.6f} {np.max(np.abs(re)):>14.3e}")
print("-"*50)
print(f"{'F2 (拘束違反)':<20} {'---':>12} {np.max(F2):>14.3e}")
print(f"{'Tr[Δ]':<20} {TrDelta[0]:>12.6f} {np.max(np.abs(TrDelta - TrDelta[0])):>14.3e}")
print(f"{'Z(0)':<20} {Z[0]:>12.6f} {'(静的=' + f'{Z_static:.5f})':>14}")
print(f"{'d(0)':<20} {docc[0]:>12.6f} {'(静的=' + f'{d_static:.5f})':>14}")

# ==========================================
# データ保存
# ==========================================
fname = f"diagnostic_U{U_initial}_to_{U_final}_T{t_max}"
np.savez(fname + ".npz", **res,
         Z_static=Z_static, d_static=d_static,
         U_initial=U_initial, U_final=U_final)
print(f"\nデータ保存: {fname}.npz")

# ==========================================
# 診断プロット (3×3)
# ==========================================
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle(f"TD-gGA 診断  U={U_initial}→{U_final}  (Route B, Λ固定)",
             fontsize=13)

# ---- 行 0: エネルギー成分 ----
ax = axes[0, 0]
ax.plot(t, E_emb,   label=r'$E_\mathrm{emb}=\langle\Phi|H_\mathrm{emb}|\Phi\rangle$', lw=1.5)
ax.plot(t, E_qp,    label=r'$E_\mathrm{qp}=2\mathrm{Tr}[RR^\dagger B]$', lw=1.5)
ax.plot(t, E_lmbda, label=r'$\mathrm{Tr}[\Lambda\Delta]$', lw=1.5, ls='--')
ax.set_title("エネルギー成分")
ax.set_xlabel("t"); ax.legend(fontsize=7); ax.grid(True)

ax = axes[0, 1]
ax.plot(t, E_B,    label=r'$E_B = E_\mathrm{qp}+\mathrm{Tr}[\Lambda\Delta]+E_\mathrm{emb}$',
        lw=2, color='tab:red')
ax.plot(t, E_phys, label=r'$E_\mathrm{phys}=E_\mathrm{qp}+U_f d$',
        lw=2, color='tab:green', ls='--')
ax.set_title("保存量候補")
ax.set_xlabel("t"); ax.legend(fontsize=8); ax.grid(True)

ax = axes[0, 2]
ax.semilogy(t, np.abs(rel_err(E_emb)),  label=r'$E_\mathrm{emb}$', lw=1.5)
ax.semilogy(t, np.abs(rel_err(E_B)),    label=r'$E_B$',    lw=2, color='tab:red')
ax.semilogy(t, np.abs(rel_err(E_phys)), label=r'$E_\mathrm{phys}$', lw=2, color='tab:green',
            ls='--')
ax.set_title("相対誤差 |ΔE/E(0)|")
ax.set_xlabel("t"); ax.legend(fontsize=8); ax.grid(True)

# ---- 行 1: 物理量 ----
ax = axes[1, 0]
ax.plot(t, docc, lw=2, color='tab:green')
ax.axhline(d_static, color='gray', ls='--', label=f'static d={d_static:.4f}')
ax.set_title("二重占有数 d(t)")
ax.set_xlabel("t"); ax.legend(); ax.grid(True)

ax = axes[1, 1]
ax.plot(t, Z, lw=2, color='tab:blue')
ax.axhline(Z_static, color='gray', ls='--', label=f'static Z={Z_static:.4f}')
ax.set_title("準粒子重み Z(t)")
ax.set_xlabel("t"); ax.legend(); ax.grid(True)

ax = axes[1, 2]
ax.plot(t, sqrtDtD, lw=2, color='tab:purple')
ax.set_title(r"$\sqrt{D^\dagger D}(t)$")
ax.set_xlabel("t"); ax.grid(True)

# ---- 行 2: 拘束条件・診断 ----
ax = axes[2, 0]
ax.semilogy(t, F2 + 1e-16, lw=2, color='tab:orange')
ax.set_title(r"拘束条件違反 $\|\langle f^\dagger f\rangle_\Phi - \Delta_n\|$ (F2)")
ax.set_xlabel("t"); ax.grid(True)

ax = axes[2, 1]
ax.plot(t, TrDelta, lw=2, color='tab:brown')
ax.axhline(1.0, color='gray', ls='--', label='half-filling = 1')
ax.set_title(r"$\mathrm{Tr}[\Delta](t)$（粒子数）")
ax.set_xlabel("t"); ax.legend(); ax.grid(True)

ax = axes[2, 2]
for i in range(eig_Lc.shape[1]):
    ax.plot(t, eig_Lc[:, i], lw=1.2, label=f'λ{i+1}')
ax.set_title(r"eig $\Lambda^c(t)$")
ax.set_xlabel("t"); ax.legend(fontsize=8); ax.grid(True)

plt.tight_layout()
plt.savefig(fname + ".png", dpi=150)
print(f"プロット保存: {fname}.png")
