"""
test_noquench_routeA.py
Route A ノークエンチテスト: U_i = U_f で完全エネルギー保存を確認する
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ga_mainfin_routeA import GA
import td_gGA_solver_routeA as td

U_val  = 0.05
t_max  = 3.0
dt     = 0.01
N_freq = 50

print("=" * 50)
print(f"静的計算 U={U_val}  (Route A, Λ=0 ゲージ)")
print("=" * 50)
ga_obj = GA(U=U_val, nghost=4, nphysorb=2, n=0.5, T=0.003, eks=-99)
calc_nqspo = 3
rinit = np.zeros(calc_nqspo); rinit[0] = 1.0
ga_obj.optimize_selfc_routeA(rinit=rinit, muinit=0.0)

Z_static = ga_obj.Z
d_static = float(np.real(ga_obj.imp_solver.docc))
print(f"  Z={Z_static:.6f}  d={d_static:.6f}")

print("\n" + "=" * 50)
print(f"Route A ノークエンチ (U={U_val} → {U_val})")
print("=" * 50)
res = td.run_full_simulation(ga_obj, U_val, t_max, dt, N_freq=N_freq)

t      = res['t']
docc   = res['docc']
E_emb  = res['E_emb']
E_phys = res['E_phys']
F2     = res['F2']
Z      = res['Z']

def rel_err(arr):
    return (arr - arr[0]) / max(abs(arr[0]), 1e-12)

print("\n--- 診断サマリー ---")
print(f"  Z[0]          = {Z[0]:.6f}  (静的: {Z_static:.6f})")
print(f"  d[0]          = {docc[0]:.6f}  (静的: {d_static:.6f})")
print(f"  E_emb  最大相対誤差 = {np.max(np.abs(rel_err(E_emb))):.2e}")
print(f"  E_phys 最大相対誤差 = {np.max(np.abs(rel_err(E_phys))):.2e}")
print(f"  F2 最大値           = {np.max(F2):.2e}  ← 0 に近いほど拘束条件OK")

# プロット
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle(f"Route A ノークエンチ  U={U_val}→{U_val}", fontsize=12)

ax = axes[0]
ax.semilogy(t, np.abs(rel_err(E_emb))  + 1e-16, label=r'$E_\mathrm{emb}$', lw=1.8)
ax.semilogy(t, np.abs(rel_err(E_phys)) + 1e-16, label=r'$E_\mathrm{phys}$', lw=1.8, ls='--')
ax.set_title("エネルギー相対誤差"); ax.set_xlabel("t")
ax.legend(); ax.grid(True)

ax = axes[1]
ax.semilogy(t, F2 + 1e-16, lw=1.8, color='tab:orange')
ax.set_title(r"拘束条件違反 F2 $=\|\langle f^\dagger f\rangle_\Phi - \Delta_n\|$")
ax.set_xlabel("t"); ax.grid(True)

ax = axes[2]
ax.plot(t, Z, lw=1.8, color='tab:blue', label='Z(t)')
ax.axhline(Z_static, color='gray', ls='--', label=f'static Z={Z_static:.5f}')
ax.plot(t, docc, lw=1.8, color='tab:green', label='d(t)')
ax.axhline(d_static, color='lightgreen', ls='--', label=f'static d={d_static:.5f}')
ax.set_title("Z(t) と d(t)"); ax.set_xlabel("t")
ax.legend(fontsize=8); ax.grid(True)

plt.tight_layout()
plt.savefig(f"noquench_routeA_U{U_val}_T{t_max}.png", dpi=150)
print(f"\nプロット保存: noquench_routeA_U{U_val}_T{t_max}.png")
