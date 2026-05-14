"""
run_paper_quench.py
論文 Fig.2 再現: TD-gGA (Route A, Λ=0) クエンチ計算

Guerci, Capone, Lanata, PRR 5, L032023 (2023) の設定:
  U_i = 0.05 (弱相関金属)
  U_f = 複数値 (強相関側へクエンチ)
  半充填 n=0.5, Bethe DOS
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ga_mainfin_routeA import GA
import td_gGA_solver_routeA as td

# ==========================================
# パラメータ設定
# ==========================================
U_initial = 0.05
U_finals  = [0.5, 1.0, 1.5, 2.0, 2.5]   # 複数クエンチ先
t_max     = 10.0
dt        = 0.01
N_freq    = 50

# ==========================================
# 静的計算 (U_i=0.05, Route A Λ=0 ゲージ)
# ==========================================
print("=" * 55)
print(f"静的計算 U_i = {U_initial}  (Route A, Λ=0 ゲージ)")
print("=" * 55)

calc_nqspo = 3
ga_obj = GA(U=U_initial, nghost=4, nphysorb=2, n=0.5, T=0.003, eks=-99)
rinit = np.zeros(calc_nqspo); rinit[0] = 1.0
ga_obj.optimize_selfc_routeA(rinit=rinit, muinit=0.0)
if not ga_obj.lconverged_root:
    print("  直接初期化が収束せず。断熱初期化に切り替えます (U=0.5 → 0.05)...")
    ga_obj = td.solve_static_adiabatic(
        U_target=U_initial, nghost=4, nphysorb=2, n=0.5, T=0.003,
        U_ref=0.5, n_steps=10
    )
else:
    print("  Route A 静的計算: 収束しました。")

Z_static = ga_obj.Z
d_static = float(np.real(ga_obj.imp_solver.docc))
print(f"\n静的値: Z={Z_static:.5f}, d={d_static:.5f}")
print(f"Λ 固有値: {np.linalg.eigvalsh(ga_obj.Lmbda)}")

# ==========================================
# 複数 U_f に対してクエンチ実行
# ==========================================
results_all = {}

for U_f in U_finals:
    print("\n" + "=" * 55)
    print(f"Route A クエンチ: U={U_initial} → {U_f},  t_max={t_max}")
    print("=" * 55)
    res = td.run_full_simulation(ga_obj, U_f, t_max, dt, N_freq=N_freq)
    results_all[U_f] = res

    t       = res['t']
    docc    = res['docc']
    E_emb   = res['E_emb']
    E_phys  = res['E_phys']
    F2      = res['F2']

    def rel_err(arr):
        return (arr - arr[0]) / max(abs(arr[0]), 1e-12)

    print(f"  Z[0]={res['Z'][0]:.5f}  d[0]={docc[0]:.5f}  (静的: Z={Z_static:.5f}, d={d_static:.5f})")
    print(f"  E_emb  最大相対誤差: {np.max(np.abs(rel_err(E_emb))):.3e}")
    print(f"  E_phys 最大相対誤差: {np.max(np.abs(rel_err(E_phys))):.3e}")
    print(f"  F2 最大値:           {np.max(F2):.3e}")

    fname = f"routeA_U{U_initial}_to_{U_f}_T{t_max}"
    np.savez(fname + ".npz", **res,
             Z_static=Z_static, d_static=d_static,
             U_initial=U_initial, U_final=U_f)
    print(f"  データ保存: {fname}.npz")

# ==========================================
# プロット: d(t) (論文 Fig.2 対応)
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"TD-gGA Route A (Λ=0)  U_i={U_initial}  [半充填 Bethe格子]", fontsize=12)

colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(U_finals)))

# 左: d(t)
ax = axes[0]
ax.axhline(d_static, color='gray', ls=':', lw=1.2, label=f'静的 d (U_i={U_initial})')
for col, U_f in zip(colors, U_finals):
    res = results_all[U_f]
    ax.plot(res['t'], res['docc'], color=col, lw=1.8, label=f'$U_f={U_f}$')
ax.set_xlabel("t"); ax.set_ylabel("d(t)")
ax.set_title("二重占有数 d(t)")
ax.legend(fontsize=8); ax.grid(True)

# 右: エネルギー保存チェック (E_phys の相対誤差)
ax = axes[1]
for col, U_f in zip(colors, U_finals):
    res = results_all[U_f]
    t = res['t']
    E = res['E_phys']
    re = np.abs((E - E[0]) / max(abs(E[0]), 1e-12))
    ax.semilogy(t, re + 1e-16, color=col, lw=1.8, label=f'$U_f={U_f}$')
ax.set_xlabel("t"); ax.set_ylabel(r'$|[E_\mathrm{phys}(t)-E_\mathrm{phys}(0)]/E_\mathrm{phys}(0)|$')
ax.set_title("物理エネルギー保存  $E_{phys}=E_{qp}+U_f d$")
ax.legend(fontsize=8); ax.grid(True)

plt.tight_layout()
fname_fig = f"routeA_quench_Ui{U_initial}_Tmax{t_max}.png"
plt.savefig(fname_fig, dpi=150)
print(f"\nプロット保存: {fname_fig}")
