"""
run_full_quench.py
Full TD-gGA: |Phi> と n(omega) を同時に時間発展させるクエンチ計算
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fonts = [f for f in fm.findSystemFonts() if 'Hiragino' in f]
if fonts:
    fm.fontManager.addfont(fonts[0])
    plt.rcParams['font.family'] = fm.FontProperties(fname=fonts[0]).get_name()
plt.rcParams['axes.unicode_minus'] = False

from ga_mainfin import GA
import td_gGA_solver_full as td

# ==========================================
# 1. 静的計算
# ==========================================
U_initial = 1.0
print("="*40)
print(f" 1. 静的計算 (U={U_initial})")
print("="*40)

ga_obj = GA(U=U_initial, nghost=4, nphysorb=2, n=0.5, T=0.003, eks=-99)

calc_nqspo = 3
rinit = np.zeros(calc_nqspo)
rinit[0] = 1.0
lambdainit = np.zeros(calc_nqspo * (calc_nqspo + 1) // 2)

ga_obj.optimize_selfc_new(rinit, lambdainit, muinit=0.0)
print(f"\n---> 静的計算 完了！ (Z={ga_obj.Z:.5f})\n")

# ==========================================
# 2. クエンチ設定と実行
# ==========================================
U_final = 2.5
t_max   = 3.0
dt      = 0.01
N_freq  = 50

print("="*40)
print(f" 2. Full TD-gGA クエンチ開始")
print(f" U: {U_initial:.3f} -> {U_final:.3f}")
print("="*40)

results = td.run_full_simulation(ga_obj, U_final, t_max, dt, N_freq=N_freq)
print("---> 時間発展 完了！\n")

# ==========================================
# 3. データ保存
# ==========================================
filename_base = f"full_quench_U{U_initial}_to_{U_final}_T{t_max}"
np.savez(f"{filename_base}.npz", **results)
print(f"データ保存: {filename_base}.npz")

# ==========================================
# 4. プロット (2x3)
# ==========================================
t        = results['t']
docc     = results['docc']
E_tot    = results['E_tot']
Z        = results['Z']
n_orb    = results['n_orb']
eig_Lc   = results['eig_Lc']
sqrtDtD  = results['sqrtDtD']

Z_static = ga_obj.Z
d_static = float(np.real(ga_obj.imp_solver.docc))

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(f'Full TD-gGA Quench ($U={U_initial} \\to {U_final}$)', fontsize=16)

# (0,0): エネルギー保存
axes[0, 0].plot(t, (E_tot - E_tot[0]) / max(abs(E_tot[0]), 1e-12), 'r-', linewidth=2)
axes[0, 0].set_title("Energy Conservation (Relative Error)")
axes[0, 0].set_ylabel(r'$[E(t)-E(0)]/|E(0)|$')
axes[0, 0].grid(True)

# (0,1): 準粒子重み Z(t)
axes[0, 1].plot(t, Z, 'b-', linewidth=2, label='$Z(t)$')
axes[0, 1].axhline(Z_static, color='gray', linestyle='--', label=f'Static Z={Z_static:.4f}')
axes[0, 1].set_title("Quasi-particle Weight $Z(t)$")
axes[0, 1].set_ylabel(r'$Z(t)$')
axes[0, 1].legend()
axes[0, 1].grid(True)

# (0,2): sqrt(D†D)
axes[0, 2].plot(t, sqrtDtD, 'm-', linewidth=2)
axes[0, 2].set_title(r"$\sqrt{D^\dagger D}(t)$")
axes[0, 2].set_ylabel(r'$\sqrt{D^\dagger D}$')
axes[0, 2].grid(True)

# (1,0): 二重占有数 d(t)
axes[1, 0].plot(t, docc, 'g-', linewidth=2, label='$d(t)$')
axes[1, 0].axhline(d_static, color='gray', linestyle='--', label=f'Static d={d_static:.4f}')
axes[1, 0].set_title("Double Occupancy $d(t)$")
axes[1, 0].set_ylabel(r'$d(t)$')
axes[1, 0].set_xlabel("Time $t$")
axes[1, 0].legend()
axes[1, 0].grid(True)

# (1,1): 各軌道の電子数
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
for i in range(n_orb.shape[0]):
    label = f'軌道{i} (物理)' if i == 0 else f'軌道{i} (ゴースト)'
    axes[1, 1].plot(t, n_orb[i], label=label, color=colors[i % len(colors)], linewidth=2)
axes[1, 1].set_title("Orbital Occupations")
axes[1, 1].set_ylabel(r'$n_{aa}(t)$')
axes[1, 1].set_xlabel("Time $t$")
axes[1, 1].legend()
axes[1, 1].grid(True)

# (1,2): eig(Λ^c)
for i in range(eig_Lc.shape[1]):
    axes[1, 2].plot(t, eig_Lc[:, i], linewidth=1.5, label=f'λ{i+1}')
axes[1, 2].set_title(r"Eigenvalues of $\Lambda^c(t)$")
axes[1, 2].set_ylabel(r'eig $\Lambda^c$')
axes[1, 2].set_xlabel("Time $t$")
axes[1, 2].legend()
axes[1, 2].grid(True)

plt.tight_layout()
plt.savefig(f"{filename_base}.png", dpi=150)
print(f"グラフ保存: {filename_base}.png")
plt.show()
