"""
run_td_quench.py
GAクラスを直接呼び出し、1回だけ静的計算を行ってからクエンチを実行する専用スクリプト
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ==========================================
# 日本語フォント設定 (Mac用)
# ==========================================
fonts = [f for f in fm.findSystemFonts() if 'Hiragino' in f]
if fonts:
    fm.fontManager.addfont(fonts[0])
    plt.rcParams['font.family'] = fm.FontProperties(fname=fonts[0]).get_name()
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# モジュールのインポート
# ※お手元のファイル名に合わせて変更してください
# ==========================================
from ga_mainfin import GA
import td_gGA_solver_frozen as td

# ==========================================
# 1. 時間発展用の初期状態を 1回だけ 計算する
# ==========================================
U_initial = 0.001
print("="*40)
print(f" 1. 静的計算の実行 (U={U_initial}) ")
print("="*40)

# パラメータ設定 (U=0.1, nghost=4, nphysorb=2, n=0.5, T=0.003)
ga_obj = GA(U=U_initial, nghost=4, nphysorb=2, n=0.5, T=0.003, eks=-99)

# 初期ゲスの設定 (金属状態)
calc_nqspo = 3
rinit = np.zeros(calc_nqspo)
rinit[0] = 1.0
lambdainit = np.zeros(calc_nqspo * (calc_nqspo + 1) // 2)

# 静的計算を実行！(余計なループは走りません)
ga_obj.optimize_selfc_new(rinit, lambdainit, muinit=0.0)

print(f"\n---> 静的計算 完了！ (Final Z: {ga_obj.Z:.5f})\n")

# ==========================================
# 2. クエンチの設定と実行
# ==========================================
U_final = 2.5
t_max = 10.0
dt = 0.01

print("="*40)
print(f" 2. 時間発展 (TD) クエンチの開始")
print(f" U: {U_initial:.3f} -> {U_final:.3f}")
print("="*40)

# 新しいTDソルバーで時間発展を実行！
results = td.run_frozen_simulation(ga_obj, U_final, t_max, dt)
print("---> 時間発展 完了！\n")

# ==========================================
# 3. データの保存
# ==========================================
filename_base = f"quench_U{U_initial}_to_{U_final}_T{t_max}"

# .npz 形式でデータを保存
np.savez(f"{filename_base}.npz", **results)
print(f"計算データを保存しました: {filename_base}.npz")

# ==========================================
# 4. 結果のプロット (2×2)
# ==========================================
t     = results['t']
docc  = results['docc']
E_tot = results['E_tot']
Z     = results['Z']
n_orb = results['n_orb']

# 静的計算の参照値
Z_static = ga_obj.Z
d_static = np.real(ga_obj.imp_solver.docc)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(f'Frozen-D Quench ($U={U_initial} \\to {U_final}$)', fontsize=16)

# (0,0): エネルギー保存（相対誤差）
axes[0, 0].plot(t, (E_tot - E_tot[0]) / np.abs(E_tot[0]), 'r-', linewidth=2)
axes[0, 0].set_title("Energy Conservation (Relative Error)")
axes[0, 0].set_ylabel(r'$[E(t) - E(0)] / |E(0)|$')
axes[0, 0].grid(True)

# (0,1): 準粒子重み Z(t)
axes[0, 1].plot(t, Z, 'b-', linewidth=2, label='$Z(t)$')
axes[0, 1].axhline(Z_static, color='gray', linestyle='--',
                   label=f'Static $Z={Z_static:.4f}$')
axes[0, 1].set_title("Quasi-particle Weight $Z(t)$")
axes[0, 1].set_ylabel(r'$Z(t)$')
axes[0, 1].legend()
axes[0, 1].grid(True)

# (1,0): 二重占有数 d(t)
axes[1, 0].plot(t, docc, 'g-', linewidth=2, label='$d(t)$')
axes[1, 0].axhline(d_static, color='gray', linestyle='--',
                   label=f'Static $d={d_static:.4f}$')
axes[1, 0].set_title("Double Occupancy $d(t)$")
axes[1, 0].set_ylabel(r'$d(t)$')
axes[1, 0].set_xlabel("Time $t$")
axes[1, 0].legend()
axes[1, 0].grid(True)

# (1,1): 各軌道の電子数 n_aa(t)
n_orb_count = n_orb.shape[0]
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
for i in range(n_orb_count):
    label_name = f'軌道 {i} (物理)' if i == 0 else f'軌道 {i} (ゴースト)'
    axes[1, 1].plot(t, n_orb[i], label=label_name,
                    color=colors[i % len(colors)], linewidth=2)
axes[1, 1].set_title("Orbital Occupations $n_{aa}(t)$")
axes[1, 1].set_ylabel(r'$n_{aa}(t)$')
axes[1, 1].set_xlabel("Time $t$")
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()

# 画像として保存
plt.savefig(f"{filename_base}.png", dpi=300)
print(f"グラフ画像を保存しました: {filename_base}.png")

plt.show()