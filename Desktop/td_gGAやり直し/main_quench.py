"""
main_quench.py
静的計算と時間発展を繋ぎ、クエンチダイナミクスを実行・描画するスクリプト
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Mac用日本語フォント設定
fonts = [f for f in fm.findSystemFonts() if 'Hiragino' in f]
if fonts:
    fm.fontManager.addfont(fonts[0])
    plt.rcParams['font.family'] = fm.FontProperties(fname=fonts[0]).get_name()
plt.rcParams['axes.unicode_minus'] = False

# 1. 静的計算の実行（インポートするだけで ga_mainfin のグローバル処理が走ります）
import ga_mainfin
import td_gGA_solver_frozen as td

# ga_mainfin で計算された GA_list の最初の結果（例えば U=0.1 の状態）を取得
ga_obj = ga_mainfin.GA_list[0]
U_initial = ga_obj.U

# 2. クエンチの設定（デバッグ用の特異点回避 U_final = 0.01）
U_final = 0.01
t_max = 10.0
dt = 0.01  # 細かい時間刻みで安全に

print("\n" + "="*40)
print(f" Starting Frozen-D Quench Dynamics")
print(f" U: {U_initial:.3f} -> {U_final:.3f}")
print("="*40)

# 3. 新しいTDソルバーで時間発展を実行！
results = td.run_frozen_simulation(ga_obj, U_final, t_max, dt)

# ====== ここを修正 ======
# 保存用のファイル名のベースを定義しておく
filename_base = f"frozen_quench_U{U_initial}_to_{U_final}"

# 結果を保存 (.npz)
np.savez(f'{filename_base}.npz', 
         t=results['t'], docc=results['docc'], E_tot=results['E_tot'], n_orb=results['n_orb'])
# ======================

# 4. 結果のプロット
t = results['t']
docc = results['docc']
E_tot = results['E_tot']
n_orb = results['n_orb']

fig, axes = plt.subplots(3, 1, figsize=(8, 12)) # 3段に変更
fig.suptitle(f'Frozen-D Quench ($U={U_initial} \\to {U_final}$)', fontsize=16)

# 上段：エネルギー保存
axes[0].plot(t, E_tot - E_tot[0], 'r-', linewidth=2)
axes[0].set_title("Energy Conservation (Error)")
axes[0].set_ylabel(r'$E(t) - E(0)$')
axes[0].grid(True)

# 中段：二重占有数 d(t)
axes[1].plot(t, docc, 'g-', linewidth=2)
axes[1].axhline(docc[0], color='gray', linestyle='--', label='Initial')
axes[1].set_title("Double Occupancy $d(t)$")
axes[1].set_ylabel(r'$d(t)$')
axes[1].legend()
axes[1].grid(True)

# 下段：各軌道の電子数 n_aa(t)
nqspo = len(n_orb)
colors = ['tab:blue', 'tab:orange', 'tab:green']
for i in range(nqspo):
    label_name = f'軌道 {i} (物理)' if i == 0 else f'軌道 {i} (ゴースト)'
    axes[2].plot(t, n_orb[i], label=label_name, color=colors[i%3], linewidth=2)

axes[2].set_title("Orbital Occupations $n_{aa}(t)$")
axes[2].set_ylabel(r'$n_{aa}(t)$')
axes[2].set_xlabel("Time $t$")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
# 画像として保存する設定を追加
plt.savefig(f"{filename_base}.png", dpi=300) # 高解像度(300dpi)で保存
print(f"グラフ画像を保存しました: {filename_base}.png")
plt.show()