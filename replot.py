"""
td_gga_result.npz から数値を読み込んで描画するだけの軽量スクリプト。
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 日本語フォント設定
fonts = [f for f in fm.findSystemFonts() if 'Hiragino' in f]
if fonts:
    fm.fontManager.addfont(fonts[0])
    plt.rcParams['font.family'] = fm.FontProperties(fname=fonts[0]).get_name()
plt.rcParams['axes.unicode_minus'] = False

data        = np.load('td_gga_result.npz')
t           = data['t']
norms       = data['norms']
occupations = data['occupations']
docc        = data['docc']
Z_t         = data['Z']
U_initial   = float(data['U_initial'])
U_final     = float(data['U_final'])
B           = occupations.shape[0]

print(f"d(t=0) = {docc[0]:.4f},  d(t=T) = {docc[-1]:.4f},  変化幅 = {docc.max()-docc.min():.4f}")
print(f"Z(t=0) = {Z_t[0]:.4f},  Z(t=T) = {Z_t[-1]:.4f},  変化幅 = {Z_t.max()-Z_t.min():.4f}")

import os, glob

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
ax_norm = axes[0, 0]
ax_occ  = axes[0, 1]
ax_docc = axes[1, 0]
ax_Z    = axes[1, 1]

eq_text = (f"平衡状態 ($U_i={U_initial}$):  "
           f"$Z_0={Z_t[0]:.4f}$,  "
           f"$d_0={docc[0]:.4f}$")
fig.suptitle(
    f'TD-gGA クエンチダイナミクス  $U_i={U_initial}$ → $U_f={U_final}$\n{eq_text}',
    fontsize=11)

# --- ノルム ---
ax_norm.plot(t, norms, 'k-', linewidth=1.5)
ax_norm.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, label='理想値 = 1')
ax_norm.set_ylabel(r'$\langle\Phi|\Phi\rangle$')
ax_norm.set_xlabel(r'時刻 $t$')
ax_norm.legend()
norm_dev = max(abs(norms - 1.0).max() * 1.5, 1e-4)
ax_norm.set_ylim([1.0 - norm_dev, 1.0 + norm_dev])

# --- 準粒子占有数 ---
for orb in range(B):
    ax_occ.plot(t, occupations[orb, :], label=f'軌道 {orb}')
ax_occ.set_ylabel(r'$n_{aa}(t)$')
ax_occ.set_xlabel(r'時刻 $t$')
ax_occ.legend()

# --- 二重占有数 ---
ax_docc.plot(t, docc, 'g-', linewidth=1.5)
ax_docc.axhline(docc[0], color='gray', linestyle='--', linewidth=0.8,
                label=f'初期値 $d_0={docc[0]:.4f}$')
ax_docc.set_ylabel(r'$d(t) = \langle n_\uparrow n_\downarrow \rangle$')
ax_docc.set_xlabel(r'時刻 $t$')
ax_docc.set_ylim(bottom=0)
ax_docc.legend()

# --- 準粒子重み ---
ax_Z.plot(t, np.clip(Z_t, 0, 1), 'm-', linewidth=1.5)
ax_Z.axhline(min(Z_t[0], 1.0), color='gray', linestyle='--', linewidth=0.8,
             label=f'初期値 $Z_0={Z_t[0]:.4f}$')
ax_Z.set_ylabel(r'$Z(t)$')
ax_Z.set_xlabel(r'時刻 $t$')
ax_Z.set_ylim([0.0, 1.0])
ax_Z.legend()

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
print(f"{save_name} を保存しました。")
