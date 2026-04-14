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

fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

# --- ノルム ---
axes[0].plot(t, norms, 'k-', linewidth=1.5)
axes[0].axhline(1.0, color='gray', linestyle='--', linewidth=0.8, label='理想値 = 1')
axes[0].set_ylabel(r'$\langle\Phi|\Phi\rangle$')
axes[0].set_title(f'TD-gGA クエンチダイナミクス  $U_i={U_initial}$ → $U_f={U_final}$')
axes[0].legend()
norm_dev = max(abs(norms - 1.0).max() * 1.5, 1e-4)
axes[0].set_ylim([1.0 - norm_dev, 1.0 + norm_dev])

# --- 準粒子占有数 ---
for orb in range(B):
    axes[1].plot(t, occupations[orb, :], label=f'軌道 {orb}')
axes[1].set_ylabel(r'$n_{aa}(t)$')
axes[1].legend()

# --- 二重占有数 ---
axes[2].plot(t, docc, 'g-', linewidth=1.5)
axes[2].set_ylabel(r'$d(t) = \langle n_\uparrow n_\downarrow \rangle$')
axes[2].set_ylim(bottom=0)

# --- 準粒子重み ---
axes[3].plot(t, Z_t, 'm-', linewidth=1.5)
axes[3].set_ylabel(r'$Z(t)$')
axes[3].set_xlabel('時刻 $t$')
axes[3].set_ylim([0.0, 1.0])

plt.tight_layout()
plt.savefig('quench_dynamics.png', dpi=150)
plt.show()
print("quench_dynamics.png を更新しました。")
