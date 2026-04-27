import numpy as np
import matplotlib.pyplot as plt

# データの読み込み
data = np.load('td_gga_result.npz')
t = data['t']
E_tot_t = data['E_tot']
docc = data['docc']
Z_bare = data['Z_bare']
Z_dressed = data['Z_dressed']
occupations = data['occupations']
U_initial = data['U_initial']
U_final = data['U_final']
B = occupations.shape[0]

# プロットの作成
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
ax_etot = axes[0, 0]
ax_occ  = axes[0, 1]
ax_docc = axes[1, 0]
ax_Z    = axes[1, 1]

fig.suptitle(f'TD-gGA クエンチダイナミクス  $U_i={U_initial}$ -> $U_f={U_final}', fontsize=12)

# --- エネルギー保存 ---
ax_etot.plot(t, E_tot_t, color='blue', linewidth=2.0, label='Conserved $E_{\\rm tot}(t)$')
ax_etot.axhline(E_tot_t[0], color='red', linestyle='--', linewidth=1.0, label=f'Target $E_0={E_tot_t[0]:.4f}$')
ax_etot.set_ylabel(r'$E_{\rm tot}(t)$')
ax_etot.set_xlabel('Time $t$')
ax_etot.set_title('Total Energy Conservation (TDVP)')
ax_etot.legend(loc='upper right')
ax_etot.grid(True, linestyle=':')

# --- 占有数 ---
for orb in range(B):
    ax_occ.plot(t, occupations[orb, :], label=f'Orbital {orb}')
ax_occ.set_ylabel('$n_{aa}(t)$')
ax_occ.legend()
ax_occ.grid(True, linestyle=':')

# --- 二重占有数 ---
ax_docc.plot(t, docc, color='green', linewidth=1.5)
ax_docc.set_ylabel('$d(t)$')
ax_docc.set_xlabel('Time $t$')
ax_docc.set_title('Double Occupancy Dynamics')
ax_docc.grid(True, linestyle=':')

# --- Z ---
ax_Z.plot(t, Z_bare, label='$Z_{\\rm bare}$', alpha=0.5)
ax_Z.plot(t, Z_dressed, label='$Z_{\\rm dressed}$', color='red', linewidth=2.0)
ax_Z.set_ylabel('$Z(t)$')
ax_Z.set_xlabel('Time $t$')
ax_Z.legend()
ax_Z.grid(True, linestyle=':')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('quench_final_fixed.png', dpi=150)
print("Successfully generated quench_final_fixed.png")
