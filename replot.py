"""
td_gga_result.npz から読み込んで即座に再プロットするスクリプト。
run_simulation.py を再実行しなくても、プロットだけ更新できる。
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import linalg as LA
import td_gga_solver as td

# 日本語フォント設定
fonts = [f for f in fm.findSystemFonts() if 'Hiragino' in f]
if fonts:
    fm.fontManager.addfont(fonts[0])
    plt.rcParams['font.family'] = fm.FontProperties(fname=fonts[0]).get_name()
plt.rcParams['axes.unicode_minus'] = False

data      = np.load('td_gga_result.npz', allow_pickle=False)
t         = data['t']
y         = data['y']
dim_Phi   = int(data['dim_Phi'])
B         = int(data['B'])
U_initial = float(data['U_initial'])
U_final   = float(data['U_final'])

# エネルギー計算に必要なデータ（新しい npz にのみ存在）
has_energy_data = ('H_emb_0' in data) and ('op_cb' in data)
if has_energy_data:
    H_emb_0 = data['H_emb_0']
    H_phys  = data['H_phys']
    op_cb   = data['op_cb']

nsteps      = len(t)
norms       = np.zeros(nsteps)
occupations = np.zeros((B, nsteps))
traces      = np.zeros(nsteps)
eig_min     = np.zeros(nsteps)
eig_max     = np.zeros(nsteps)
E_emb       = np.zeros(nsteps)   # ⟨Φ|H_emb^0|Φ⟩
E_qp        = np.zeros(nsteps)   # Tr(R† H_phys R · n)

I = np.eye(B)

for k in range(nsteps):
    Phi_t, n_ab_t = td.unpack_state(y[:, k], dim_Phi, B)
    norms[k]          = np.real(Phi_t.conj() @ Phi_t)
    occupations[:, k] = np.real(np.diag(n_ab_t))
    traces[k]         = np.real(np.trace(n_ab_t))
    eigs              = np.linalg.eigvalsh((n_ab_t + n_ab_t.conj().T) / 2)
    eig_min[k]        = eigs.min()
    eig_max[k]        = eigs.max()

    if has_energy_data:
        # 埋め込みエネルギー ⟨Φ|H_emb^0|Φ⟩
        E_emb[k] = np.real(Phi_t.conj() @ H_emb_0 @ Phi_t)

        # 準粒子エネルギー Tr(R† H_phys R · n)
        Delta = (n_ab_t + n_ab_t.conj().T) / 2.0
        fdaggerc = np.array([[Phi_t.conj() @ op_cb[a, al] @ Phi_t
                              for al in range(B)] for a in range(B)])
        sqrt_D1mD = LA.sqrtm(Delta @ (I - Delta))
        R = LA.pinv(sqrt_D1mD) @ fdaggerc
        h_qp_0 = R.conj().T @ H_phys @ R
        E_qp[k] = np.real(np.trace(h_qp_0 @ n_ab_t))

print(f"Tr(n) 変化幅:  {traces.max()-traces.min():.2e}  (理想: 0)")
print(f"固有値範囲:   [{eig_min.min():.4f}, {eig_max.max():.4f}]  (物理: [0,1])")
if has_energy_data:
    print(f"E_emb 変化幅: {E_emb.max()-E_emb.min():.4e}  (保存なら小さいほど良い)")
    print(f"E_qp  変化幅: {E_qp.max()-E_qp.min():.4e}")
    E_tot = E_emb + E_qp
    print(f"E_tot 変化幅: {E_tot.max()-E_tot.min():.4e}  (クエンチ後は保存されるべき)")

# ============================================================
# プロット
# ============================================================
nplots = 4 if has_energy_data else 2
fig, axes = plt.subplots(nplots, 1, figsize=(8, 3 * nplots), sharex=True)

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

if has_energy_data:
    # --- 埋め込みエネルギー ---
    axes[2].plot(t, E_emb, 'b-', linewidth=1.2, label=r'$\langle\Phi|H_\mathrm{emb}^0|\Phi\rangle$')
    axes[2].set_ylabel('$E_\\mathrm{emb}$')
    axes[2].legend()

    # --- 合計エネルギー ---
    axes[3].plot(t, E_emb + E_qp, 'r-', linewidth=1.2, label='$E_\\mathrm{emb} + E_\\mathrm{qp}$')
    axes[3].set_ylabel('$E_\\mathrm{tot}$ (近似)')
    axes[3].legend()

axes[-1].set_xlabel('時刻 $t$')
plt.tight_layout()
plt.savefig('quench_dynamics.png', dpi=150)
plt.show()
print("quench_dynamics.png を更新しました。")
