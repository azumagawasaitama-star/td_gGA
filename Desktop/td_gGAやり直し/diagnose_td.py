"""
diagnose_td.py: ODEのステップ数・Delta固有値・H_emb固有値を確認する
"""
import numpy as np
from ga_mainfin import GA
import td_gGA_solver_full as td
import convenience_routines as cr

# 静的計算
U_initial = 1.0
ga_obj = GA(U=U_initial, nghost=4, nphysorb=2, n=0.5, T=0.003, eks=-99)
calc_nqspo = 3
rinit = np.zeros(calc_nqspo); rinit[0] = 1.0
lambdainit = np.zeros(calc_nqspo * (calc_nqspo + 1) // 2)
ga_obj.optimize_selfc_new(rinit, lambdainit, muinit=0.0)
print(f"Z={ga_obj.Z:.5f}")

# パラメータ準備
N_freq = 50
U_final = 1.0
params = td.prepare_full_params(ga_obj, U_final, N_freq)
omega_arr = params['omega_arr']
nqspo = ga_obj.nqspo

# 初期 n(omega)
R_init = ga_obj.R
Lmbda_init = ga_obj.Lmbda
RopRt = R_init @ R_init.T
n0_omega = np.array([cr.calc_C(RopRt * omega + Lmbda_init, T=ga_obj.T)
                     for omega in omega_arr])

# Delta の固有値を確認
w_rho = params['weights'] * params['rho_arr']
Delta_init = np.einsum('f,fab->ab', w_rho, np.real(n0_omega))
evals_Delta = np.linalg.eigvalsh(Delta_init)
print(f"\nDelta 固有値: {evals_Delta}")
print(f"  → denR = [x(1-x)]^(-1/2) の最大値: {max(1/np.sqrt(np.clip(evals_Delta*(1-evals_Delta),1e-14,1))):.2e}")

# H_emb の固有値範囲を確認
Phi_0 = params['Phi_0']
y0 = td.pack_state(Phi_0, n0_omega)
from td_gGA_solver_full import unpack_state, compute_RDLc, build_H_emb_fast
Phi, n_omega = unpack_state(y0, params['dim_Phi'], N_freq, nqspo)
R, D, Lmbdac, Delta = compute_RDLc(Phi, n_omega, params, ga_obj)
D_sp = np.real(D[:, 0])
H_emb = build_H_emb_fast(D_sp, np.real(Lmbdac), params['H_const'], params['M_D'], params['M_Lc_full'])
evals_Hemb = np.linalg.eigvalsh(H_emb)
print(f"\nH_emb 固有値範囲: [{evals_Hemb.min():.4f}, {evals_Hemb.max():.4f}]")
print(f"  → 最大振動周波数 λ_max = {abs(evals_Hemb).max():.4f}")
print(f"  → RK45 推定ステップ幅 h ~ {(1e-6)**0.2 / abs(evals_Hemb).max():.4f}")
print(f"  → t=3 の推定ステップ数 ~ {int(3.0 / ((1e-6)**0.2 / abs(evals_Hemb).max()) * 6)}")

# n(omega) の最大時間微分
dYdt = td.compute_derivatives_full(0.0, y0, params, ga_obj)
_, dn_omega_dt = unpack_state(dYdt, params['dim_Phi'], N_freq, nqspo)
print(f"\n|dn/dt|_max = {np.abs(dn_omega_dt).max():.4e}")
print(f"|dPhi/dt|   = {np.linalg.norm(dYdt[:2*params['dim_Phi']]):.4e}")
