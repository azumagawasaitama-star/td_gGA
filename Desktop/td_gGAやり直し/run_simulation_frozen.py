import numpy as np
# run_simulation_frozen.py などの別ファイルで

# --- 演算子の準備 (フル空間から正しく切り出したもの) ---
params = build_operators_correctly(...) 

# --- 時間発展の実行 ---
y0 = pack_state_frozen(Phi_0)
sol = solve_ivp(compute_derivatives_frozen, [0, t_max], y0, args=(params,), method='RK45', ...)

# --- 事後解析 (エネルギーやd(t)の計算) ---
E_tot_list = []
d_list = []
for k in range(len(sol.t)):
    Phi_t = unpack_state_frozen(sol.y[:, k], dim_Phi)
    Phi_t /= np.linalg.norm(Phi_t)
    
    # ここで初めて E_tot や d(t) を Phi_t から計算してリストに保存