import re

with open('td_gga_solver.py', 'r') as f:
    code = f.read()

# Replace dn_kw_dt
old_str = """        # [修正] 符号を正解の -1j に戻す
        dn_kw_dt = np.array([
            -1j * self.omega_k[idx] * (n_kw[idx] @ RRdag - RRdag @ n_kw[idx])
            for idx in range(self.N_omega)
        ])"""

new_str = """        # [修正] 符号を正解の -1j に戻す
        # 追加: λ = Λ^c + h_qp_0 の項を n_kw の時間発展に追加 (理論上の完全な Eq.17)
        lambda_mat = (Lmbdac + h_qp_0).conj()
        dn_kw_dt = np.array([
            -1j * self.omega_k[idx] * (n_kw[idx] @ RRdag - RRdag @ n_kw[idx])
            -1j * (lambda_mat @ n_kw[idx] - n_kw[idx] @ lambda_mat)
            for idx in range(self.N_omega)
        ])"""

if old_str in code:
    code = code.replace(old_str, new_str)
    with open('td_gga_solver.py', 'w') as f:
        f.write(code)
    print("Patch applied successfully.")
else:
    print("Patch failed. Could not find old_str.")

