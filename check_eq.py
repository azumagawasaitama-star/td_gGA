import numpy as np

d = np.load('td_gga_result.npz')
print(f"U_initial: {d['U_initial']}, U_final: {d['U_final']}")
print(f"E_emb start: {d['E_emb'][0]:.6f}")
print(f"docc start: {d['docc'][0]:.6f}")

