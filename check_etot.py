import numpy as np

d = np.load('td_gga_result.npz')
E = d['E_tot']
t = d['t']

for i in range(0, len(t), 200):
    print(f"t={t[i]:.2f}, E={E[i]:.6f}")

print(f"t={t[-1]:.2f}, E={E[-1]:.6f}")

