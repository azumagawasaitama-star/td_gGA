import numpy as np

d = np.load('td_gga_result.npz')
norms = d['norms']
print(f"Norms span: {np.ptp(norms):.3e}")

print(f"docc start: {d['docc'][0]:.6f}, end: {d['docc'][-1]:.6f}")

E_tot = d['E_tot']
print(f"E_tot drift: {E_tot[-1] - E_tot[0]:.6f}")

# Check the quasi-particle energy
Z = d['Z_dressed']
print(f"Z start: {Z[0]:.6f}, end: {Z[-1]:.6f}")

