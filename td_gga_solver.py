import numpy as np
from scipy import integrate, linalg as LA

# ==========================================
# 1. パッキング・アンパック用ヘルパー関数
# ==========================================
def pack_state(Phi, n_ab=None):
    """
    Phi のみを ODE ソルバー用の1次元実数配列にパックする。
    n_ab は Phi から毎ステップ直接計算するため ODE 状態に含めない。
    """
    return np.concatenate([Phi.real, Phi.imag])

def unpack_state(Y_flat, dim_Phi, B=None):
    """
    1次元実数配列から Phi を復元する。
    n_ab は None を返す（呼び出し側で op_bb から計算すること）。
    """
    Phi = Y_flat[:dim_Phi] + 1j * Y_flat[dim_Phi:2 * dim_Phi]
    return Phi, None

# ==========================================
# 2. Lambda_c の直接計算（シルベスター方程式）
# ==========================================
def compute_Lambda_c(Delta, K):
    """
    Delta の固有値基底でシルベスター方程式 [Delta, Lambda^c] = K を解き、
    Lambda^c を返す。対角成分（ゲージ自由度）は 0 とする。

    Parameters
    ----------
    Delta : (B, B) complex ndarray — 局所密度行列（エルミート）
    K     : (B, B) complex ndarray — 右辺行列

    Returns
    -------
    Lmbdac : (B, B) complex ndarray
    """
    eigenvalues, U = LA.eigh(Delta)
    K_tilde = U.conj().T @ K @ U

    B = len(eigenvalues)
    Lmbdac_tilde = np.zeros((B, B), dtype=complex)
    epsilon = 1e-4  # ローレンツ型スムージングの幅
    for i in range(B):
        for j in range(B):
            if i != j:
                diff = eigenvalues[j] - eigenvalues[i]
                Lmbdac_tilde[i, j] = K_tilde[i, j] * diff / (diff**2 + epsilon**2)

    Lmbdac = U @ Lmbdac_tilde @ U.conj().T
    return (Lmbdac + Lmbdac.conj().T) / 2.0

# ==========================================
# 3. ODE時間発展のコアクラス
# ==========================================
class TDgGADynamics:
    def __init__(self, dim_Phi, B, H_emb_0_fock, H_phys_1body):
        """
        物理パラメータを保持するクラス。
        D は H_emb_0_fock に事前に組み込み済み (Frozen-D 近似)。

        前提: インスタンス生成後に以下の属性を設定しておくこと。
              system.op_cb = op_cb  — (B,1,dim_Phi,dim_Phi) c†_α b_a
              system.op_bb = op_bb  — (B,B,dim_Phi,dim_Phi) b†_b b_a
        """
        self.dim_Phi  = dim_Phi
        self.B        = B
        self.H_emb_0  = H_emb_0_fock
        self.H_phys   = H_phys_1body

    def compute_derivatives(self, t, Y_flat):
        """
        ODE の右辺。n_ab は Phi から直接計算することで拘束条件を厳密に維持する。
        """
        B = self.B

        # 1. Phi のみアンパック
        Phi = Y_flat[:self.dim_Phi] + 1j * Y_flat[self.dim_Phi:]

        # 2. n_ab を Phi から直接計算（拘束条件 n_ab = <Phi|op_bb|Phi> を厳密に保つ）
        n_ab = np.array([[Phi.conj() @ self.op_bb[a, b] @ Phi
                          for b in range(B)] for a in range(B)])

        # 3. Delta をエルミート化
        Delta = (n_ab + n_ab.conj().T) / 2.0

        # 4. R の計算
        fdaggerc = np.array([[Phi.conj() @ self.op_cb[a, 0] @ Phi] for a in range(B)])
        eigs_D, U_D = LA.eigh(Delta)
        eigs_D = np.clip(np.real(eigs_D), 1e-4, 1.0 - 1e-4)
        inv_sqrt_eigs = 1.0 / np.sqrt(eigs_D * (1.0 - eigs_D))
        R = (U_D * inv_sqrt_eigs[np.newaxis, :]) @ U_D.conj().T @ fdaggerc

        # 5. h_qp^0 = R H_phys R† （Λ^c = 0 の仮計算）
        h_qp_0 = R @ self.H_phys @ R.conj().T

        # 6. K 行列（拘束条件の整合性条件）
        comm_n_hqp = n_ab @ h_qp_0.T - h_qp_0.T @ n_ab
        K = np.zeros((B, B), dtype=complex)
        for a in range(B):
            for b in range(B):
                comm_op = self.op_bb[a, b] @ self.H_emb_0 - self.H_emb_0 @ self.op_bb[a, b]
                K[a, b] = Phi.conj() @ comm_op @ Phi
        K -= comm_n_hqp

        # 7. Λ^c を決定
        Lmbdac = compute_Lambda_c(Delta, K)

        # 8. H_emb = H_emb^0 + Σ Λ^c_{ab} op_bb[a,b]
        H_emb = np.array(self.H_emb_0, dtype=complex)
        for a in range(B):
            for b in range(B):
                if abs(Lmbdac[a, b]) > 1e-14:
                    H_emb += Lmbdac[a, b] * self.op_bb[a, b]

        H_emb = (H_emb + H_emb.conj().T) / 2.0

        # 9. シュレーディンガー方程式: dPhi/dt = -i H_emb Phi
        dPhi_dt = -1j * (H_emb @ Phi)

        return pack_state(dPhi_dt)

# ==========================================
# 4. メイン実行関数
# ==========================================
def run_quench_dynamics(Phi_0, n_ab_0, t_max, dt, physics_params, rtol=1e-8, atol=1e-10):
    """
    時間発展を実行するメイン関数。n_ab_0 は互換性のため残すが使用しない。
    """
    dim_Phi      = physics_params['dim_Phi']
    B            = physics_params['B']
    H_emb_0_fock = physics_params['H_emb_0_fock']
    H_phys_1body = physics_params['H_phys_1body']
    op_cb        = physics_params['op_cb']
    op_bb        = physics_params['op_bb']

    system = TDgGADynamics(dim_Phi, B, H_emb_0_fock, H_phys_1body)
    system.op_cb = op_cb
    system.op_bb = op_bb

    # Phi のみパック（n_ab は ODE 状態に含めない）
    Y0_flat = pack_state(Phi_0)

    t_eval = np.arange(0, t_max, dt)

    sol = integrate.solve_ivp(
        fun=system.compute_derivatives,
        t_span=(0, t_max),
        y0=Y0_flat,
        method='RK45',
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )

    return sol

if __name__ == "__main__":
    pass
