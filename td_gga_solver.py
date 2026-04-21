import numpy as np
from scipy import integrate, linalg as LA

# ==========================================
# 1. パッキング・アンパック用ヘルパー関数
# ==========================================
def pack_state(Phi, n_kw=None):
    """
    ODE ソルバー用の1次元実数配列にパックする。

    Frozen-D モード (n_kw=None):
        Y = [Φ.real, Φ.imag]                          長さ 2*dim_Phi

    Full TD-gGA モード (n_kw は (N_omega, B, B) complex):
        Y = [Φ.real, Φ.imag, n_kw.real.flat, n_kw.imag.flat]
    """
    parts = [Phi.real, Phi.imag]
    if n_kw is not None:
        parts.extend([n_kw.real.flatten(), n_kw.imag.flatten()])
    return np.concatenate(parts)


def unpack_state(Y_flat, dim_Phi, B=None, N_omega=0):
    """
    1次元実数配列から Phi と n(ω_k) を復元する。

    Parameters
    ----------
    Y_flat  : 1D real array
    dim_Phi : int — Φ の次元
    B       : int — バス軌道数 (Full モードで必要)
    N_omega : int — 周波数メッシュ数 (0 のとき Frozen-D モード → n_kw=None)

    Returns
    -------
    Phi  : (dim_Phi,) complex
    n_kw : (N_omega, B, B) complex  or  None
    """
    Phi = Y_flat[:dim_Phi] + 1j * Y_flat[dim_Phi:2 * dim_Phi]
    if N_omega > 0 and B is not None:
        offset = 2 * dim_Phi
        sz = N_omega * B * B
        n_kw_re = Y_flat[offset         : offset + sz    ].reshape(N_omega, B, B)
        n_kw_im = Y_flat[offset + sz    : offset + 2 * sz].reshape(N_omega, B, B)
        return Phi, n_kw_re + 1j * n_kw_im
    return Phi, None


# ==========================================
# 2. Lambda_c の直接計算（シルベスター方程式）
# ==========================================
def compute_Lambda_c(Delta, K, is_frozen_D=True):
    """
    Delta の固有値基底でシルベスター方程式 [Delta, Lambda^c] = K を解き、
    Lambda^c を返す。対角成分（ゲージ自由度）は 0 とする。

    Parameters
    ----------
    Delta       : (B, B) complex ndarray — 局所密度行列（エルミート）
    K           : (B, B) complex ndarray — 右辺行列
    is_frozen_D : bool
        True  → ローレンツ型スムージング ε=1e-4（縮退固有値に頑健）
        False → ε=1e-12（Full TD-gGA: 縮退が解消されている想定）

    Returns
    -------
    Lmbdac : (B, B) complex ndarray
    """
    eigenvalues, U = LA.eigh(Delta)
    K_tilde = U.conj().T @ K @ U

    B_loc = len(eigenvalues)
    Lmbdac_tilde = np.zeros((B_loc, B_loc), dtype=complex)
    epsilon = 1e-4 if is_frozen_D else 1e-12
    for i in range(B_loc):
        for j in range(B_loc):
            if i != j:
                diff = eigenvalues[j] - eigenvalues[i]
                Lmbdac_tilde[i, j] = K_tilde[i, j] * diff / (diff**2 + epsilon**2)

    Lmbdac = U @ Lmbdac_tilde @ U.conj().T
    return (Lmbdac + Lmbdac.conj().T) / 2.0


# ==========================================
# 3. ODE時間発展のコアクラス
# ==========================================
class TDgGADynamics:
    def __init__(self, dim_Phi, B, H_emb_0_fock, H_phys_1body,
                 is_frozen_D=True, D_gf=None, omega_k=None, weights_k=None):
        """
        物理パラメータを保持するクラス。

        Parameters
        ----------
        dim_Phi       : int — インプリシットフォック空間の次元
        B             : int — バス軌道数
        H_emb_0_fock  : (dim_Phi, dim_Phi) complex — 埋め込みハミルトニアン基底部
            Frozen-D: D 項を組み込み済みのまま渡す。
            Full TD-gGA: 同じく D 項込みで渡す（初期化時に D 項を分離する）。
        H_phys_1body  : 準粒子分散を表す1体ハミルトニアン
        is_frozen_D   : bool — True: Frozen-D 近似 / False: Full TD-gGA
        D_gf          : (B, C) complex or None — 平衡ハイブリダイゼーション行列
            Full TD-gGA の初期化に使用。Frozen-D では参照しない。
        omega_k       : (N_omega,) float or None — 周波数メッシュ
        weights_k     : (N_omega,) float or None — 積分重み (ρ(ω)dω)

        属性: インスタンス生成後に設定し、finalize_setup() を呼ぶこと。
            system.op_cb = (B, C, dim_Phi, dim_Phi)  c†_α b_a
            system.op_bb = (B, B, dim_Phi, dim_Phi)  b†_b b_a
        """
        self.dim_Phi     = dim_Phi
        self.B           = B
        self.H_emb_0     = H_emb_0_fock
        self.H_phys      = H_phys_1body
        self.is_frozen_D = is_frozen_D
        self.D_gf        = D_gf
        self.omega_k     = omega_k
        self.weights_k   = weights_k
        self.N_omega     = len(omega_k) if omega_k is not None else 0
        self.H_emb_0_no_D = None  # finalize_setup() で設定

    def finalize_setup(self):
        """
        op_cb/op_bb/op_D 設定後に呼ぶ初期化。
        Full TD-gGA モードでは H_emb_0 から D 項を取り除いた基底部を計算する。
        op_D[a] = (c†_up b_{a,up} + h.c.) + (c†_dn b_{a,dn} + h.c.)  — 既にエルミート。
        Frozen-D モードでは何もしない。
        """
        if self.is_frozen_D or self.D_gf is None:
            return
        # H_emb_0_no_D = H_emb_0 - Σ_a D_gf[a,0] * op_D[a]
        H_D_eq = np.einsum('a,aij->ij', self.D_gf[:, 0], self.op_D)
        self.H_emb_0_no_D = self.H_emb_0 - H_D_eq

    def _compute_D_from_nk(self, Delta, R, n_kw):
        """
        Eq.(18): D = [Δ(1-Δ)]^{-1/2} Q
                 Q_{b,α} = Σ_c (∫dω ρ(ω) ω n_{bc}(ω)) R_{c,α}^*

        Parameters
        ----------
        Delta : (B, B)        — 局所密度行列
        R     : (B, C)        — Gutzwiller 繰り込み行列
        n_kw  : (N_omega,B,B) — 準粒子占有数行列

        Returns
        -------
        D_new : (B, C) complex
        """
        # ∫dω ρ(ω) ω n_{bc}(ω) → 重み付き和 (B, B)
        n_weighted = np.einsum('k,kbc->bc', self.weights_k * self.omega_k, n_kw)
        # Q_{b,α} = Σ_c n_weighted_{b,c} R*_{c,α}  → (B, C)
        Q = n_weighted @ R.conj()
        # D = [Δ(1-Δ)]^{-1/2} Q
        eigs_D, U_D = LA.eigh(Delta)
        eigs_D = np.clip(np.real(eigs_D), 1e-4, 1.0 - 1e-4)
        inv_sqrt = 1.0 / np.sqrt(eigs_D * (1.0 - eigs_D))
        return (U_D * inv_sqrt[np.newaxis, :]) @ U_D.conj().T @ Q

    def _rebuild_H_emb_with_D(self, D_new):
        """
        Full TD-gGA 専用: 更新された D から H_emb_current を構成する。
            H_emb_current = H_emb_0_no_D + Σ_a D_new[a,0] * op_D[a]
        op_D[a] は既にエルミート (spin-up + spin-dn + h.c. を含む)。
        """
        H_D = np.einsum('a,aij->ij', D_new[:, 0], self.op_D)
        return self.H_emb_0_no_D + H_D

    def compute_derivatives(self, t, Y_flat):
        """
        ODE の右辺。

        Frozen-D モード: Y = [Φ.real, Φ.imag]
        Full TD-gGA モード: Y = [Φ.real, Φ.imag, n_kw.real.flat, n_kw.imag.flat]
        """
        B = self.B

        # --- 1. アンパック ---
        if self.is_frozen_D:
            Phi = Y_flat[:self.dim_Phi] + 1j * Y_flat[self.dim_Phi:]
            n_kw = None
        else:
            Phi, n_kw = unpack_state(Y_flat, self.dim_Phi, B=B, N_omega=self.N_omega)

        # --- 2. n_ab を Phi から直接計算（拘束条件 n_ab = <Φ|b†b|Φ> を厳密維持）---
        n_ab = np.array([[Phi.conj() @ self.op_bb[a, b] @ Phi
                          for b in range(B)] for a in range(B)])

        # --- 3. Delta のエルミート化 ---
        Delta = (n_ab + n_ab.conj().T) / 2.0

        # --- 4. R の計算 ---
        fdaggerc = np.array([[Phi.conj() @ self.op_cb[a, 0] @ Phi] for a in range(B)])
        eigs_D, U_D = LA.eigh(Delta)
        eigs_D = np.clip(np.real(eigs_D), 1e-4, 1.0 - 1e-4)
        inv_sqrt_eigs = 1.0 / np.sqrt(eigs_D * (1.0 - eigs_D))
        R = (U_D * inv_sqrt_eigs[np.newaxis, :]) @ U_D.conj().T @ fdaggerc  # (B, 1)

        # --- 5. H_emb 基底部の決定 ---
        if self.is_frozen_D:
            H_emb_current = np.array(self.H_emb_0, dtype=complex)
        else:
            # Full TD-gGA: Eq.(18) で D を代数的に更新
            D_new = self._compute_D_from_nk(Delta, R, n_kw)
            H_emb_current = self._rebuild_H_emb_with_D(D_new)

        # --- 6. h_qp^0 = R H_phys R† （Λ^c = 0 の仮計算）---
        h_qp_0 = R @ self.H_phys @ R.conj().T

        # --- 7. K 行列（拘束条件の整合性条件）---
        comm_n_hqp = n_ab @ h_qp_0 - h_qp_0 @ n_ab
        K = np.zeros((B, B), dtype=complex)
        for a in range(B):
            for b in range(B):
                comm_op = self.op_bb[a, b] @ H_emb_current - H_emb_current @ self.op_bb[a, b]
                K[a, b] = Phi.conj() @ comm_op @ Phi
        K -= comm_n_hqp

        # --- 8. Λ^c を決定 ---
        Lmbdac = compute_Lambda_c(Delta, K, is_frozen_D=self.is_frozen_D)

        # --- 9. H_emb = H_emb_current + Σ Λ^c_{ab} op_bb[a,b] ---
        H_emb = H_emb_current.copy()
        for a in range(B):
            for b in range(B):
                if abs(Lmbdac[a, b]) > 1e-14:
                    H_emb += Lmbdac[a, b] * self.op_bb[a, b]
        H_emb = (H_emb + H_emb.conj().T) / 2.0

        # --- 10. dΦ/dt = -i H_emb Φ ---
        dPhi_dt = -1j * (H_emb @ Phi)

        if self.is_frozen_D:
            return pack_state(dPhi_dt)

        # --- 11. Full TD-gGA: Eq.(17) ---
        # dn_{ab}/dt = -iω Σ_c (R_b R_c† n_{ac} - R_c R_a† n_{cb})
        # n_kw は n(ω)^T を格納 (Eq.11 の転置規約)
        # 行列形式: dn_kw_dt[k] = -iω_k [RR†, n_kw[k]]  (転置規約での等価形)
        RRdag = R @ R.conj().T  # (B, B)
        dn_kw_dt = np.array([
            -1j * self.omega_k[idx] * (RRdag @ n_kw[idx] - n_kw[idx] @ RRdag)
            for idx in range(self.N_omega)
        ])

        return pack_state(dPhi_dt, dn_kw_dt)


# ==========================================
# 4. メイン実行関数
# ==========================================
def run_quench_dynamics(Phi_0, n_ab_0, t_max, dt, physics_params, rtol=1e-8, atol=1e-10):
    """
    時間発展を実行するメイン関数。

    physics_params の必須キー:
        dim_Phi, B, H_emb_0_fock, H_phys_1body, op_cb, op_bb

    physics_params の追加キー (Full TD-gGA):
        is_frozen_D : bool (default True)
        D_gf        : (B, C) ndarray or None
        op_D        : (B, dim_Phi, dim_Phi) — D 項演算子 (spin-up+dn+h.c.)
        omega_k     : (N_omega,) ndarray or None
        weights_k   : (N_omega,) ndarray or None
        n_kw_0      : (N_omega, B, B) ndarray or None — 初期 n(ω_k)^T
    """
    dim_Phi      = physics_params['dim_Phi']
    B            = physics_params['B']
    H_emb_0_fock = physics_params['H_emb_0_fock']
    H_phys_1body = physics_params['H_phys_1body']
    op_cb        = physics_params['op_cb']
    op_bb        = physics_params['op_bb']
    is_frozen_D  = physics_params.get('is_frozen_D', True)
    D_gf         = physics_params.get('D_gf', None)
    op_D         = physics_params.get('op_D', None)
    omega_k      = physics_params.get('omega_k', None)
    weights_k    = physics_params.get('weights_k', None)
    n_kw_0       = physics_params.get('n_kw_0', None)

    system = TDgGADynamics(dim_Phi, B, H_emb_0_fock, H_phys_1body,
                           is_frozen_D=is_frozen_D, D_gf=D_gf,
                           omega_k=omega_k, weights_k=weights_k)
    system.op_cb = op_cb
    system.op_bb = op_bb
    if op_D is not None:
        system.op_D = op_D
    system.finalize_setup()  # Full TD-gGA: H_emb_0_no_D を計算

    # 初期状態のパック (Frozen-D では n_kw_0 を含めない)
    Y0_flat = pack_state(Phi_0, n_kw_0 if not is_frozen_D else None)

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
