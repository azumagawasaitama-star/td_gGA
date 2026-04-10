import numpy as np
from scipy import integrate, linalg as LA

# ==========================================
# 1. パッキング・アンパック用ヘルパー関数
# ==========================================
def pack_state(Phi, n_ab):
    """
    状態ベクトルPhi(複素1次元配列)と密度行列n_ab(複素2次元配列)を、
    ODEソルバー用の1次元実数配列(Y_flat)にパッキングする。
    - Phiの実部、Phiの虚部、n_abの実部、n_abの虚部をnp.concatenateで1次元に繋ぐ。
    """
    return np.concatenate([
        Phi.real,
        Phi.imag,
        n_ab.real.ravel(),
        n_ab.imag.ravel(),
    ])

def unpack_state(Y_flat, dim_Phi, B):
    """
    1次元実数配列(Y_flat)から、Phi(複素1次元配列)とn_ab(複素2次元配列)を復元する。
    - Y_flatをスライスして実部と虚部を取り出し、Phiとn_ab(サイズB x B)を再構築する。
    """
    i0 = 0
    i1 = dim_Phi
    i2 = 2 * dim_Phi
    i3 = 2 * dim_Phi + B * B
    i4 = 2 * dim_Phi + 2 * B * B

    Phi  = Y_flat[i0:i1] + 1j * Y_flat[i1:i2]
    n_ab = (Y_flat[i2:i3] + 1j * Y_flat[i3:i4]).reshape(B, B)

    return Phi, n_ab

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
    for i in range(B):
        for j in range(B):
            if i != j:
                # [Delta, Lambda^c]_{ij} = (lam_j - lam_i) * Lambda^c_tilde_{ij} = K_tilde_{ij}
                Lmbdac_tilde[i, j] = K_tilde[i, j] / (eigenvalues[j] - eigenvalues[i] + 1e-12)
            # i == j : ゲージ自由度のため 0（デフォルト値のまま）

    # 元の基底に戻す
    return U @ Lmbdac_tilde @ U.conj().T

# ==========================================
# 3. ODE時間発展のコアクラス
# ==========================================
class TDgGADynamics:
    def __init__(self, dim_Phi, B, H_loc, D_guess_0):
        """
        物理パラメータを保持するクラス。
        Frozen-D 近似のため D_0 のみを保持する。

        前提: インスタンス生成後に以下の属性を設定しておくこと。
              system.op_cb = op_cb  — フォック空間演算子行列 c†_α b_a (B,B,dim_Phi,dim_Phi)
              system.op_bb = op_bb  — フォック空間演算子行列 b†_b b_a  (B,B,dim_Phi,dim_Phi)
                                      現在は K=0 仮置きのため未使用だが、
                                      K 行列の厳密実装時（TODO参照）に必須となる。
        """
        self.dim_Phi = dim_Phi
        self.B = B
        self.H_loc = H_loc
        self.D_0 = D_guess_0  # Frozen-D 近似: D を時間発展させない

    def compute_derivatives(self, t, Y_flat):
        """
        scipy.integrate.solve_ivp から各時間ステップで呼び出される関数。
        dY/dt を反復ソルバーなしで直接計算して返す。
        """
        B = self.B
        I = np.eye(B)

        # 1. 1次元実数配列 Y_flat → 複素 Phi, n_ab
        Phi, n_ab = unpack_state(Y_flat, self.dim_Phi, B)

        # a. 数値安定化: 局所密度行列 Δ を明示的にエルミート化
        Delta = (n_ab + n_ab.conj().T) / 2.0

        # b. R の直接計算
        #    <Φ|c†_α b_a|Φ> = [Δ(1-Δ)]^{1/2} R
        #    → R = pinv([Δ(1-Δ)]^{1/2}) @ fdaggerc
        fdaggerc = np.array([[Phi.conj() @ self.op_cb[a, al] @ Phi
                              for al in range(B)] for a in range(B)])
        sqrt_D1mD = LA.sqrtm(Delta @ (I - Delta))
        R = LA.pinv(sqrt_D1mD) @ fdaggerc

        # c. Frozen-D 近似: D を初期値に固定
        D = self.D_0

        # H_emb の Λ^c を含まない部分（H_emb^0）を構築
        #   H_emb^0 = H_loc + D ブロックのみ（Λ^c は後から加算）
        H_emb_0 = np.array(self.H_loc, dtype=complex)
        H_emb_0[:B, B:]  += D
        H_emb_0[B:, :B]  += D.conj().T

        H_phys = self.H_loc[:B, :B]

        # d. Λ^c の計算（シルベスター方程式 [Λ^c, n] = K）
        #
        # K の循環依存を解決するため 2 段階で計算する:
        #   Step 1: Λ^c = 0 として h_qp を仮計算 → K を計算
        #   Step 2: K を使って最終的な Λ^c を決定
        #
        # K の導出:
        #   制約条件 d/dt⟨b†b⟩_{ab} = dn_{ab}/dt を Heisenberg 方程式で展開すると
        #   [Λ^c, n]_{ab} = ⟨Φ|[op_bb[a,b], H_emb^0]|Φ⟩ − [n, h_qp^T]_{ab}
        #   すなわち K[a,b] = ⟨Φ|[op_bb[a,b], H_emb^0]|Φ⟩ − (n @ h_qp^T − h_qp^T @ n)[a,b]

        # Step 1: Λ^c = 0 で h_qp を仮計算
        h_qp_0 = R.conj().T @ H_phys @ R   # Λ^c = 0

        # K 行列の計算
        comm_n_hqp = n_ab @ h_qp_0.T - h_qp_0.T @ n_ab
        K = np.zeros((B, B), dtype=complex)
        for a in range(B):
            for b in range(B):
                comm_op = self.op_bb[a, b] @ H_emb_0 - H_emb_0 @ self.op_bb[a, b]
                K[a, b] = Phi.conj() @ comm_op @ Phi
        K -= comm_n_hqp

        # Step 2: K を使って Λ^c を決定
        Lmbdac = compute_Lambda_c(Delta, K)

        # 最終的な H_emb と h_qp を構築
        H_emb = np.array(H_emb_0)
        H_emb[:B, :B] += Lmbdac

        # 準粒子ハミルトニアン h_qp を構築 (B × B)
        #   h_qp = R† H_phys R + Λ^c
        h_qp = h_qp_0 + Lmbdac

        # シュレーディンガー方程式: i ∂_t |Φ⟩ = H_emb |Φ⟩
        dPhi_dt = -1j * (H_emb @ Phi)

        # von Neumann 方程式: i ∂_t n = [n, h_qp^T]
        #   → ∂_t n = -i (n h_qp^T - h_qp^T n)
        dn_ab_dt = -1j * (n_ab @ h_qp.T - h_qp.T @ n_ab)

        return pack_state(dPhi_dt, dn_ab_dt)

# ==========================================
# 4. メイン実行関数
# ==========================================
def run_quench_dynamics(Phi_0, n_ab_0, t_max, dt, physics_params):
    """
    時間発展を実行するメイン関数。

    physics_params に必要なキー:
      dim_Phi, B, H_loc_final, D_guess_0, op_cb
    """
    dim_Phi     = physics_params['dim_Phi']
    B           = physics_params['B']
    H_loc_final = physics_params['H_loc_final']
    D_guess_0   = physics_params['D_guess_0']
    op_cb       = physics_params['op_cb']
    op_bb       = physics_params['op_bb']   # K 行列実装時（TODO）に使用

    # TDgGADynamics インスタンスを生成し、フォック空間演算子行列を属性として設定
    system = TDgGADynamics(dim_Phi, B, H_loc_final, D_guess_0)
    system.op_cb = op_cb
    system.op_bb = op_bb

    # 初期状態を1次元実数配列にパック
    Y0_flat = pack_state(Phi_0, n_ab_0)

    # 評価時刻の配列を作成
    t_eval = np.arange(0, t_max, dt)

    # ODE ソルバーで時間発展を計算
    sol = integrate.solve_ivp(
        fun=system.compute_derivatives,
        t_span=(0, t_max),
        y0=Y0_flat,
        method='RK45',
        t_eval=t_eval,
    )

    return sol

if __name__ == "__main__":
    # ここにダミーデータを与えてテスト実行するコードを書く
    pass
