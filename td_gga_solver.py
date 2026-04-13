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
                diff = eigenvalues[j] - eigenvalues[i]
                # +1e-12 の代わりに閾値でゼロ除算を回避:
                # diff の符号が保たれ、Λ^c のエルミート性が壊れない
                if abs(diff) > 1e-10:
                    Lmbdac_tilde[i, j] = K_tilde[i, j] / diff
                else:
                    Lmbdac_tilde[i, j] = 0.0
            # i == j : ゲージ自由度のため 0（デフォルト値のまま）

    Lmbdac = U @ Lmbdac_tilde @ U.conj().T
    # 数値誤差による非エルミート化を防ぐ強制エルミート化
    return (Lmbdac + Lmbdac.conj().T) / 2.0

# ==========================================
# 3. ODE時間発展のコアクラス
# ==========================================
class TDgGADynamics:
    def __init__(self, dim_Phi, B, H_emb_0_fock, H_phys_1body):
        """
        物理パラメータを保持するクラス。
        D は H_emb_0_fock に事前に組み込み済み (Frozen-D 近似)。

        Parameters
        ----------
        H_emb_0_fock : (dim_Phi, dim_Phi) ndarray
            Λ^c を除いたすべての項を含む多体ハミルトニアン（Fock 空間）。
            物理 on-site (-U/2)、D ブロック、Hubbard U を含む。
        H_phys_1body : (B, B) ndarray
            h_qp の 0 次項 R† H_phys R を計算するための 1 体物理ブロック。

        前提: インスタンス生成後に以下の属性を設定しておくこと。
              system.op_cb = op_cb  — (B,B,dim_Phi,dim_Phi) c†_α b_a
              system.op_bb = op_bb  — (B,B,dim_Phi,dim_Phi) b†_b b_a
        """
        self.dim_Phi  = dim_Phi
        self.B        = B
        self.H_emb_0  = H_emb_0_fock   # (dim_Phi, dim_Phi): 固定ハミルトニアン
        self.H_phys   = H_phys_1body    # (B, B): h_qp 用物理ブロック

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

        # c. H_emb^0 は run_simulation 側で事前構築済み（D・Hubbard U を含む）
        #    self.H_emb_0 をそのまま使用。D を毎ステップ足す処理は不要。

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
        h_qp_0 = R.conj().T @ self.H_phys @ R   # Λ^c = 0

        # K 行列の計算
        # self.H_emb_0 は (dim_Phi, dim_Phi) のフォック空間行列
        # self.op_bb[a,b] も (dim_Phi, dim_Phi) のため、交換子がフォック空間で正しく計算される
        comm_n_hqp = n_ab @ h_qp_0.T - h_qp_0.T @ n_ab
        K = np.zeros((B, B), dtype=complex)
        for a in range(B):
            for b in range(B):
                comm_op = self.op_bb[a, b] @ self.H_emb_0 - self.H_emb_0 @ self.op_bb[a, b]
                K[a, b] = Phi.conj() @ comm_op @ Phi
        K -= comm_n_hqp

        # Step 2: K を使って Λ^c を決定
        Lmbdac = compute_Lambda_c(Delta, K)

        # H_emb = H_emb^0 + Σ_{ab} Λ^c_{ab} b†_a b_b（フォック空間での正しい加算）
        # TD-gGA の変分原理: Λ^c は ⟨Φ|b†_b b_a|Φ⟩ = Δ_{ab} の拘束力
        # → 浴/ゴースト軌道ブロックにかかる演算子であり、op_bb で加算する
        H_emb = np.array(self.H_emb_0, dtype=complex)
        for a in range(B):
            for b in range(B):
                if abs(Lmbdac[a, b]) > 1e-14:
                    H_emb += Lmbdac[a, b] * self.op_bb[a, b]

        # 準粒子ハミルトニアン h_qp を構築 (B × B)
        #   h_qp = R† H_phys R + Λ^c
        h_qp = h_qp_0 + Lmbdac

        # 時間発展の直前に強制エルミート化（非エルミート成分が混入するとノルムが発散する）
        H_emb = (H_emb + H_emb.conj().T) / 2.0
        h_qp  = (h_qp  + h_qp.conj().T)  / 2.0

        # シュレーディンガー方程式: i ∂_t |Φ⟩ = H_emb |Φ⟩
        # H_emb は (dim_Phi, dim_Phi)、Phi は (dim_Phi,) — 次元一致
        dPhi_dt = -1j * (H_emb @ Phi)

        # von Neumann 方程式: i ∂_t n = [n, h_qp^T]
        #   → ∂_t n = -i (n h_qp^T - h_qp^T n)
        dn_ab_dt = -1j * (n_ab @ h_qp.T - h_qp.T @ n_ab)

        return pack_state(dPhi_dt, dn_ab_dt)

# ==========================================
# 4. メイン実行関数
# ==========================================
def run_quench_dynamics(Phi_0, n_ab_0, t_max, dt, physics_params, rtol=1e-8, atol=1e-10):
    """
    時間発展を実行するメイン関数。

    physics_params に必要なキー:
      dim_Phi, B, H_emb_0_fock, H_phys_1body, op_cb, op_bb
    """
    dim_Phi      = physics_params['dim_Phi']
    B            = physics_params['B']
    H_emb_0_fock = physics_params['H_emb_0_fock']
    H_phys_1body = physics_params['H_phys_1body']
    op_cb        = physics_params['op_cb']
    op_bb        = physics_params['op_bb']

    # TDgGADynamics インスタンスを生成し、フォック空間演算子行列を属性として設定
    system = TDgGADynamics(dim_Phi, B, H_emb_0_fock, H_phys_1body)
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
        rtol=rtol,
        atol=atol,
    )

    return sol

if __name__ == "__main__":
    # ここにダミーデータを与えてテスト実行するコードを書く
    pass
