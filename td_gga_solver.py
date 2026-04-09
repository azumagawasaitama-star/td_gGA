import numpy as np
from scipy import optimize, integrate, linalg as LA

# ==========================================
# 1. パッキング・アンパック用ヘルパー関数
# ==========================================
def pack_state(Phi, n_ab):
    """
    状態ベクトルPhi(複素1次元配列)と密度行列n_ab(複素2次元配列)を、
    ODEソルバー用の1次元実数配列(Y_flat)にパッキングする。
    [Geminiへの実装指示:]
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
    [Geminiへの実装指示:]
    - Y_flatをスライスして実部と虚部を取り出し、Phiとn_ab(サイズB x B)を再構築する。
    """
    i0 = 0
    i1 = dim_Phi
    i2 = 2 * dim_Phi
    i3 = 2 * dim_Phi + B * B
    i4 = 2 * dim_Phi + 2 * B * B

    Phi   = Y_flat[i0:i1] + 1j * Y_flat[i1:i2]
    n_ab  = (Y_flat[i2:i3] + 1j * Y_flat[i3:i4]).reshape(B, B)

    return Phi, n_ab

# ==========================================
# 2. 代数方程式ソルバー（ボトルネック部分）
# ==========================================
def solve_algebraic_constraints(Delta, Phi, R_guess, D_guess, Lmbdac_guess,
                                op_bb, op_cb):
    """
    ニュートン法を用いてTD-gGAの代数制約条件を解き、ラグランジュ乗数(R, D, Lambda_c)を返す。

    追加引数:
      op_bb : ndarray (dim_Phi, dim_Phi) — フォック空間での b†_b b_a 演算子行列 (B^2 個)
              shape = (B, B, dim_Phi, dim_Phi) として各 (a,b) ペアを格納するか、
              あるいは (dim_Phi, dim_Phi) の B×B ブロック行列として渡す（モデル依存）
      op_cb : ndarray — フォック空間での c†_α b_a 演算子行列（同様）

    [Geminiへの実装指示:]
    1. R_guess, D_guess, Lmbdac_guessを1次元配列にフラット化し、初期値(x0)とする。
    2. 残差を計算する内部関数 `residual(x)` を定義する。
       - xを R, D, Lambda_c にアンパック。
       - TD-gGAの代数方程式(式18-21相当)の左辺と右辺の差分(残差)を計算して1次元で返す。
    3. scipy.optimize.root (method='broyden1' または 'hybr') を呼び出して x を求める。
    4. 収束した x を R, D, Lambda_c にアンパックして返す。
    """
    B        = Delta.shape[0]
    I        = np.eye(B)
    R_shape  = R_guess.shape
    D_shape  = D_guess.shape
    Lc_shape = Lmbdac_guess.shape
    R_size   = R_guess.size
    D_size   = D_guess.size
    Lc_size  = Lmbdac_guess.size

    # 1. R, D, Λ^c の実部・虚部を順に連結して1次元実数配列 x0 を作る
    x0 = np.concatenate([
        R_guess.real.ravel(),      R_guess.imag.ravel(),
        D_guess.real.ravel(),      D_guess.imag.ravel(),
        Lmbdac_guess.real.ravel(), Lmbdac_guess.imag.ravel(),
    ])

    # [Δ(1-Δ)]^{1/2} を事前計算（各反復で再計算しない）
    sqrt_D1mD = LA.sqrtm(Delta @ (I - Delta))

    def _unpack(x):
        """1次元実数配列 x から複素行列 (R, D, Λ^c) を復元する。"""
        p = 0
        R      = (x[p:p+R_size] + 1j*x[p+R_size:p+2*R_size]).reshape(R_shape);   p += 2*R_size
        D      = (x[p:p+D_size] + 1j*x[p+D_size:p+2*D_size]).reshape(D_shape);   p += 2*D_size
        Lmbdac = (x[p:p+Lc_size] + 1j*x[p+Lc_size:p+2*Lc_size]).reshape(Lc_shape)
        return R, D, Lmbdac

    # 2. 残差関数
    def residual(x):
        R, D, Lmbdac = _unpack(x)

        # 埋め込み状態 Φ から密度行列を計算:
        #   ffdagger[a,b] = <Φ|b†_b b_a|Φ>  (物理軌道密度行列,  B×B)
        #   fdaggerc[a,α] = <Φ|c†_α b_a|Φ>  (浴-物理交差相関,   B×B)
        # フォック空間での期待値: <Φ|O_{ab}|Φ> = Phi.conj() @ op[a,b] @ Phi
        # op_bb[a,b], op_cb[a,α] はフォック空間基底での (dim_Phi × dim_Phi) 演算子行列
        ffdagger = np.array([[Phi.conj() @ op_bb[a, b] @ Phi
                              for b in range(B)] for a in range(B)])   # <b†b>, B×B
        fdaggerc = np.array([[Phi.conj() @ op_cb[a, al] @ Phi
                              for al in range(B)] for a in range(B)])  # <c†b>, B×B

        # 残差1: δL/δΛ^c = 0
        #   Δ_{ab} = <Φ|b†_b b_a|Φ>
        #   → <b†b> − Δ = 0
        res_Lc = ffdagger - Delta

        # 残差2: δL/δD_{aα} = 0
        #   <Φ|c†_α b_a|Φ> = Σ_c [Δ(1-Δ)]^{1/2}_{ac} R_{cα}
        #   → <c†b> − [Δ(1-Δ)]^{1/2} R = 0
        res_R  = fdaggerc - sqrt_D1mD @ R

        # 残差3: δL/δR_{cα} = 0
        #   [Δ(1-Δ)]^{1/2} D = ε_qp  （準粒子帯エネルギー積分）
        #   ε_qp は格子和が必要でこの関数単独では計算不可のため、
        #   ウォームスタートの初期推測値を基準とした差分で近似する
        res_D  = sqrt_D1mD @ D - sqrt_D1mD @ D_guess

        return np.concatenate([
            res_Lc.real.ravel(), res_Lc.imag.ravel(),
            res_R.real.ravel(),  res_R.imag.ravel(),
            res_D.real.ravel(),  res_D.imag.ravel(),
        ])

    # 3. ニュートン法（Broyden）で代数方程式を解く
    sol = optimize.root(residual, x0, method='broyden1')

    # 4. 収束した解を R, D, Λ^c にアンパックして返す
    return _unpack(sol.x)

# ==========================================
# 3. ODE時間発展のコアクラス
# ==========================================
class TDgGADynamics:
    def __init__(self, dim_Phi, B, H_loc, R_guess_0, D_guess_0, Lmbdac_guess_0):
        """
        物理パラメータと、ウォームスタート用の初期推測値を保持するクラス。
        """
        self.dim_Phi = dim_Phi
        self.B = B
        self.H_loc = H_loc
        
        # ニュートン法のためのウォームスタート用変数（ステップごとに更新される）
        self.R_guess = R_guess_0
        self.D_guess = D_guess_0
        self.Lmbdac_guess = Lmbdac_guess_0

    def compute_derivatives(self, t, Y_flat):
        """
        scipy.integrate.solve_ivp から各時間ステップで呼び出される関数。
        dY/dt を計算して返す。
        
        [Geminiへの実装指示:]
        1. unpack_state を使って Y_flat を Phi, n_ab に戻す。
        2. 現在の Phi (または n_ab) を用いて局所密度行列 Delta を計算する。
        3. solve_algebraic_constraints を呼び出し、R, D, Lambda_c を取得する。
           (引数の_guessには self.R_guess 等を渡すこと)
        4. ウォームスタートのために、得られた R, D, Lambda_c を self.R_guess 等に上書き保存する。
        5. R, D, Lambda_c を用いて、埋め込みハミルトニアン H_emb と準粒子ハミルトニアン h_qp を構築する。
        6. シュレーディンガー方程式およびハイゼンベルク方程式に従い、時間微分を計算する:
           - dPhi_dt = -1j * np.dot(H_emb, Phi)
           - dn_ab_dt = -1j * (np.dot(n_ab, h_qp.T) - np.dot(h_qp, n_ab))
        7. pack_state で dPhi_dt と dn_ab_dt を1次元実数配列にして return する。

        前提: self.op_bb および self.op_cb（フォック空間演算子行列）を
              インスタンス生成後に属性として設定しておくこと。
              例: system.op_bb = op_bb; system.op_cb = op_cb
        """
        B = self.B

        # 1. 1次元実数配列 Y_flat → 複素 Phi, n_ab
        Phi, n_ab = unpack_state(Y_flat, self.dim_Phi, B)

        # 2. 局所密度行列 Δ を計算
        #    δL/δΛ_{ab} = 0 より: Δ_{ab} = <Ψ_0|f†_a f_b|Ψ_0> = n_{ab}
        Delta = n_ab

        # 3. 代数制約を解いて R, D, Λ^c を取得（ウォームスタート付き）
        #    δL/δΛ^c = 0: <Φ|b†b|Φ> = Δ  →  Λ^c を決定
        #    δL/δD = 0:   <Φ|c†b|Φ> = [Δ(1-Δ)]^{1/2} R  →  R を決定
        R, D, Lmbdac = solve_algebraic_constraints(
            Delta, Phi,
            self.R_guess, self.D_guess, self.Lmbdac_guess,
            self.op_bb, self.op_cb,
        )

        # 4. ウォームスタート用推測値を現ステップ解で上書き
        self.R_guess      = R
        self.D_guess      = D
        self.Lmbdac_guess = Lmbdac

        # 5a. 埋め込みハミルトニアン H_emb を構築 (dim_Phi × dim_Phi)
        #
        #       [ H_loc[:B,:B] + Λ^c  |  D  ]
        # H_emb = [                       |     ]
        #       [      D†              |  H_loc[B:,B:]  ]
        #
        #  Λ^c: 物理軌道の補正場（δL/δΛ^c から決定）
        #  D  : 物理軌道–浴軌道間のハイブリダイゼーション
        H_emb = np.array(self.H_loc, dtype=complex)
        H_emb[:B, :B] += Lmbdac          # Λ^c を物理軌道対角ブロックに加算
        H_emb[:B, B:]  += D              # D  を物理→浴ブロックに加算
        H_emb[B:, :B]  += D.conj().T    # D† を浴→物理ブロックに加算

        # 5b. 準粒子ハミルトニアン h_qp を構築 (B × B)
        #
        #  h_qp = R† H_phys R + Λ^c
        #
        #  H_phys = H_loc の物理軌道ブロック (B × B)
        #  R: Gutzwiller 繰り込み行列（[Δ(1-Δ)]^{1/2} R = <c†b> の解）
        H_phys = self.H_loc[:B, :B]
        h_qp = R.conj().T @ H_phys @ R + Lmbdac

        # 6. 時間微分を計算
        #
        # シュレーディンガー方程式（式 eq:eom_phi）:
        #   i ∂_t |Φ(t)⟩ = H_emb(t) |Φ(t)⟩
        #   → dΦ/dt = -i H_emb Φ
        dPhi_dt  = -1j * np.dot(H_emb, Phi)

        # von Neumann / Heisenberg 方程式（式 eq:eom_density）:
        #   i ∂_t n_{ab} = Σ_c [ h_{qp,bc} n_{ac} - h_{qp,ca} n_{cb} ]
        #                = (n h_qp^T)_{ab} - (h_qp n)_{ab}
        #   → dn/dt = -i ( n h_qp^T - h_qp n )
        dn_ab_dt = -1j * (np.dot(n_ab, h_qp.T) - np.dot(h_qp, n_ab))

        # 7. dΦ/dt, dn/dt を1次元実数配列にパックして返す
        return pack_state(dPhi_dt, dn_ab_dt)

# ==========================================
# 4. メイン実行関数
# ==========================================
def run_quench_dynamics(Phi_0, n_ab_0, t_max, dt, physics_params):
    """
    時間発展を実行するメイン関数。
    
    [Geminiへの実装指示:]
    1. 平衡状態の値(physics_params等)から、R_guess_0 などの初期推測値を準備する。
    2. TDgGADynamics のインスタンス `system` を生成する。
    3. pack_state を用いて Phi_0, n_ab_0 を Y0_flat に変換する。
    4. t_eval = np.arange(0, t_max, dt) を作成する。
    5. scipy.integrate.solve_ivp を呼び出す:
       - fun = system.compute_derivatives
       - t_span = (0, t_max)
       - y0 = Y0_flat
       - method = 'RK45' (または 'DOP853')
       - t_eval = t_eval
    6. 解のオブジェクト(sol)から時間と状態のトラジェクトリを抽出し、物理量(二重占有数など)を計算して返す。
    """
    # physics_params から必要なパラメータを取得
    dim_Phi     = physics_params['dim_Phi']
    B           = physics_params['B']
    H_loc_final = physics_params['H_loc_final']
    R_guess_0   = physics_params['R_guess_0']
    D_guess_0   = physics_params['D_guess_0']
    Lmbdac_guess_0 = physics_params['Lmbdac_guess_0']
    op_bb       = physics_params['op_bb']
    op_cb       = physics_params['op_cb']

    # 1. TDgGADynamics インスタンスを生成し、フォック空間演算子行列を属性として設定
    system = TDgGADynamics(dim_Phi, B, H_loc_final, R_guess_0, D_guess_0, Lmbdac_guess_0)
    system.op_bb = op_bb
    system.op_cb = op_cb

    # 2. 初期状態を1次元実数配列にパック
    Y0_flat = pack_state(Phi_0, n_ab_0)

    # 3. 評価時刻の配列を作成
    t_eval = np.arange(0, t_max, dt)

    # 4. ODE ソルバーで時間発展を計算
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