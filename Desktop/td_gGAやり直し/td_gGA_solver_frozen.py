import numpy as np
import scipy.sparse as sp
from math import comb
import convenience_routines as cr

# ==========================================
# Step 1-1: 状態のパック / アンパック
# ==========================================
def pack_state_frozen(Phi):
    """ 複素数ベクトル Phi を 1次元の実数配列 (実部, 虚部) に潰す """
    return np.concatenate((np.real(Phi), np.imag(Phi)))

def unpack_state_frozen(y, dim_Phi):
    """ 1次元実数配列から 複素数ベクトル Phi を復元する """
    return y[:dim_Phi] + 1j * y[dim_Phi:2*dim_Phi]


# ==========================================
# Step 1-2: 演算子とパラメータの準備
# ==========================================
def prepare_frozen_params(ga_obj):
    """ 
    静的計算が終わった ga_obj から、時間発展に必要な演算子と
    パラメータをすべて抽出し、一つの辞書にまとめて返す。
    （※フル空間で掛け算してからブロックを抽出することでゼロ行列を防ぐ）
    """
    ed_solver = ga_obj.imp_solver
    dim_Phi = ed_solver.hsize_half
    
    # 半分の粒子数のブロックを切り出すためのインデックス計算
    ioff = sum(int(comb(ed_solver.n_tot_orb, i)) for i in range(ed_solver.n_half))
    iend = ioff + dim_Phi
    
    # フル空間の生成・消滅演算子
    FH = ed_solver.build_creation_ops()
    c_up_dag = FH[0]
    c_dn_dag = FH[1]
    c_up = c_up_dag.conj().T
    c_dn = c_dn_dag.conj().T

    # td_gGA_solver_frozen.py : prepare_frozen_params の後半

    # --- 物理量の演算子（フル空間で作ってから切り出し） ---
    n_up_full = c_up_dag @ c_up
    n_dn_full = c_dn_dag @ c_dn
    
    op_docc = (n_up_full @ n_dn_full)[ioff:iend, ioff:iend].toarray()
    op_n_phys = (n_up_full + n_dn_full)[ioff:iend, ioff:iend].toarray()

    # ====== ここから追加 ======
    op_n_orb_list = []
    # 物理サイト(FH[0,1]) + 全バスサイト(FH[2,3], FH[4,5], FH[6,7]) = nqspo+1個
    for i in range(ga_obj.nqspo + 1):
        n_up = FH[2*i] @ FH[2*i].conj().T
        n_dn = FH[2*i+1] @ FH[2*i+1].conj().T
        op_n_i = (n_up + n_dn)[ioff:iend, ioff:iend].toarray()
        op_n_orb_list.append(op_n_i)
    # ====== ここまで追加 ======

    # Z(t) 計算用: ファイルから fdaggerc・ffdagger 演算子行列を読み込む
    n_phys_orb = ed_solver.n_phys_orb
    n_bath_orb = ed_solver.n_bath_orb
    imp_nr     = ed_solver.impurity_nr
    imp_type   = ed_solver.impurity_type

    op_fdaggerc_arr = np.zeros((n_bath_orb, n_phys_orb, dim_Phi, dim_Phi))
    for i in range(n_bath_orb):
        for j in range(n_phys_orb):
            fname = f"bath-phys_imp-{imp_nr}_{imp_type}_op+{i}-{j}.npz"
            op_fdaggerc_arr[i, j] = sp.load_npz(fname).toarray()

    op_ffdagger_arr = np.zeros((n_bath_orb, n_bath_orb, dim_Phi, dim_Phi))
    for i in range(n_bath_orb):
        for j in range(n_bath_orb):
            fname = f"bath-bath_imp-{imp_nr}_{imp_type}_op+{i}-{j}.npz"
            op_ffdagger_arr[i, j] = sp.load_npz(fname).toarray()

    # npzファイルとimp_solverのbath軌道順序の変換行列を計算
    # npzファイルはbath軌道を別の順序で保存しているため、
    # Z計算の前にR_tをimp_solver順序(=fix_gauge順序)に変換する必要がある
    nqspo = ga_obj.nqspo
    Phi_0_tmp = ed_solver.eig_vec.copy()
    Phi_conj_tmp = np.conj(Phi_0_tmp)

    # npz演算子からt=0のR_npzを計算
    fdaggerc_sp_0 = np.array([[0.5*(np.real(Phi_conj_tmp @ (op_fdaggerc_arr[2*i,2*j]   @ Phi_0_tmp)) +
                                    np.real(Phi_conj_tmp @ (op_fdaggerc_arr[2*i+1,2*j+1] @ Phi_0_tmp)))
                                for j in range(n_phys_orb // 2)]
                               for i in range(n_bath_orb // 2)])
    ffdagger_sp_0 = np.array([[0.5*(np.real(Phi_conj_tmp @ (op_ffdagger_arr[2*i,2*j]   @ Phi_0_tmp)) +
                                    np.real(Phi_conj_tmp @ (op_ffdagger_arr[2*i+1,2*j+1] @ Phi_0_tmp)))
                                for j in range(n_bath_orb // 2)]
                               for i in range(n_bath_orb // 2)])
    R_npz_0 = (fdaggerc_sp_0.T @ cr.funcMat(ffdagger_sp_0, cr.denR)).T  # (nqspo, 1)

    # imp_solverから直接R_impを計算 (imp_solver順序 = fix_gauge順序)
    R_imp_0 = (ed_solver.fdaggerc.T @ cr.funcMat(ed_solver.ffdagger, cr.denR)).T  # (nqspo, 1)

    # npz→imp 変換行列P: 大きい成分順にソートして対応付け
    idx_npz = np.argsort(-np.abs(R_npz_0[:, 0]))
    idx_imp = np.argsort(-np.abs(R_imp_0[:, 0]))
    P_npz2imp = np.zeros((nqspo, nqspo))
    for k in range(nqspo):
        P_npz2imp[idx_imp[k], idx_npz[k]] = 1.0  # npz[idx_npz[k]] → imp[idx_imp[k]]

    # --- 必要なものを辞書にパックして返す ---
    params = {
        'dim_Phi':          dim_Phi,
        'Phi_0':            ed_solver.eig_vec.copy(),
        'op_docc':          op_docc,
        'op_n_phys':        op_n_phys,
        'op_n_orb_list':    op_n_orb_list,
        'op_fdaggerc_arr':  op_fdaggerc_arr,
        'op_ffdagger_arr':  op_ffdagger_arr,
        'n_phys_orb':       n_phys_orb,
        'n_bath_orb':       n_bath_orb,
        'P_npz2imp':        P_npz2imp,
    }

    return params

# ==========================================
# Step 1-3: 波動関数から Z(t) を計算
# ==========================================
def compute_Z_from_Phi(Phi_t, params, ga_obj):
    """
    時刻 t の埋め込み波動関数 Phi_t から、
    瞬間的な密度行列を経由して準粒子重み Z(t) を返す。
    """
    op_fdaggerc_arr = params['op_fdaggerc_arr']  # (n_bath, n_phys, dim, dim)
    op_ffdagger_arr = params['op_ffdagger_arr']  # (n_bath, n_bath, dim, dim)
    n_phys_orb = params['n_phys_orb']
    n_bath_orb = params['n_bath_orb']
    Phi_conj = np.conj(Phi_t)

    # fdaggerc[i,j] = <f†_{bath,i} c_{phys,j}>
    fdaggerc_t = np.array([[np.real(Phi_conj @ (op_fdaggerc_arr[i, j] @ Phi_t))
                             for j in range(n_phys_orb)]
                            for i in range(n_bath_orb)])

    # ffdagger[i,j] = <f_{bath,j} f†_{bath,i}>  (lreverse=True で保存済み)
    ffdagger_t = np.array([[np.real(Phi_conj @ (op_ffdagger_arr[i, j] @ Phi_t))
                             for j in range(n_bath_orb)]
                            for i in range(n_bath_orb)])

    # スピン平均して空間行列に縮約 (スピン対称性を仮定)
    fdaggerc_sp = np.array([[0.5 * (fdaggerc_t[2*i, 2*j] + fdaggerc_t[2*i+1, 2*j+1])
                              for j in range(n_phys_orb // 2)]
                             for i in range(n_bath_orb // 2)])

    ffdagger_sp = np.array([[0.5 * (ffdagger_t[2*i, 2*j] + ffdagger_t[2*i+1, 2*j+1])
                              for j in range(n_bath_orb // 2)]
                             for i in range(n_bath_orb // 2)])

    R_t = (fdaggerc_sp.T @ cr.funcMat(ffdagger_sp, cr.denR)).T
    # npz軌道順序 → imp_solver/fix_gauge軌道順序に変換してからZ計算
    P = params.get('P_npz2imp')
    if P is not None:
        R_t = P @ R_t
    return ga_obj.calc_Z(ga_obj.Lmbda, R_t)


# ==========================================
# Step 2: H_emb の構築と時間発展エンジン
# ==========================================

def build_frozen_H_emb(ga_obj, U_final):
    """
    クエンチ後の U_final と、初期状態(Frozen)の D, Lambda^c を使って、
    時間発展用の定数ハミルトニアン H_emb_final を組み立てる。
    """
    ed_solver = ga_obj.imp_solver
    
    # 🌟修正ポイント：静的計算と同じ「ゲージ固定された」D と Lmbdac を取得
    D_fixed, Lmbdac_fixed = ga_obj.fix_gauge(ga_obj.D, ga_obj.Lmbdac, lfor_D=True)
    
    D_init = D_fixed
    # 🌟修正ポイント：np.diagを2回使うことで、「行列から対角成分を抽出(1D) → それを対角行列に復元(2D)」する
    Lc_init = np.diag(np.diag(Lmbdac_fixed)) 
    
    # 物理軌道の1体ハミルトニアン H1 (クエンチ後の U_final を反映)
    H1_final = np.array([[-U_final / 2.0]])
    
    # ed_solver の既存メソッドを再利用して H_emb を構築
    H_emb_final = ed_solver.build_Hemb(D_init, H1_final, Lc_init, U_final)
    
    return H_emb_final


def compute_derivatives_frozen(t, y, params):
    """
    微分方程式の計算エンジン (Frozen-D用)
    d|Phi>/dt = -i H_emb_final |Phi>
    """
    dim_Phi = params['dim_Phi']
    
    # 1. 1次元配列から複素数ベクトル Phi を復元
    Phi = unpack_state_frozen(y, dim_Phi)
    
    # 2. シュレーディンガー方程式の計算
    # Frozen-D では H_emb が定数なので、単なる行列とベクトルの掛け算です
    H_emb_final = params['H_emb_final']
    dPhi_dt = -1j * H_emb_final.dot(Phi)
    
    # 3. 再び1次元の実数配列に潰してソルバーに返す
    return pack_state_frozen(dPhi_dt)

# ==========================================
# Step 3: 実行と事後解析 (プロット用データの作成)
# ==========================================

def run_frozen_simulation(ga_obj, U_final, t_max, dt):
    # 1. 下準備 (Step 1 & 2 を利用)
    params = prepare_frozen_params(ga_obj)
    
    # 時間発展用の定数ハミルトニアンを準備して params に入れる
    params['H_emb_final'] = build_frozen_H_emb(ga_obj, U_final)
    
    # 2. 積分実行 (solve_ivp)
    y0 = pack_state_frozen(params['Phi_0'])
    t_eval = np.arange(0, t_max, dt)
    
    from scipy.integrate import solve_ivp
    sol = solve_ivp(
        compute_derivatives_frozen, 
        [0, t_max], 
        y0, 
        t_eval=t_eval, 
        args=(params,), 
        method='RK45', # 最初は安定した RK45 がおすすめ
        rtol=1e-8, atol=1e-10
    )
    
    # 3. 事後解析 (積分結果から物理量を取り出す)
    # ここで初めて「他のパラメータ(物理量)」を作ります
    dim_Phi = params['dim_Phi']
    op_docc = params['op_docc']
    op_n_orb_list = params['op_n_orb_list']
    n_orb_count = len(op_n_orb_list)  # nqspo + 1

    docc_list   = []
    E_tot_list  = []
    Z_list      = []
    n_orb_lists = [[] for _ in range(n_orb_count)]

    for k in range(len(sol.t)):
        # 状態の復元
        Phi_t = unpack_state_frozen(sol.y[:, k], dim_Phi)
        Phi_t /= np.linalg.norm(Phi_t) # 念のため規格化

        # 二重占有数 d(t) の計算
        d_val = np.real(np.dot(np.conj(Phi_t), op_docc @ Phi_t))
        docc_list.append(d_val)

        # 全エネルギー E_tot(t) の計算 (.dot() でsparse対応)
        H_Phi = params['H_emb_final'].dot(Phi_t)
        E_total = np.real(np.dot(np.conj(Phi_t), H_Phi))
        E_tot_list.append(E_total)

        # 準粒子重み Z(t) の計算
        Z_list.append(compute_Z_from_Phi(Phi_t, params, ga_obj))

        for i in range(n_orb_count):
            n_val = np.real(np.dot(np.conj(Phi_t), op_n_orb_list[i] @ Phi_t))
            n_orb_lists[i].append(n_val)

    # np.savez や算術演算のために numpy 配列に変換して返す
    results = {
        't':     sol.t,
        'docc':  np.array(docc_list),
        'E_tot': np.array(E_tot_list),
        'Z':     np.array(Z_list),
        'n_orb': np.array(n_orb_lists),  # shape: (nqspo+1, ntime)
    }
    return results