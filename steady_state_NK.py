import numpy as np
from numba import njit
import utils
from het_block import het
from simple_block import simple
from het_firm import firm, firm_Bertrand_ss, firm_exit
from matplotlib import pyplot as plt


'''Part 1: HA block'''


@het(exogenous='Pi', policy=['b', 'a'], backward=['Vb', 'Va'])  # order as in grid!
def household(Va_p, Vb_p, Pi_p, a_grid, b_grid, z_grid, e_grid, kk_grid, beta, eis, rb, ra, chi0, chi1, chi2):
    # get grid dimensions
    nZ, nB, nA = Va_p.shape
    nK = kk_grid.shape[0]

    # step 2: Wb(z, b', a') and Wa(z, b', a')
    Wb, Wa = post_decision_vfun(Va_p, Vb_p, Pi_p, beta)

    # step 3: a'(z, b', a) for UNCONSTRAINED
    lhs_unc = Wa / Wb
    Psi1 = Psi1_fun(a_grid[:, np.newaxis], a_grid[np.newaxis, :], ra, chi0, chi1, chi2)
    a_endo_unc, c_endo_unc = step3(lhs_unc, 1 + Psi1, Wb, a_grid, eis, nZ, nB, nA)

    # step 4: b'(z, b, a), a'(z, b, a) for UNCONSTRAINED
    b_unc, a_unc = step4(a_endo_unc, c_endo_unc, z_grid, b_grid, a_grid, ra, rb, chi0, chi1, chi2)

    # step 5: a'(z, kappa, a) for CONSTRAINED
    lhs_con = lhs_unc[:, 0, :]
    lhs_con = lhs_con[:, np.newaxis, :] / (1 + kk_grid[np.newaxis, :, np.newaxis])
    a_endo_con, c_endo_con = step5(lhs_con, 1 + Psi1, Wb, a_grid, kk_grid, eis, nZ, nK, nA)

    # step 6: a'(z, b, a) for CONSTRAINED
    a_con = step6(a_endo_con, c_endo_con, z_grid, b_grid, a_grid, ra, rb, chi0, chi1, chi2)

    # step 7a: put policy functions together
    a, b = a_unc.copy(), b_unc.copy()
    b[b <= b_grid[0]] = b_grid[0]
    a[b <= b_grid[0]] = a_con[b <= b_grid[0]]
    zzz = z_grid[:, np.newaxis, np.newaxis]
    bbb = b_grid[np.newaxis, :, np.newaxis]
    aaa = a_grid[np.newaxis, np.newaxis, :]
    c = zzz + (1 + ra) * aaa + (1 + rb) * bbb - Psi_fun(a, aaa, ra, chi0, chi1, chi2) - a - b
    uc = c ** (-1 / eis)
    u = e_grid[:, np.newaxis, np.newaxis] * uc

    # step 7b: update guesses
    Psi2 = Psi2_fun(a, aaa, ra, chi0, chi1, chi2)
    Va = (1 + ra - Psi2) * uc
    Vb = (1 + rb) * uc

    chi = Psi_fun(a, a_grid, ra, chi0, chi1, chi2)

    return Va, Vb, a, b, c, u, chi


def post_decision_vfun(Va_p, Vb_p, Pi, beta):
    Wb = (Vb_p.T @ (beta * Pi.T)).T
    Wa = (Va_p.T @ (beta * Pi.T)).T
    return Wb, Wa


def Psi_fun(ap, a, ra, chi0, chi1, chi2):
    a_with_return = (1 + ra) * a
    a_change = ap - a_with_return
    abs_a_change = np.abs(a_change)
    sign_change = np.sign(a_change)
    adj_denominator = a_with_return + chi0
    core_factor = (abs_a_change / adj_denominator) ** (chi2 - 1)
    Psi = chi1 / chi2 * abs_a_change * core_factor
    return Psi


def Psi1_fun(ap, a, ra, chi0, chi1, chi2):
    a_with_return = (1 + ra) * a
    a_change = ap - a_with_return
    abs_a_change = np.abs(a_change)
    sign_change = np.sign(a_change)
    adj_denominator = a_with_return + chi0
    core_factor = (abs_a_change / adj_denominator) ** (chi2 - 1)
    Psi1 = chi1 * sign_change * core_factor
    return Psi1


def Psi2_fun(ap, a, ra, chi0, chi1, chi2):
    a_with_return = (1 + ra) * a
    a_change = ap - a_with_return
    abs_a_change = np.abs(a_change)
    sign_change = np.sign(a_change)
    adj_denominator = a_with_return + chi0
    core_factor = (abs_a_change / adj_denominator) ** (chi2 - 1)
    Psi = chi1 / chi2 * abs_a_change * core_factor
    Psi1 = chi1 * sign_change * core_factor
    Psi2 = -(1 + ra) * (Psi1 + (chi2 - 1) * Psi / adj_denominator)
    return Psi2


@njit
def step3(lhs, rhs, Wb, a_grid, eis, nZ, nB, nA):
    ap_endo = np.empty((nZ, nB, nA))
    Wb_endo = np.empty((nZ, nB, nA))
    for iz in range(nZ):
        for ibp in range(nB):
            iap = 0  # use mononicity in a
            for ia in range(nA):
                while True:
                    if lhs[iz, ibp, iap] < rhs[iap, ia]:
                        break
                    elif iap < nA - 1:
                        iap += 1
                    else:
                        break
                if iap == 0:
                    ap_endo[iz, ibp, ia] = 0
                    Wb_endo[iz, ibp, ia] = Wb[iz, ibp, 0]
                elif iap == nA:
                    ap_endo[iz, ibp, ia] = a_grid[iap]
                    Wb_endo[iz, ibp, ia] = Wb[iz, ibp, iap]
                else:
                    y0 = lhs[iz, ibp, iap - 1] - rhs[iap - 1, ia]
                    y1 = lhs[iz, ibp, iap] - rhs[iap, ia]
                    ap_endo[iz, ibp, ia] = a_grid[iap - 1] - y0 * (a_grid[iap] - a_grid[iap - 1]) / (y1 - y0)
                    Wb_endo[iz, ibp, ia] = Wb[iz, ibp, iap - 1] + (
                                ap_endo[iz, ibp, ia] - a_grid[iap - 1]) * (
                                Wb[iz, ibp, iap] - Wb[iz, ibp, iap - 1]) / (a_grid[iap] - a_grid[iap - 1])
    c_endo = Wb_endo ** (-eis)
    return ap_endo, c_endo


def step4(ap_endo, c_endo, z_grid, b_grid, a_grid, ra, rb, chi0, chi1, chi2):
    # b(z, b', a)
    zzz = z_grid[:, np.newaxis, np.newaxis]
    bbb = b_grid[np.newaxis, :, np.newaxis]
    aaa = a_grid[np.newaxis, np.newaxis, :]
    b_endo = (c_endo + ap_endo + bbb - (1 + ra) * aaa + Psi_fun(ap_endo, aaa, ra, chi0, chi1, chi2) -
              zzz) / (1 + rb)

    # b'(z, b, a), a'(z, b, a)
    # assert np.min(np.diff(b_endo, axis=1)) > 0, 'b(bp) is not increasing'
    # assert np.min(np.diff(ap_endo, axis=1)) > 0, 'ap(bp) is not increasing'
    i, pi = utils.interpolate_coord(b_endo.swapaxes(1, 2), b_grid)
    ap = utils.apply_coord(i, pi, ap_endo.swapaxes(1, 2)).swapaxes(1, 2)
    bp = utils.apply_coord(i, pi, b_grid).swapaxes(1, 2)
    return bp, ap


@njit
def step5(lhs, rhs, Wb, a_grid, k_grid, eis, nZ, nK, nA):
    ap_endo = np.empty((nZ, nK, nA))
    Wb_endo = np.empty((nZ, nK, nA))
    for iz in range(nZ):
        for ik in range(nK):
            iap = 0  # use mononicity in a
            for ia in range(nA):
                while True:
                    if lhs[iz, ik, iap] < rhs[iap, ia]:
                        break
                    elif iap < nA - 1:
                        iap += 1
                    else:
                        break
                if iap == 0:
                    ap_endo[iz, ik, ia] = 0
                    Wb_endo[iz, ik, ia] = (1 + k_grid[ik]) * Wb[iz, 0, 0]
                elif iap == nA:
                    ap_endo[iz, ik, ia] = a_grid[iap]
                    Wb_endo[iz, ik, ia] = (1 + k_grid[ik]) * Wb[iz, 0, iap]
                else:
                    y0 = lhs[iz, ik, iap - 1] - rhs[iap - 1, ia]
                    y1 = lhs[iz, ik, iap] - rhs[iap, ia]
                    ap_endo[iz, ik, ia] = a_grid[iap - 1] - y0 * (a_grid[iap] - a_grid[iap - 1]) / (y1 - y0)
                    Wb_endo[iz, ik, ia] = (1 + k_grid[ik]) * (
                            Wb[iz, 0, iap - 1] + (ap_endo[iz, ik, ia] - a_grid[iap - 1]) *
                            (Wb[iz, 0, iap] - Wb[iz, 0, iap - 1]) / (a_grid[iap] - a_grid[iap - 1]))
    c_endo = Wb_endo ** (-eis)
    return ap_endo, c_endo


def step6(ap_endo, c_endo, z_grid, b_grid, a_grid, ra, rb, chi0, chi1, chi2):
    # b(z, k, a)
    zzz = z_grid[:, np.newaxis, np.newaxis]
    aaa = a_grid[np.newaxis, np.newaxis, :]
    b_endo = (c_endo + ap_endo + b_grid[0] - (1 + ra) * aaa + Psi_fun(ap_endo, aaa, ra, chi0, chi1, chi2) -
              zzz) / (1 + rb)

    # b'(z, b, a), a'(z, b, a)
    # assert np.min(np.diff(b_endo, axis=1)) < 0, 'b(kappa) is not decreasing'
    # assert np.min(np.diff(ap_endo, axis=1)) < 0, 'ap(kappa) is not decreasing'
    ap = utils.interpolate_y(b_endo[:, ::-1, :].swapaxes(1, 2), b_grid,
                             ap_endo[:, ::-1, :].swapaxes(1, 2)).swapaxes(1, 2)
    return ap

def income(e_grid, tax, w, N):
    z_grid = (1 - tax) * w * N * e_grid
    return z_grid


household_inc = household.attach_hetinput(income)


'''Part 3: Steady state'''


def hank_ss(beta_guess=0.9762610613299287, vphi_guess=1.7143455972367145, chi1_guess=6.408464776531462, r=0.0125, tot_wealth=13.9851201554679, K=10, delta=0.02, kappap=0.1,
            muw=1.1, Bh=1.04, Bg=2.8, G=0.2, eis=0.5, frisch=1, chi0=0.25, chi2=2, epsI=4, omega=0.005, kappaw=0.1,
            phi=1.5, nZ=3, nB=50, nA=70, nK=50, bmax=50, amax=4000, kmax=1, rho_z=0.966, sigma_z=0.92, noisy=True):
    """Solve steady state of full GE model. Calibrate (beta, vphi, chi1, alpha, mup, Z) to hit targets for
       (r, tot_wealth, Bh, K, Y=N=1).
    """

    # set up grid
    b_grid = utils.agrid(amax=bmax, n=nB)
    a_grid = utils.agrid(amax=amax, n=nA)

    p_grid = utils.agrid(amax=1.05, n=nB * 10, amin=0.95)
    #p_grid = np.concatenate((np.array([0.001]), p_grid)) # for exit
    k_grid = utils.agrid(amax=13, n=nA * 3, amin=7)

    #ef_grid, pif, Pif = utils.markov_rouwenhorst(rho=1.0, sigma=0.0, N=2) # representative firm
    ef_grid, pif, Pif = utils.markov_rouwenhorst(rho=0.66, sigma=0.0428, N=nZ) # baseline (Nakamura, Steinsson)
    #ef_grid, pif, Pif = utils.markov_rouwenhorst(rho=0.99, sigma=0.428, N=nZ) # larger persistence and std

    # solve analytically what we can for representative firm (serves as initializer)
    I = delta * K
    mc = 1 - r * (tot_wealth - Bg - K)
    alpha = (r + delta) * K / mc
    mup = 1 / mc
    w = (1 - alpha) * mc
    div = 1 - w - I


    # figure out initializer for firm
    Vk = 100 * (1 - delta + mc * alpha * K ** (alpha-1) - 2 * p_grid[:, np.newaxis] - k_grid) * np.ones((ef_grid.shape[0], 1, 1))
    Vp = 100 * (mup / (mup-1) / kappap - 5 * p_grid[:, np.newaxis] - 0.1 * k_grid) * np.ones((ef_grid.shape[0], 1, 1))
    V = (div + p_grid[:, np.newaxis] + k_grid) * np.ones((ef_grid.shape[0], 1, 1))
    Vd = div * np.ones((ef_grid.shape[0], 1, 1))

    beta_guess = 0.98355

    # for Bertrand we need an initializer for firm distribution, use model with monopolistic competition as initializer
    # PATH = "C:\\Users\\pyltsyna\\Dropbox\\PC\\Documents\\het_firm\\computed_models\\ss_RH_HF_closed_Bertrand.npz"
    # ss_loaded = np.load(PATH, allow_pickle=True)
    #
    # ss1 = {}
    # newvar = ss_loaded.files
    # for var in newvar:
    #     var_support = np.array(ss_loaded[var])
    #     ss1[var] = var_support[()]
    #
    # D_init = ss1['Df']

    alpha = 0.3299492385786802
    epsilon = 1.015 / 0.015
    def res_firm(x):
        w, Z = x
        out_firm = firm.ss(Vp=Vp, Vk=Vk, V=V, Vd=Vd, Pif=Pif, p_grid=p_grid, k_grid=k_grid, ef_grid=ef_grid, rf=r, w=w, Y=1,
                           alpha=alpha, kappap=kappap, delta=delta, epsI=epsI, epsilon=epsilon, P_index=1, Z=Z)
        p_equity = out_firm['DIV_IND'] / r
        return np.array([1 - out_firm['N_IND'],
                     1 - out_firm['YY_IND'] ** (epsilon / (epsilon - 1))])


    (w, Z), _ = utils.broyden_solver(res_firm, np.array([0.6601422079333671, 0.4677566343377928]), maxcount=500,
                                     noisy=noisy)
    # (w, Z), _ = utils.broyden_solver(res_firm, np.array([0.6601422079333671, 0.2677566343377928]), maxcount=500,
    #                                  noisy=noisy)


    out_firm = firm.ss(Vp=Vp, Vk=Vk, V=V, Vd=Vd, Pif=Pif, p_grid=p_grid, k_grid=k_grid, ef_grid=ef_grid, rf=r, w=w, Y=1,
                       alpha=alpha, kappap=kappap, delta=delta, epsI=epsI, epsilon=epsilon, P_index=1, Z=Z)

    check1 = 1 - out_firm['PP'] ** (1/(1-epsilon))

    check2 = np.sum(np.sum(np.sum((out_firm['V'] * r - out_firm['div_ind'] * (1 + r)) * out_firm['D'])))

    tax = (r * Bg + G) / w
    div = out_firm['DIV_IND']
    p_equity = out_firm['V_AGG']

    ra = r
    rb = r - omega
    mup = epsilon / (epsilon-1)

    capital_share = (r + delta) * out_firm['K'] / out_firm['MC_IND']

    # figure out initializer
    kk_grid = utils.agrid(amax=kmax, n=nK)
    e_grid, pi, Pi = utils.markov_rouwenhorst(rho=rho_z, sigma=sigma_z, N=nZ)
    z_grid = income(e_grid, tax, w, 1)
    Va = (0.6 + 1.1 * b_grid[:, np.newaxis] + a_grid) ** (-1 / eis) * np.ones((z_grid.shape[0], 1, 1))
    Vb = (0.5 + b_grid[:, np.newaxis] + 1.2 * a_grid) ** (-1 / eis) * np.ones((z_grid.shape[0], 1, 1))

    def res(x):
        beta_loc, vphi_loc, chi_loc = x
        if beta_loc > 0.999 / (1 + r) or vphi_loc < 0.001:
            raise ValueError('Clearly invalid inputs')
        out = household_inc.ss(Va=Va, Vb=Vb, Pi=Pi, a_grid=a_grid, b_grid=b_grid, N=1, tax=tax, w=w, e_grid=e_grid,
                               kk_grid=kk_grid, beta=beta_loc, eis=eis, rb=rb, ra=ra, chi0=chi0, chi1=chi_loc, chi2=chi2)
        asset_mkt = out['A'] + out['B'] - p_equity - Bg
        labor_mkt = vphi_loc - (1 - tax) * w * out['U'] / muw
        return np.array([asset_mkt, labor_mkt, out['B'] - Bh])

    # solve for beta, vphi, omega
    # (beta, vphi, chi1), _ = utils.broyden_solver(res, np.array([beta_guess, 1.4, 6]), noisy=noisy)
    (beta, vphi, chi1), _ = utils.broyden_solver(res, np.array([beta_guess, vphi_guess, chi1_guess]), noisy=noisy)

    # extra evaluation to report variables
    ss = household_inc.ss(Va=Va, Vb=Vb, Pi=Pi, a_grid=a_grid, b_grid=b_grid, N=1, tax=tax, w=w, e_grid=e_grid,
                          kk_grid=kk_grid, beta=beta, eis=eis, rb=rb, ra=ra, chi0=chi0, chi1=chi1, chi2=chi2)
    # other things of interest
    tot_wealth = ss['A'] + ss['B']
    pshare = p_equity / (tot_wealth - Bh)

    # calculate aggregate adjustment cost and check Walras's law
    chi = Psi_fun(ss['a'], a_grid, r, chi0, chi1, chi2)
    Chi = np.vdot(ss['D'], chi)
    goods_mkt = ss['C'] + out_firm['I_IND'] + G + Chi + omega * ss['B'] + out_firm['PSI_IND'] - 1

    ss.update({ # model variables
               'pi': 0, 'piw': 0, 'Q': 1, 'Y': 1, 'P_index': 1, 'div': div, 'Z': Z, 'w': w, 'tax': tax,
               'p_equity': p_equity, 'r': r, 'Bg': Bg, 'G': G, 'Chi': Chi, 'goods_mkt': goods_mkt,
               'chi': chi, 'phi': phi, 'pshare': pshare, 'rstar': r, 'i': r, 'rf': r,
               'tot_wealth': tot_wealth, 'Bh': Bh, 'N': 1,

                # parameters
               'beta': beta, 'vphi': vphi, 'omega': omega, 'alpha': alpha, 'delta': delta, 'mup': mup, 'muw': muw,
               'frisch': frisch, 'epsI': epsI, 'kappap': kappap, 'kappaw': kappaw, 'epsilon': epsilon,

                # grids
               'a_grid': a_grid, 'b_grid': b_grid, 'z_grid': z_grid, 'e_grid': e_grid, 'kk_grid': kk_grid,
               'ef_grid': ef_grid, 'p_grid': p_grid, 'k_grid': k_grid,
               'Vk': out_firm['Vk'], 'Vp': out_firm['Vp'], 'V': out_firm['V'], 'Pi': Pi, 'pif': pif, 'Pif': Pif,
               'Df': out_firm['D'],

                # het firms output
               'i_ind': out_firm['i_ind'], 'n_ind': out_firm['n_ind'], 'mc_ind': out_firm['mc_ind'],
               'markup_ind': out_firm['markup_ind'], 'k_adj_ind': out_firm['k_adj_ind'], 'pp': out_firm['pp'],
               'yy_ind': out_firm['yy_ind'], 'psi_ind': out_firm['psi_ind'], 'div_ind': out_firm['div_ind'],
               'y_ind': out_firm['y_ind'], 'k': out_firm['k'], 'p': out_firm['p'], 'q_ind': out_firm['q_ind'],
               'V_agg': out_firm['V_agg'],

               'I_IND': out_firm['I_IND'], 'N_IND': out_firm['N_IND'], 'MC_IND': out_firm['MC_IND'],
               'MARKUP_IND': out_firm['MARKUP_IND'], 'K_ADJ_IND': out_firm['K_ADJ_IND'], 'PP': out_firm['PP'],
               'YY_IND': out_firm['YY_IND'], 'PSI_IND': out_firm['PSI_IND'], 'DIV_IND': div,
               'Y_IND': out_firm['Y_IND'], 'K': out_firm['K'], 'P': out_firm['P'], 'Q_IND': out_firm['Q_IND'],
               'V_AGG': out_firm['V_AGG'], 'Vd': out_firm['Vd']})
    return ss

# ss = hank_ss()
# PATH = "C:\\Users\\pyltsyna\\Dropbox\\PC\\Documents\\het_firm\\computed_models\\ss_HF_NK_Bertrand.npz"
# np.savez(PATH, **ss)
# print('done')