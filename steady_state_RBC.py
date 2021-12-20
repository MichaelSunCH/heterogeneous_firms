import numpy as np
from numba import njit
import utils
from het_block import het
from simple_block import simple
from het_firm import firm_no_price_adj_costs_exit, firm_no_price_adj_costs, firm_no_price_adj_costs_exit_Bertrand_ss
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

def Psi_fun_RA(ap, a, ra, chi0, chi1, chi2):
    a_with_return = (1 + ra) * a
    a_change = ap - a_with_return
    abs_a_change = np.abs(a_change)
    Psi = chi1 / chi2 * abs_a_change ** chi2
    return Psi


def Psi1_fun_RA(ap, a, ra, chi0, chi1, chi2):
    a_with_return = (1 + ra) * a
    a_change = ap - a_with_return
    abs_a_change = np.abs(a_change)
    sign_change = np.sign(a_change)
    Psi1 = chi1 * sign_change * abs_a_change ** (chi2 - 1)
    return Psi1


def Psi2_fun_RA(ap, a, ra, chi0, chi1, chi2):
    a_with_return = (1 + ra) * a
    a_change = ap - a_with_return
    abs_a_change = np.abs(a_change)
    sign_change = np.sign(a_change)
    Psi2 = -chi1 * sign_change * abs_a_change ** (chi2 - 1) * (1 + ra)
    return Psi2



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

def Psi_fun_rank(ap, a, ra, chi0, chi1, chi2):
    return chi1 / chi2 * np.abs(ap - (1 + ra) * a) ** chi2

def Psi1_fun_rank(ap, a, ra, chi0, chi1, chi2):
    return chi1 * np.abs(ap - (1 + ra) * a) ** (chi2 - 1) * np.sign(ap - (1+ra) * a)

def Psi2_fun_rank(ap, a, ra, chi0, chi1, chi2):
    return - (1 + ra) * chi1 * np.abs(ap - (1 + ra) * a) ** (chi2 - 1) * np.sign(ap - (1+ra) * a)



'''Part 3: Steady state'''

def hank_ss(beta_guess=0.976, vphi_guess=2.07, chi1_guess=6.416, r=0.0125, tot_wealth=13.95, K=10, delta=0.02,
            muw=1.1, Bh=1.04, Bg=2.8, G=0.2, eis=0.5, frisch=1, chi0=0.25, chi2=2, epsI=4, omega=0.005, kappaw=0.1,
            phi=1.5, nZ=3, nB=50, nA=70, nK=50, bmax=50, amax=4000, kmax=1, rho_z=0.966, sigma_z=0.92, noisy=True):
    """Solve steady state of full GE model. Calibrate (beta, vphi, chi1, alpha, mup, Z) to hit targets for
       (r, tot_wealth, Bh, K, Y=N=1).
    """

    # set up grid
    b_grid = utils.agrid(amax=bmax, n=nB)
    a_grid = utils.agrid(amax=amax, n=nA)
    k_grid = utils.agrid(amax=15, n=nA * 30, amin=0.001)

    ef_grid, pif, Pif = utils.markov_rouwenhorst(rho=0.66, sigma=0.0428, N=nZ) # baseline
    #ef_grid, pif, Pif = utils.markov_rouwenhorst(rho=0.99, sigma=0.0428*10, N=nZ) # larger persistence and std
    #ef_grid, pif, Pif = utils.markov_rouwenhorst(rho=1.0, sigma=0.0, N=nZ) # representative firm

    # solve analytically what we can for representative firm (serves as initializer)
    I = delta * K
    mc = 1 - r * (tot_wealth - Bg - K)
    alpha = (r + delta) * K / mc
    mup = 1 / mc
    w = (1 - alpha) * mc
    div = 1 - w - I


    # figure out initializer for firm
    Vk = 100 * (1 - delta + mc * alpha * K ** (alpha-1) - k_grid) * np.ones((ef_grid.shape[0], 1))
    V = (div + k_grid) * np.ones((ef_grid.shape[0], 1))
    Vd = (div + 0.01 * k_grid) * np.ones((ef_grid.shape[0], 1))

    alpha = 0.3299492385786802
    epsilon = 1.015 / 0.015
    
    ############ Uncomment for Bertrand ###############

    # PATH = "C:\\Users\\pyltsyna\\Dropbox\\PC\\Documents\\het_firm\\computed_models\\exit\\ss_RBC_HH_HF_closed_exit0.npz"
    # ss2_loaded = np.load(PATH, allow_pickle=True)

    # ss2 = {}
    # newvar = ss2_loaded.files
    # for var in newvar:
    #     var_support = np.array(ss2_loaded[var])
    #     ss2[var] = var_support[()]

    # D_init = ss2['Df']

    # D_init = np.load(
    #      'C:\\Users\\pyltsyna\\Dropbox\\PC\\Documents\\het_firm\\computed_models\\exit\\D_ex.npy')

    # def res_firm(x):
    #     w, Z = x
    #     out_firm = firm_no_price_adj_costs_exit_Bertrand_ss.ss_with_dist_iter(Vk=Vk, V=V, Vd=Vd, Pif=Pif, k_grid=k_grid, ef_grid=ef_grid, rf=r, w=w, Y=1,
    #                        alpha=alpha, delta=delta, epsI=epsI, epsilon=epsilon, P_index=1, Z=Z, r=r, D=D_init)
    #     p_equity = out_firm['DIV_IND'] / r
    #     return np.array([1 - out_firm['N_IND'],
    #                  1 - out_firm['YY_IND'] ** (epsilon / (epsilon - 1))])


    # (w, Z), _ = utils.broyden_solver(res_firm, np.array([0.482148533420027, 0.3607331432303956]), maxcount=500,
    #                                  noisy=noisy)
    # out_firm = firm_no_price_adj_costs_exit_Bertrand_ss.ss_with_dist_iter(Vk=Vk,  V=V, Vd=Vd, Pif=Pif, k_grid=k_grid, ef_grid=ef_grid, rf=r, w=w, Y=1,
    #                    alpha=alpha, delta=delta, epsI=epsI, epsilon=epsilon, P_index=1, Z=Z, r=r, D=D_init)

    ###################################################

    ############ Comment for Bertrand ###############
    def res_firm(x):
        w, Z = x
        out_firm = firm_no_price_adj_costs.ss(Vk=Vk, V=V, Pif=Pif, k_grid=k_grid, ef_grid=ef_grid, rf=r, w=w, Y=1,
                           alpha=alpha, delta=delta, epsI=epsI, epsilon=epsilon, P_index=1, Z=Z, Vd=Vd, r=r)
        p_equity = out_firm['DIV_IND'] / r
        return np.array([1 - out_firm['N_IND'],
                         1 - out_firm['YY_IND'] ** (epsilon / (epsilon - 1))])
    
    (w, Z), _ = utils.broyden_solver(res_firm, np.array([0.660132422449988, 0.46700545253883574]), maxcount=500,
                                     noisy=noisy)
    out_firm = firm_no_price_adj_costs.ss(Vk=Vk, V=V, Pif=Pif, k_grid=k_grid, ef_grid=ef_grid, rf=r, w=w, Y=1,
                       alpha=alpha, delta=delta, epsI=epsI, epsilon=epsilon, P_index=1, Z=Z, Vd=Vd, r=r)
    ###################################################


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
        # if beta_loc > 0.999 / (1 + r) or vphi_loc < 0.001:
        #     raise ValueError('Clearly invalid inputs')
        out = household_inc.ss(Va=Va, Vb=Vb, Pi=Pi, a_grid=a_grid, b_grid=b_grid, N=1, tax=tax, w=w, e_grid=e_grid,
                               kk_grid=kk_grid, beta=beta_loc, eis=eis, rb=rb, ra=ra, chi0=chi0, chi1=chi_loc, chi2=chi2)
        asset_mkt = out['A'] + out['B'] - p_equity - Bg
        labor_mkt = vphi_loc - (1 - tax) * w * out['U'] / muw
        return np.array([asset_mkt, labor_mkt, out['B'] - Bh])

    # solve for beta, vphi, omega
    (beta, vphi, chi1), _ = utils.broyden_solver(res, np.array([beta_guess, vphi_guess, chi1_guess]), noisy=noisy)
    
    ##### Representative household #####
    # beta = 1 / (1 + r)
    # def res(x):
    #     chi1_loc, A_loc = x
    #     foc_a = 1 + Psi1_fun_rank(A_loc, A_loc, ra, chi0, chi1_loc, chi2) - beta * (
    #                 1 + ra - Psi2_fun_rank(A_loc, A_loc, ra, chi0, chi1_loc, chi2))
    #     B = p_equity + Bg - A_loc
    #     return np.array([B - Bh, foc_a])
    #
    # # solve for beta, vphi, omega
    # (chi1, A), _ = utils.broyden_solver(res, np.array([chi1_guess, 14-1.04]), noisy=noisy)
    # B = Bh
    #
    # Chi = Psi_fun_rank(A, A, r, chi0, chi1, chi2)
    # C = (1 - tax) * w + ra * A + rb * B - Chi
    # vphi = C ** (-1 / eis) * (1 - tax) * w / muw
    # U = C ** (-1 / eis)
    #
    # # extra evaluation to report variables
    ss = household_inc.ss(Va=Va, Vb=Vb, Pi=Pi, a_grid=a_grid, b_grid=b_grid, N=1, tax=tax, w=w, e_grid=e_grid,
                          kk_grid=kk_grid, beta=beta, eis=eis, rb=rb, ra=ra, chi0=chi0, chi1=chi1, chi2=chi2)
    # ss = {'C': C, 'N': 1, 'w': w, 'tax': tax, 'beta': beta, 'eis': eis, 'rb': rb, 'ra': ra, 'chi0': chi0,
    #       'chi1': chi1, 'chi2': chi2, 'A': A, 'B': B, 'CHI': Chi, 'U': U}

    # other things of interest
    pshare = p_equity / (ss['A'] + ss['B'] - Bh)

    # calculate aggregate adjustment cost and check Walras's law
    chi = Psi_fun(ss['a'], a_grid, r, chi0, chi1, chi2)
    Chi = np.vdot(ss['D'], chi)
    goods_mkt = ss['C'] + out_firm['I_IND'] + G + Chi + omega * ss['B'] - 1
    assert np.abs(goods_mkt) < 1E-6

    ss.update({ # model variables
               'pi': 0, 'piw': 0, 'Q': 1, 'Y': 1, 'P_index': 1, 'div': div, 'Z': Z, 'w': w, 'tax': tax,
               'p_equity': p_equity, 'r': r, 'Bg': Bg, 'G': G, 'Chi': Chi, 'goods_mkt': goods_mkt,
               'phi': phi, 'pshare': pshare, 'rstar': r, 'i': r, 'rf': r,
               'tot_wealth': tot_wealth, 'Bh': Bh, 'N': 1,

                # parameters
               'beta': beta, 'vphi': vphi, 'omega': omega, 'alpha': alpha, 'delta': delta, 'mup': mup, 'muw': muw,
               'frisch': frisch, 'epsI': epsI, 'kappaw': kappaw, 'epsilon': epsilon,

                # grids
               'a_grid': a_grid, 'b_grid': b_grid, 'z_grid': z_grid, 'e_grid': e_grid, 'kk_grid': kk_grid,
               'ef_grid': ef_grid, 'k_grid': k_grid,
               'Vk': out_firm['Vk'], 'V': out_firm['V'], 'Pi': Pi, 'pif': pif, 'Pif': Pif,
               'Df': out_firm['D'], 'V_AGG': out_firm['V_AGG'], 'V_agg': out_firm['V_agg'],

                # het firms output
               'i_ind': out_firm['i_ind'], 'n_ind': out_firm['n_ind'], 'mc_ind': out_firm['mc_ind'],
               'markup_ind': out_firm['markup_ind'], 'k_adj_ind': out_firm['k_adj_ind'], 'pp': out_firm['pp'],
               'yy_ind': out_firm['yy_ind'], 'div_ind': out_firm['div_ind'],
               'y_ind': out_firm['y_ind'], 'k': out_firm['k'], 'p': out_firm['p'], 'q_ind': out_firm['q_ind'],

               'I_IND': out_firm['I_IND'], 'N_IND': out_firm['N_IND'], 'MC_IND': out_firm['MC_IND'],
               'MARKUP_IND': out_firm['MARKUP_IND'], 'K_ADJ_IND': out_firm['K_ADJ_IND'], 'PP': out_firm['PP'],
               'YY_IND': out_firm['YY_IND'], 'DIV_IND': div,
               'Y_IND': out_firm['Y_IND'], 'K': out_firm['K'], 'P': out_firm['P'], 'PSI_IND': 0,
               'Q_IND': out_firm['Q_IND'],

               'Vd': out_firm['Vd']})

    return ss

# ss = hank_ss()
#
# PATH = "C:\\Users\\Administrator\\Documents\\het_firm\\computed_models\\ss_HA_no_p.npz"
# np.savez(PATH, **ss)
# print('done')