from simple_block import simple
from solved_block import solved
from het_firm import firm_no_price_adj_costs_exit, firm_no_price_adj_costs, firm_no_price_adj_costs_exit_Bertrand_ss



@simple
def equil_firms(YY_IND, PP, Y, epsilon):
    y_equilibrium = YY_IND ** (epsilon / (epsilon-1)) - Y
    p_equilibrium = PP ** (1 / (1 - epsilon)) - 1
    return y_equilibrium, p_equilibrium

pricing_het = solved(block_list=[firm_no_price_adj_costs, equil_firms],
                    unknowns=['P_index'],
                    targets=['p_equilibrium'])

@simple
def firm_redefinition(N_IND):
    N = N_IND
    return N

@simple
def future_interest(r):
    rf = r(+1)
    return rf


@simple
def fiscal(r, w, N, G, Bg):
    tax = (r * Bg + G) / w / N
    return tax


@simple
def finance(p_equity, r, DIV_IND, omega, pshare, P_index):
    rb = r - omega
    ra = pshare(-1) * (DIV_IND + p_equity) / p_equity(-1) + (1-pshare(-1)) * (1 + r) - 1
    fisher = P_index - 1
    return rb, ra, fisher

@simple
def union(piw, N, tax, w, U, kappaw, muw, vphi, frisch, beta):
    wnkpc = vphi * N**(1+1/frisch) - (1-tax)*w*N*U / muw
    return wnkpc

@simple
def mkt_clearing(p_equity, A, B, Bg, C, I_IND, CHI, PSI_IND, omega, Y, G):
    asset_mkt = p_equity + Bg - B - A
    goods_mkt = C + I_IND + G + CHI + PSI_IND + omega * B(-1) - Y
    return asset_mkt, goods_mkt


@simple
def arbitrage(V_AGG):
    p_equity = V_AGG
    return p_equity

@simple
def share_value(p_equity, tot_wealth, Bh):
    pshare = p_equity / (tot_wealth - Bh)
    return pshare


@simple
def RANK_hh(C, eis, beta, rb, ra, chi0, chi1, chi2, tax, w, N, B, CHI, A):
    euler_eq = C ** (-1/eis) - beta * (1 + rb(+1)) * C(+1) ** (-1/eis)
    Psi1 = chi1 * np.abs(A - (1 + ra) * A(-1)) ** (chi2 - 1) * np.sign(A - (1+ra) * A(-1))
    Psi2 = - (1 + ra(+1)) * chi1 * np.abs(A(+1) - (1 + ra(+1)) * A) ** (chi2 - 1) * np.sign(A(+1) - (1+ra(+1)) * A)
    nonliq_asset_eq = (1 + Psi1) * C ** (-1/eis) - beta * (1 + ra(+1) - Psi2) * C(+1) ** (-1/eis)
    budget_const = (1-tax) * w * N + (1 + ra) * A(-1) + (1 + rb) * B(-1) - chi1 * np.abs(A - (1 + ra) * A(-1)) ** chi2 - C - A - B
    U = C ** (-1/eis)
    return euler_eq, nonliq_asset_eq, budget_const, U, Psi2

@simple
def markup(mup):
    epsilon = mup / (mup - 1)
    return epsilon