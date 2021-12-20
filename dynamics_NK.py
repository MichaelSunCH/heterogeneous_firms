from simple_block import simple
from solved_block import solved
from het_firm import firm, firm_Bertrand_ss, firm_exit


@simple
def equil_firms(YY_IND, PP, Y, P_index, epsilon):
    y_equilibrium = YY_IND ** (epsilon / (epsilon - 1)) - Y
    p_equilibrium = PP ** (1/(1-epsilon)) - P_index
    return p_equilibrium, y_equilibrium

pricing_het = solved(block_list=[firm, equil_firms],
                    unknowns=['P_index'],
                    targets=['p_equilibrium'])

@simple
def firm_redefinition(PP, K, delta, epsI, epsilon, N_IND, P_index):
    pi = (PP / PP(-1)) ** (1/(1-epsilon)) - 1
    pi2 = P_index / P_index(-1) - 1
    N = N_IND
    return pi, N, pi2

@simple
def future_interest(r):
    rf = r(+1)
    return rf

@simple
def taylor(rstar, pi, phi):
    #i = rstar
    i = rstar + phi * pi
    return i

@simple
def fiscal(r, w, N, G, Bg):
    tax = (r * Bg + G) / w / N
    return tax

@simple
def finance(pi, p_equity, i, r, DIV_IND, omega, pshare):
    rb = r - omega
    ra = pshare(-1) * (DIV_IND + p_equity) / p_equity(-1) + (1-pshare(-1)) * (1 + r) - 1
    fisher = 1 + i(-1) - (1 + r) * (1 + pi)
    return rb, ra, fisher

@simple
def wage(pi, w, N, muw, kappaw):
    piw = (1 + pi) * w / w(-1) - 1
    psiw = muw / (1 - muw) / 2 / kappaw * np.log(1 + piw) ** 2 * N
    return piw, psiw

@simple
def union(piw, N, tax, w, U, kappaw, muw, vphi, frisch, beta):
    wnkpc = kappaw * (vphi * N**(1+1/frisch) - (1-tax)*w*N*U / muw) + beta * np.log(1 + piw(+1)) - np.log(1 + piw)
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
def markup(mup):
    epsilon = mup / (mup - 1)
    return epsilon
