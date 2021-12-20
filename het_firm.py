import numpy as np
from numba import njit
import utils
from het_block import het


@het(exogenous='Pif', policy=['p', 'k'], backward=['Vp', 'Vk', 'V', 'Vd'])  # order as in grid!
def firm(Vp_p, Vk_p, V_p, Vd_p, Pif_p, p_grid, k_grid, ef_grid, rf, w, Y, alpha, kappap, delta, epsI, epsilon, P_index, Z):
    # get grid dimensions
    nZ, nP, nK = Vp_p.shape

    # step 2: calculate Wp(z, p', k'), Wk(z, p', k'), W(z, p', k') for convenience
    Wp, Wk, W = post_decision_vfun(Vp_p, Vk_p, V_p, Pif_p, rf)
    Wd = (Vd_p.T @ (1 / (1 + rf) * Pif_p.T)).T

    # step 3: obtain k(z, p', k) and Wp(z, p', k) and W(z, p', k) through Wk(z, p', k') = f(k', k)
    k_endo, Wp_endo, W_endo, Wd_endo = foc_capital_d(Wk - 1,
                                          1 / delta / epsI * (k_grid[:, np.newaxis] / k_grid[np.newaxis, :] - 1),
                                          k_grid, nZ, nP, nK, Wp, W, Wd)

    # mc(z, p', k)
    mc = w / (1 - alpha) / Z / ef_grid[np.newaxis, np.newaxis, :, np.newaxis] / k_grid[np.newaxis, np.newaxis,
                                                                                np.newaxis, :] ** alpha * (
                 (p_grid[:, np.newaxis, np.newaxis, np.newaxis] / P_index) ** (-epsilon) * Y / Z / ef_grid[np.newaxis,
                                                                                                   np.newaxis, :,
                                                                                                   np.newaxis] /
                 k_grid[np.newaxis, np.newaxis, np.newaxis, :] ** alpha) ** (alpha / (1 - alpha))

    # step 4: p(z, p, k), k(z, p, k) and W(z, p, k) through rhs(p', p, z, k) = Wp(z, p', k)
    rhs = epsilon / kappap * np.log(
        p_grid[:, np.newaxis, np.newaxis, np.newaxis] / p_grid[np.newaxis, :, np.newaxis, np.newaxis]
        ) * Y / p_grid[:, np.newaxis, np.newaxis, np.newaxis] - (1 - epsilon) * Y / P_index * (
                  p_grid[:, np.newaxis, np.newaxis, np.newaxis] / P_index) ** (
              -epsilon) - epsilon * mc * Y / P_index * (
                  p_grid[:, np.newaxis, np.newaxis, np.newaxis] / P_index) ** (-epsilon - 1)

    p, k, W, Wd = foc_price_d(Wp_endo, rhs, p_grid, nZ, nP, nK, k_endo, W_endo, Wd_endo)

    y_ind = (p / P_index) ** (-epsilon) * Y

    n_ind = (y_ind / (Z * ef_grid[:, np.newaxis, np.newaxis] * k_grid[np.newaxis, np.newaxis, :] ** alpha)) ** (
                1 / (1 - alpha))

    mc_ind = w * n_ind ** alpha / (
                (1 - alpha) * Z * ef_grid[:, np.newaxis, np.newaxis] * k_grid[np.newaxis, np.newaxis, :] ** alpha)

    psi_ind = epsilon * 1 / (2 * kappap) * (np.log(p / p_grid[np.newaxis, :, np.newaxis])) ** 2 * Y

    k_adj_ind = 1 / delta / epsI / 2 * (k / k_grid[np.newaxis, np.newaxis, :] - 1) ** 2 * k_grid[np.newaxis, np.newaxis,
                                                                                          :]

    i_ind = k - (1 - delta) * k_grid[np.newaxis, np.newaxis, :] + k_adj_ind

    div_ind = p / P_index * y_ind - w * n_ind - i_ind - psi_ind

    markup_ind = p / P_index / mc_ind

    q_ind = (k / k_grid[np.newaxis, np.newaxis, :] - 1) / (delta * epsI) + 1

    # envelope conditions and update of the value function guess
    Vk = (Z * mc_ind * ef_grid[:, np.newaxis, np.newaxis] * alpha * k_grid[np.newaxis, np.newaxis, :] ** (
                alpha - 1) * n_ind ** (1 - alpha) +
          1 - delta - 1 / delta / 2 / epsI * (k / k_grid[np.newaxis, np.newaxis, :] - 1) ** 2 +
          1 / delta / epsI * (k / k_grid[np.newaxis, np.newaxis, :] - 1) * k / k_grid[np.newaxis, np.newaxis, :])

    Vp = epsilon * Y / kappap * np.log(p / p_grid[np.newaxis, :, np.newaxis]) / p_grid[np.newaxis, :, np.newaxis]

    #V = W + div_ind
    V = W + Wd
    Vd = div_ind

    pp = p ** (1 - epsilon)

    yy_ind = y_ind ** ((epsilon - 1) / epsilon)

    V_agg = V

    return Vp, Vk, V, Vd, p, k, i_ind, div_ind, y_ind, psi_ind, n_ind, mc_ind, markup_ind, pp, yy_ind, k_adj_ind, V_agg, q_ind

@het(exogenous='Pif', policy=['p', 'k'], backward=['Vp', 'Vk', 'V', 'Vd'])  # order as in grid!
def firm_exit(Vp_p, Vk_p, V_p, Vd_p, Pif_p, p_grid, k_grid, ef_grid, rf, w, Y, alpha, kappap, delta, epsI, epsilon, P_index, Z):
    # get grid dimensions
    nZ, nP, nK = Vp_p.shape

    # step 2: calculate Wp(z, p', k'), Wk(z, p', k'), W(z, p', k') for convenience
    Wp, Wk, W = post_decision_vfun(Vp_p, Vk_p, V_p, Pif_p, rf)
    Wd = (Vd_p.T @ (1 / (1 + rf) * Pif_p.T)).T

    # step 3: obtain k(z, p', k) and Wp(z, p', k) and W(z, p', k) through Wk(z, p', k') = f(k', k)
    k_endo, Wp_endo, W_endo, Wd_endo = foc_capital_d(Wk - 1,
                                          1 / delta / epsI * (k_grid[:, np.newaxis] / k_grid[np.newaxis, :] - 1),
                                          k_grid, nZ, nP, nK, Wp, W, Wd)

    # mc(z, p', k)
    mc = w / (1 - alpha) / Z / ef_grid[np.newaxis, np.newaxis, :, np.newaxis] / k_grid[np.newaxis, np.newaxis,
                                                                                np.newaxis, :] ** alpha * (
                 (p_grid[:, np.newaxis, np.newaxis, np.newaxis] / P_index) ** (-epsilon) * Y / Z / ef_grid[np.newaxis,
                                                                                                   np.newaxis, :,
                                                                                                   np.newaxis] /
                 k_grid[np.newaxis, np.newaxis, np.newaxis, :] ** alpha) ** (alpha / (1 - alpha))

    # step 4: p(z, p, k), k(z, p, k) and W(z, p, k) through rhs(p', p, z, k) = Wp(z, p', k)
    rhs = epsilon / kappap * np.log(
        p_grid[:, np.newaxis, np.newaxis, np.newaxis] / p_grid[np.newaxis, :, np.newaxis, np.newaxis]
        ) * Y / p_grid[:, np.newaxis, np.newaxis, np.newaxis] - (1 - epsilon) * Y / P_index * (
                  p_grid[:, np.newaxis, np.newaxis, np.newaxis] / P_index) ** (
              -epsilon) - epsilon * mc * Y / P_index * (
                  p_grid[:, np.newaxis, np.newaxis, np.newaxis] / P_index) ** (-epsilon - 1)

    p, k, W, Wd = foc_price_d(Wp_endo, rhs, p_grid, nZ, nP, nK, k_endo, W_endo, Wd_endo)

    y_ind = (p / P_index) ** (-epsilon) * Y

    n_ind = (y_ind / (Z * ef_grid[:, np.newaxis, np.newaxis] * k_grid[np.newaxis, np.newaxis, :] ** alpha)) ** (
                1 / (1 - alpha))

    mc_ind = w * n_ind ** alpha / (
                (1 - alpha) * Z * ef_grid[:, np.newaxis, np.newaxis] * k_grid[np.newaxis, np.newaxis, :] ** alpha)

    psi_ind = epsilon * 1 / (2 * kappap) * (np.log(p / p_grid[np.newaxis, :, np.newaxis])) ** 2 * Y

    k_adj_ind = 1 / delta / epsI / 2 * (k / k_grid[np.newaxis, np.newaxis, :] - 1) ** 2 * k_grid[np.newaxis, np.newaxis,
                                                                                          :]

    i_ind = k - (1 - delta) * k_grid[np.newaxis, np.newaxis, :] + k_adj_ind

    div_ind = p / P_index * y_ind - w * n_ind - i_ind - psi_ind

    markup_ind = p / P_index / mc_ind

    q_ind = (k / k_grid[np.newaxis, np.newaxis, :] - 1) / (delta * epsI) + 1

    # envelope conditions and update of the value function guess
    Vk = (Z * mc_ind * ef_grid[:, np.newaxis, np.newaxis] * alpha * k_grid[np.newaxis, np.newaxis, :] ** (
                alpha - 1) * n_ind ** (1 - alpha) +
          1 - delta - 1 / delta / 2 / epsI * (k / k_grid[np.newaxis, np.newaxis, :] - 1) ** 2 +
          1 / delta / epsI * (k / k_grid[np.newaxis, np.newaxis, :] - 1) * k / k_grid[np.newaxis, np.newaxis, :])

    Vp = epsilon * Y / kappap * np.log(p / p_grid[np.newaxis, :, np.newaxis]) / p_grid[np.newaxis, :, np.newaxis]

    #V = W + div_ind
    V = W + Wd
    Vd = div_ind

    # V = W + np.maximum(Wd, sell_next / (1 + rf))
    V = W + np.maximum(Wd, 0)
    Vj = div_ind + V

    p[Vj < 0] = p_grid[0]
    k[Vj < 0] = k_grid[0]

    y_ind = (p / P_index) ** (-epsilon) * Y

    n_ind = (y_ind / (Z * ef_grid[:, np.newaxis, np.newaxis] * k_grid[np.newaxis, np.newaxis, :] ** alpha)) ** (
            1 / (1 - alpha))

    mc_ind = w * n_ind ** alpha / (
            (1 - alpha) * Z * ef_grid[:, np.newaxis, np.newaxis] * k_grid[np.newaxis, np.newaxis, :] ** alpha)

    psi_ind = epsilon * 1 / (2 * kappap) * (np.log(p / p_grid[np.newaxis, :, np.newaxis])) ** 2 * Y

    k_adj_ind = 1 / delta / epsI / 2 * (k / k_grid[np.newaxis, np.newaxis, :] - 1) ** 2 * k_grid[np.newaxis, np.newaxis,
                                                                                          :]

    i_ind = k - (1 - delta) * k_grid[np.newaxis, np.newaxis, :] + k_adj_ind

    div_ind = p / P_index * y_ind - w * n_ind - i_ind - psi_ind

    markup_ind = p / P_index / mc_ind

    q_ind = (k / k_grid[np.newaxis, np.newaxis, :] - 1) / (delta * epsI) + 1

    # envelope conditions and update of the value function guess
    Vk = (Z * mc_ind * ef_grid[:, np.newaxis, np.newaxis] * alpha * k_grid[np.newaxis, np.newaxis, :] ** (
            alpha - 1) * n_ind ** (1 - alpha) +
          1 - delta - 1 / delta / 2 / epsI * (k / k_grid[np.newaxis, np.newaxis, :] - 1) ** 2 +
          1 / delta / epsI * (k / k_grid[np.newaxis, np.newaxis, :] - 1) * k / k_grid[np.newaxis, np.newaxis, :])

    Vp = epsilon * Y / kappap * np.log(p / p_grid[np.newaxis, :, np.newaxis]) / p_grid[np.newaxis, :, np.newaxis]

    pp = p ** (1 - epsilon)

    yy_ind = y_ind ** ((epsilon - 1) / epsilon)

    Vd = div_ind
    # V[Vj < sell_ind] = 0
    V[Vj < 0] = 0
    V_agg = V

    return Vp, Vk, V, Vd, p, k, i_ind, div_ind, y_ind, psi_ind, n_ind, mc_ind, markup_ind, pp, yy_ind, k_adj_ind, V_agg, q_ind

# here distribution of firms is calculated too
@het(exogenous='Pif', policy=['p', 'k'], backward=['Vp', 'Vk', 'V', 'Vd'])  # order as in grid!
def firm_Bertrand_ss(Vp_p, Vk_p, V_p, Vd_p, Pif_p, p_grid, k_grid, ef_grid, rf, w, Y, alpha, kappap, delta, epsI, epsilon, P_index, Z, D):
    # get grid dimensions
    nZ, nP, nK = Vp_p.shape

    # step 2: calculate Wp(z, p', k'), Wk(z, p', k'), W(z, p', k') for convenience
    Wp, Wk, W = post_decision_vfun(Vp_p, Vk_p, V_p, Pif_p, rf)
    Wd = (Vd_p.T @ (1 / (1 + rf) * Pif_p.T)).T

    # step 3: obtain k(z, p', k) and Wp(z, p', k) and W(z, p', k) through Wk(z, p', k') = f(k', k)
    k_endo, Wp_endo, W_endo, Wd_endo = foc_capital_d(Wk - 1,
                                          1 / delta / epsI * (k_grid[:, np.newaxis] / k_grid[np.newaxis, :] - 1),
                                          k_grid, nZ, nP, nK, Wp, W, Wd)

    # mc(z, p', k)
    mc = w / (1 - alpha) / Z / ef_grid[np.newaxis, np.newaxis, :, np.newaxis] / k_grid[np.newaxis, np.newaxis,
                                                                                np.newaxis, :] ** alpha * (
                 (p_grid[:, np.newaxis, np.newaxis, np.newaxis] / P_index) ** (-epsilon) * Y / Z / ef_grid[np.newaxis,
                                                                                                   np.newaxis, :,
                                                                                                   np.newaxis] /
                 k_grid[np.newaxis, np.newaxis, np.newaxis, :] ** alpha) ** (alpha / (1 - alpha))

    D_swaped = np.swapaxes(D, 0, 1)
    der_pP = p_grid[:, np.newaxis, np.newaxis, np.newaxis] ** (- epsilon) / P_index ** (- epsilon) * D_swaped[np.newaxis, :, :, :]

    # step 4: p(z, p, k), k(z, p, k) and W(z, p, k) through rhs(p', p, z, k) = Wp(z, p', k)
    rhs = epsilon / kappap * np.log(
        p_grid[:, np.newaxis, np.newaxis, np.newaxis] / p_grid[np.newaxis, :, np.newaxis, np.newaxis]
        ) * Y / p_grid[:, np.newaxis, np.newaxis, np.newaxis] - (1 - epsilon) * Y / P_index * (
                  p_grid[:, np.newaxis, np.newaxis, np.newaxis] / P_index) ** (
              -epsilon) * (1 - p_grid[:, np.newaxis, np.newaxis, np.newaxis] / P_index * der_pP) - epsilon * mc * Y / P_index * (
                  p_grid[:, np.newaxis, np.newaxis, np.newaxis] / P_index) ** (-epsilon - 1) * (1 - p_grid[:, np.newaxis, np.newaxis, np.newaxis] / P_index * der_pP)

    p, k, W, Wd = foc_price_d(Wp_endo, rhs, p_grid, nZ, nP, nK, k_endo, W_endo, Wd_endo)

    y_ind = (p / P_index) ** (-epsilon) * Y

    n_ind = (y_ind / (Z * ef_grid[:, np.newaxis, np.newaxis] * k_grid[np.newaxis, np.newaxis, :] ** alpha)) ** (
                1 / (1 - alpha))

    mc_ind = w * n_ind ** alpha / (
                (1 - alpha) * Z * ef_grid[:, np.newaxis, np.newaxis] * k_grid[np.newaxis, np.newaxis, :] ** alpha)

    psi_ind = epsilon * 1 / (2 * kappap) * (np.log(p / p_grid[np.newaxis, :, np.newaxis])) ** 2 * Y

    k_adj_ind = 1 / delta / epsI / 2 * (k / k_grid[np.newaxis, np.newaxis, :] - 1) ** 2 * k_grid[np.newaxis, np.newaxis,
                                                                                          :]

    i_ind = k - (1 - delta) * k_grid[np.newaxis, np.newaxis, :] + k_adj_ind

    div_ind = p / P_index * y_ind - w * n_ind - i_ind - psi_ind

    markup_ind = p / P_index / mc_ind

    q_ind = (k / k_grid[np.newaxis, np.newaxis, :] - 1) / (delta * epsI) + 1

    # envelope conditions and update of the value function guess
    Vk = (Z * mc_ind * ef_grid[:, np.newaxis, np.newaxis] * alpha * k_grid[np.newaxis, np.newaxis, :] ** (
                alpha - 1) * n_ind ** (1 - alpha) +
          1 - delta - 1 / delta / 2 / epsI * (k / k_grid[np.newaxis, np.newaxis, :] - 1) ** 2 +
          1 / delta / epsI * (k / k_grid[np.newaxis, np.newaxis, :] - 1) * k / k_grid[np.newaxis, np.newaxis, :])

    Vp = epsilon * Y / kappap * np.log(p / p_grid[np.newaxis, :, np.newaxis]) / p_grid[np.newaxis, :, np.newaxis]

    #V = W + div_ind
    V = W + Wd
    Vd = div_ind

    pp = p ** (1 - epsilon)

    yy_ind = y_ind ** ((epsilon - 1) / epsilon)

    V_agg = V

    return Vp, Vk, V, Vd, p, k, i_ind, div_ind, y_ind, psi_ind, n_ind, mc_ind, markup_ind, pp, yy_ind, k_adj_ind, V_agg, q_ind

# here distribution of firms is passed as a parameter
@het(exogenous='Pif', policy=['p', 'k'], backward=['Vp', 'Vk', 'V', 'Vd'])  # order as in grid!
def firm_Bertrand(Vp_p, Vk_p, V_p, Vd_p, Pif_p, p_grid, k_grid, ef_grid, rf, w, Y, alpha, kappap, delta, epsI,
                     epsilon, P_index, Z, Df):
    # get grid dimensions
    nZ, nP, nK = Vp_p.shape

    # step 2: calculate Wp(z, p', k'), Wk(z, p', k'), W(z, p', k') for convenience
    Wp, Wk, W = post_decision_vfun(Vp_p, Vk_p, V_p, Pif_p, rf)
    Wd = (Vd_p.T @ (1 / (1 + rf) * Pif_p.T)).T

    # step 3: obtain k(z, p', k) and Wp(z, p', k) and W(z, p', k) through Wk(z, p', k') = f(k', k)
    k_endo, Wp_endo, W_endo, Wd_endo = foc_capital_d(Wk - 1,
                                                     1 / delta / epsI * (
                                                                 k_grid[:, np.newaxis] / k_grid[np.newaxis, :] - 1),
                                                     k_grid, nZ, nP, nK, Wp, W, Wd)

    # mc(z, p', k)
    mc = w / (1 - alpha) / Z / ef_grid[np.newaxis, np.newaxis, :, np.newaxis] / k_grid[np.newaxis, np.newaxis,
                                                                                np.newaxis, :] ** alpha * (
                 (p_grid[:, np.newaxis, np.newaxis, np.newaxis] / P_index) ** (-epsilon) * Y / Z / ef_grid[np.newaxis,
                                                                                                   np.newaxis, :,
                                                                                                   np.newaxis] /
                 k_grid[np.newaxis, np.newaxis, np.newaxis, :] ** alpha) ** (alpha / (1 - alpha))

    D_swaped = np.swapaxes(Df, 0, 1)
    der_pP = p_grid[:, np.newaxis, np.newaxis, np.newaxis] ** (- epsilon) / P_index ** (- epsilon) * D_swaped[
                                                                                                     np.newaxis, :, :,
                                                                                                     :]

    # step 4: p(z, p, k), k(z, p, k) and W(z, p, k) through rhs(p', p, z, k) = Wp(z, p', k)
    rhs = epsilon / kappap * np.log(
        p_grid[:, np.newaxis, np.newaxis, np.newaxis] / p_grid[np.newaxis, :, np.newaxis, np.newaxis]
    ) * Y / p_grid[:, np.newaxis, np.newaxis, np.newaxis] - (1 - epsilon) * Y / P_index * (
                  p_grid[:, np.newaxis, np.newaxis, np.newaxis] / P_index) ** (
              -epsilon) * (1 - p_grid[:, np.newaxis, np.newaxis,
                               np.newaxis] / P_index * der_pP) - epsilon * mc * Y / P_index * (
                  p_grid[:, np.newaxis, np.newaxis, np.newaxis] / P_index) ** (-epsilon - 1) * (
                      1 - p_grid[:, np.newaxis, np.newaxis, np.newaxis] / P_index * der_pP)

    p, k, W, Wd = foc_price_d(Wp_endo, rhs, p_grid, nZ, nP, nK, k_endo, W_endo, Wd_endo)

    y_ind = (p / P_index) ** (-epsilon) * Y

    n_ind = (y_ind / (Z * ef_grid[:, np.newaxis, np.newaxis] * k_grid[np.newaxis, np.newaxis, :] ** alpha)) ** (
            1 / (1 - alpha))

    mc_ind = w * n_ind ** alpha / (
            (1 - alpha) * Z * ef_grid[:, np.newaxis, np.newaxis] * k_grid[np.newaxis, np.newaxis, :] ** alpha)

    psi_ind = epsilon * 1 / (2 * kappap) * (np.log(p / p_grid[np.newaxis, :, np.newaxis])) ** 2 * Y

    k_adj_ind = 1 / delta / epsI / 2 * (k / k_grid[np.newaxis, np.newaxis, :] - 1) ** 2 * k_grid[np.newaxis, np.newaxis,
                                                                                          :]

    i_ind = k - (1 - delta) * k_grid[np.newaxis, np.newaxis, :] + k_adj_ind

    div_ind = p / P_index * y_ind - w * n_ind - i_ind - psi_ind

    markup_ind = p / P_index / mc_ind

    q_ind = (k / k_grid[np.newaxis, np.newaxis, :] - 1) / (delta * epsI) + 1

    # envelope conditions and update of the value function guess
    Vk = (Z * mc_ind * ef_grid[:, np.newaxis, np.newaxis] * alpha * k_grid[np.newaxis, np.newaxis, :] ** (
            alpha - 1) * n_ind ** (1 - alpha) +
          1 - delta - 1 / delta / 2 / epsI * (k / k_grid[np.newaxis, np.newaxis, :] - 1) ** 2 +
          1 / delta / epsI * (k / k_grid[np.newaxis, np.newaxis, :] - 1) * k / k_grid[np.newaxis, np.newaxis, :])

    Vp = epsilon * Y / kappap * np.log(p / p_grid[np.newaxis, :, np.newaxis]) / p_grid[np.newaxis, :, np.newaxis]

    # V = W + div_ind
    V = W + Wd
    Vd = div_ind

    pp = p ** (1 - epsilon)

    yy_ind = y_ind ** ((epsilon - 1) / epsilon)

    V_agg = V

    return Vp, Vk, V, Vd, p, k, i_ind, div_ind, y_ind, psi_ind, n_ind, mc_ind, markup_ind, pp, yy_ind, k_adj_ind, V_agg, q_ind


@het(exogenous='Pif', policy=['p', 'k'], backward=['Vp', 'Vk', 'V', 'Vd'])  # order as in grid!
def firm_open(Vp_p, Vk_p, V_p, Vd_p, Pif_p, p_grid, k_grid, ef_grid, rf, w, Y, alpha, kappap, delta, epsI, epsilon, P_index, Z, p_busket):
    # get grid dimensions
    nZ, nP, nK = Vp_p.shape

    # step 2: calculate Wp(z, p', k'), Wk(z, p', k'), W(z, p', k') for convenience
    Wp, Wk, W = post_decision_vfun(Vp_p, Vk_p, V_p, Pif_p, rf)
    Wd = (Vd_p.T @ (1 / (1 + rf) * Pif_p.T)).T

    # step 3: obtain k(z, p', k) and Wp(z, p', k) and W(z, p', k) through Wk(z, p', k') = f(k', k)
    k_endo, Wp_endo, W_endo, Wd_endo = foc_capital_d(Wk - 1,
                                          1 / delta / epsI * (k_grid[:, np.newaxis] / k_grid[np.newaxis, :] - 1),
                                          k_grid, nZ, nP, nK, Wp, W, Wd)

    # mc(z, p', k)
    mc = w / (1 - alpha) / Z / ef_grid[np.newaxis, np.newaxis, :, np.newaxis] / k_grid[np.newaxis, np.newaxis,
                                                                                np.newaxis, :] ** alpha * (
                 (p_grid[:, np.newaxis, np.newaxis, np.newaxis] / P_index) ** (-epsilon) * Y / Z / ef_grid[np.newaxis,
                                                                                                   np.newaxis, :,
                                                                                                   np.newaxis] /
                 k_grid[np.newaxis, np.newaxis, np.newaxis, :] ** alpha) ** (alpha / (1 - alpha))

    # step 4: p(z, p, k), k(z, p, k) and W(z, p, k) through rhs(p', p, z, k) = Wp(z, p', k)
    rhs = epsilon / kappap * np.log(
        p_grid[:, np.newaxis, np.newaxis, np.newaxis] / p_grid[np.newaxis, :, np.newaxis, np.newaxis]
        ) * Y / p_grid[:, np.newaxis, np.newaxis, np.newaxis] - (1 - epsilon) * Y / p_busket * (
                  p_grid[:, np.newaxis, np.newaxis, np.newaxis] / P_index) ** (
              -epsilon) - epsilon * mc * Y / P_index * (
                  p_grid[:, np.newaxis, np.newaxis, np.newaxis] / P_index) ** (-epsilon - 1)

    p, k, W, Wd = foc_price_d(Wp_endo, rhs, p_grid, nZ, nP, nK, k_endo, W_endo, Wd_endo)

    y_ind = (p / P_index) ** (-epsilon) * Y

    n_ind = (y_ind / (Z * ef_grid[:, np.newaxis, np.newaxis] * k_grid[np.newaxis, np.newaxis, :] ** alpha)) ** (
                1 / (1 - alpha))

    mc_ind = w * n_ind ** alpha / (
                (1 - alpha) * Z * ef_grid[:, np.newaxis, np.newaxis] * k_grid[np.newaxis, np.newaxis, :] ** alpha)

    psi_ind = epsilon * 1 / (2 * kappap) * (np.log(p / p_grid[np.newaxis, :, np.newaxis])) ** 2 * Y

    k_adj_ind = 1 / delta / epsI / 2 * (k / k_grid[np.newaxis, np.newaxis, :] - 1) ** 2 * k_grid[np.newaxis, np.newaxis,
                                                                                          :]

    i_ind = k - (1 - delta) * k_grid[np.newaxis, np.newaxis, :] + k_adj_ind

    div_ind = p / p_busket * y_ind - w * n_ind - i_ind - psi_ind

    markup_ind = p / P_index / mc_ind

    q_ind = (k / k_grid[np.newaxis, np.newaxis, :] - 1) / (delta * epsI) + 1

    # envelope conditions and update of the value function guess
    Vk = (Z * mc_ind * ef_grid[:, np.newaxis, np.newaxis] * alpha * k_grid[np.newaxis, np.newaxis, :] ** (
                alpha - 1) * n_ind ** (1 - alpha) +
          1 - delta - 1 / delta / 2 / epsI * (k / k_grid[np.newaxis, np.newaxis, :] - 1) ** 2 +
          1 / delta / epsI * (k / k_grid[np.newaxis, np.newaxis, :] - 1) * k / k_grid[np.newaxis, np.newaxis, :])

    Vp = epsilon * Y / kappap * np.log(p / p_grid[np.newaxis, :, np.newaxis]) / p_grid[np.newaxis, :, np.newaxis]

    #V = W + div_ind
    V = W + Wd
    Vd = div_ind

    pp = p ** (1 - epsilon)

    yy_ind = y_ind ** ((epsilon - 1) / epsilon)

    V_agg = V

    return Vp, Vk, V, Vd, p, k, i_ind, div_ind, y_ind, psi_ind, n_ind, mc_ind, markup_ind, pp, yy_ind, k_adj_ind, V_agg, q_ind

@het(exogenous='Pif', policy=['p', 'k'], backward=['Vp', 'Vk', 'V', 'Vd'])  # order as in grid!
def firm_open_Bertrand_ss(Vp_p, Vk_p, V_p, Vd_p, Pif_p, p_grid, k_grid, ef_grid, rf, w, Y, alpha, kappap, delta, epsI,
                          epsilon, P_index, Z, p_busket, gamma_hb, eta_hh, D):
    # get grid dimensions
    nZ, nP, nK = Vp_p.shape

    # step 2: calculate Wp(z, p', k'), Wk(z, p', k'), W(z, p', k') for convenience
    Wp, Wk, W = post_decision_vfun(Vp_p, Vk_p, V_p, Pif_p, rf)
    Wd = (Vd_p.T @ (1 / (1 + rf) * Pif_p.T)).T

    # step 3: obtain k(z, p', k) and Wp(z, p', k) and W(z, p', k) through Wk(z, p', k') = f(k', k)
    k_endo, Wp_endo, W_endo, Wd_endo = foc_capital_d(Wk - 1,
                                          1 / delta / epsI * (k_grid[:, np.newaxis] / k_grid[np.newaxis, :] - 1),
                                          k_grid, nZ, nP, nK, Wp, W, Wd)

    # mc(z, p', k)
    mc = w / (1 - alpha) / Z / ef_grid[np.newaxis, np.newaxis, :, np.newaxis] / k_grid[np.newaxis, np.newaxis,
                                                                                np.newaxis, :] ** alpha * (
                 (p_grid[:, np.newaxis, np.newaxis, np.newaxis] / P_index) ** (-epsilon) * Y / Z / ef_grid[np.newaxis,
                                                                                                   np.newaxis, :,
                                                                                                   np.newaxis] /
                 k_grid[np.newaxis, np.newaxis, np.newaxis, :] ** alpha) ** (alpha / (1 - alpha))

    D_swaped = np.swapaxes(D, 0, 1)
    der_pP = p_grid[:, np.newaxis, np.newaxis, np.newaxis] ** (- epsilon) / P_index ** (- epsilon) * D_swaped[np.newaxis, :, :, :]

    # step 4: p(z, p, k), k(z, p, k) and W(z, p, k) through rhs(p', p, z, k) = Wp(z, p', k)
    rhs = epsilon / kappap * np.log(
        p_grid[:, np.newaxis, np.newaxis, np.newaxis] / p_grid[np.newaxis, :, np.newaxis, np.newaxis]
        ) * Y / p_grid[:, np.newaxis, np.newaxis, np.newaxis] - (1 - epsilon) * Y / p_busket * (
                  p_grid[:, np.newaxis, np.newaxis, np.newaxis] / P_index) ** (
              -epsilon) - epsilon * (p_grid[:, np.newaxis, np.newaxis, np.newaxis] / P_index) ** (
              1-epsilon) / p_busket * Y * der_pP + (1 - gamma_hb) * (
            p_grid[:, np.newaxis, np.newaxis, np.newaxis] / P_index) ** (1-epsilon) * P_index / p_busket ** 2 * (
            P_index / p_busket) ** (-eta_hh) * der_pP - epsilon * mc * Y / P_index * (
                  p_grid[:, np.newaxis, np.newaxis, np.newaxis] / P_index) ** (-epsilon - 1) * (
            1 - p_grid[:, np.newaxis, np.newaxis, np.newaxis] / P_index * der_pP)

    p, k, W, Wd = foc_price_d(Wp_endo, rhs, p_grid, nZ, nP, nK, k_endo, W_endo, Wd_endo)

    y_ind = (p / P_index) ** (-epsilon) * Y

    n_ind = (y_ind / (Z * ef_grid[:, np.newaxis, np.newaxis] * k_grid[np.newaxis, np.newaxis, :] ** alpha)) ** (
                1 / (1 - alpha))

    mc_ind = w * n_ind ** alpha / (
                (1 - alpha) * Z * ef_grid[:, np.newaxis, np.newaxis] * k_grid[np.newaxis, np.newaxis, :] ** alpha)

    psi_ind = epsilon * 1 / (2 * kappap) * (np.log(p / p_grid[np.newaxis, :, np.newaxis])) ** 2 * Y

    k_adj_ind = 1 / delta / epsI / 2 * (k / k_grid[np.newaxis, np.newaxis, :] - 1) ** 2 * k_grid[np.newaxis, np.newaxis,
                                                                                          :]

    i_ind = k - (1 - delta) * k_grid[np.newaxis, np.newaxis, :] + k_adj_ind

    div_ind = p / p_busket * y_ind - w * n_ind - i_ind - psi_ind

    markup_ind = p / P_index / mc_ind

    q_ind = (k / k_grid[np.newaxis, np.newaxis, :] - 1) / (delta * epsI) + 1

    # envelope conditions and update of the value function guess
    Vk = (Z * mc_ind * ef_grid[:, np.newaxis, np.newaxis] * alpha * k_grid[np.newaxis, np.newaxis, :] ** (
                alpha - 1) * n_ind ** (1 - alpha) +
          1 - delta - 1 / delta / 2 / epsI * (k / k_grid[np.newaxis, np.newaxis, :] - 1) ** 2 +
          1 / delta / epsI * (k / k_grid[np.newaxis, np.newaxis, :] - 1) * k / k_grid[np.newaxis, np.newaxis, :])

    Vp = epsilon * Y / kappap * np.log(p / p_grid[np.newaxis, :, np.newaxis]) / p_grid[np.newaxis, :, np.newaxis]

    #V = W + div_ind
    V = W + Wd
    Vd = div_ind

    pp = p ** (1 - epsilon)

    yy_ind = y_ind ** ((epsilon - 1) / epsilon)

    V_agg = V

    return Vp, Vk, V, Vd, p, k, i_ind, div_ind, y_ind, psi_ind, n_ind, mc_ind, markup_ind, pp, yy_ind, k_adj_ind, V_agg, q_ind


@het(exogenous='Pif', policy=['p', 'k'], backward=['Vp', 'Vk', 'V', 'Vd'])  # order as in grid!
def firm_open_Bertrand(Vp_p, Vk_p, V_p, Vd_p, Pif_p, p_grid, k_grid, ef_grid, rf, w, Y, alpha, kappap, delta, epsI,
                          epsilon, P_index, Z, p_busket, gamma_hb, eta_hh, Df):
    # get grid dimensions
    nZ, nP, nK = Vp_p.shape

    # step 2: calculate Wp(z, p', k'), Wk(z, p', k'), W(z, p', k') for convenience
    Wp, Wk, W = post_decision_vfun(Vp_p, Vk_p, V_p, Pif_p, rf)
    Wd = (Vd_p.T @ (1 / (1 + rf) * Pif_p.T)).T

    # step 3: obtain k(z, p', k) and Wp(z, p', k) and W(z, p', k) through Wk(z, p', k') = f(k', k)
    k_endo, Wp_endo, W_endo, Wd_endo = foc_capital_d(Wk - 1,
                                          1 / delta / epsI * (k_grid[:, np.newaxis] / k_grid[np.newaxis, :] - 1),
                                          k_grid, nZ, nP, nK, Wp, W, Wd)

    # mc(z, p', k)
    mc = w / (1 - alpha) / Z / ef_grid[np.newaxis, np.newaxis, :, np.newaxis] / k_grid[np.newaxis, np.newaxis,
                                                                                np.newaxis, :] ** alpha * (
                 (p_grid[:, np.newaxis, np.newaxis, np.newaxis] / P_index) ** (-epsilon) * Y / Z / ef_grid[np.newaxis,
                                                                                                   np.newaxis, :,
                                                                                                   np.newaxis] /
                 k_grid[np.newaxis, np.newaxis, np.newaxis, :] ** alpha) ** (alpha / (1 - alpha))

    D_swaped = np.swapaxes(Df, 0, 1)
    der_pP = p_grid[:, np.newaxis, np.newaxis, np.newaxis] ** (- epsilon) / P_index ** (- epsilon) * D_swaped[np.newaxis, :, :, :]


    # step 4: p(z, p, k), k(z, p, k) and W(z, p, k) through rhs(p', p, z, k) = Wp(z, p', k)
    rhs = epsilon / kappap * np.log(
        p_grid[:, np.newaxis, np.newaxis, np.newaxis] / p_grid[np.newaxis, :, np.newaxis, np.newaxis]
    ) * Y / p_grid[:, np.newaxis, np.newaxis, np.newaxis] - (1 - epsilon) * Y / p_busket * (
                  p_grid[:, np.newaxis, np.newaxis, np.newaxis] / P_index) ** (
              -epsilon) - epsilon * (p_grid[:, np.newaxis, np.newaxis, np.newaxis] / P_index) ** (
                  1 - epsilon) / p_busket * Y * der_pP + (1 - gamma_hb) * (
                  p_grid[:, np.newaxis, np.newaxis, np.newaxis] / P_index) ** (
                      1 - epsilon) * P_index / p_busket ** 2 * (
                  P_index / p_busket) ** (-eta_hh) * der_pP - epsilon * mc * Y / P_index * (
                  p_grid[:, np.newaxis, np.newaxis, np.newaxis] / P_index) ** (-epsilon - 1) * (
                  1 - p_grid[:, np.newaxis, np.newaxis, np.newaxis] / P_index * der_pP)

    p, k, W, Wd = foc_price_d(Wp_endo, rhs, p_grid, nZ, nP, nK, k_endo, W_endo, Wd_endo)

    y_ind = (p / P_index) ** (-epsilon) * Y

    n_ind = (y_ind / (Z * ef_grid[:, np.newaxis, np.newaxis] * k_grid[np.newaxis, np.newaxis, :] ** alpha)) ** (
            1 / (1 - alpha))

    mc_ind = w * n_ind ** alpha / (
            (1 - alpha) * Z * ef_grid[:, np.newaxis, np.newaxis] * k_grid[np.newaxis, np.newaxis, :] ** alpha)

    psi_ind = epsilon * 1 / (2 * kappap) * (np.log(p / p_grid[np.newaxis, :, np.newaxis])) ** 2 * Y

    k_adj_ind = 1 / delta / epsI / 2 * (k / k_grid[np.newaxis, np.newaxis, :] - 1) ** 2 * k_grid[np.newaxis, np.newaxis,
                                                                                          :]

    i_ind = k - (1 - delta) * k_grid[np.newaxis, np.newaxis, :] + k_adj_ind

    div_ind = p / p_busket * y_ind - w * n_ind - i_ind - psi_ind

    markup_ind = p / P_index / mc_ind

    q_ind = (k / k_grid[np.newaxis, np.newaxis, :] - 1) / (delta * epsI) + 1

    # envelope conditions and update of the value function guess
    Vk = (Z * mc_ind * ef_grid[:, np.newaxis, np.newaxis] * alpha * k_grid[np.newaxis, np.newaxis, :] ** (
            alpha - 1) * n_ind ** (1 - alpha) +
          1 - delta - 1 / delta / 2 / epsI * (k / k_grid[np.newaxis, np.newaxis, :] - 1) ** 2 +
          1 / delta / epsI * (k / k_grid[np.newaxis, np.newaxis, :] - 1) * k / k_grid[np.newaxis, np.newaxis, :])

    Vp = epsilon * Y / kappap * np.log(p / p_grid[np.newaxis, :, np.newaxis]) / p_grid[np.newaxis, :, np.newaxis]

    # V = W + div_ind
    V = W + Wd
    Vd = div_ind

    pp = p ** (1 - epsilon)

    yy_ind = y_ind ** ((epsilon - 1) / epsilon)

    V_agg = V

    return Vp, Vk, V, Vd, p, k, i_ind, div_ind, y_ind, psi_ind, n_ind, mc_ind, markup_ind, pp, yy_ind, k_adj_ind, V_agg, q_ind


@het(exogenous='Pif', policy=['k'], backward=['Vk', 'V', 'Vd'])  # order as in grid!
def firm_no_price_adj_costs(Vk_p, V_p, Vd_p, Pif_p, k_grid, ef_grid, rf, w, Y, alpha, delta, epsI, epsilon, P_index, Z, r):
    # get grid dimensions
    nZ, nK = Vk_p.shape

    # step 2: calculate Wk(z, k'), W(z, k') for convenience
    Wk, W, Wd = post_decision_vfun(Vk_p, V_p, Vd_p, Pif_p, rf)

    # step 3: obtain k(z, k) and W(z, k) through Wk(z, k') = f(k', k)
    k, W, Wd = foc_capital_no_p_d(Wk - 1, 1 / delta / epsI * (k_grid[:, np.newaxis] / k_grid[np.newaxis, :] - 1),
                            k_grid, nZ, nK, W, Wd)

    # p(z, k)
    p = (epsilon / (epsilon - 1) * w / (1 - alpha) / Z / ef_grid[:, np.newaxis] / k_grid[np.newaxis, :] ** alpha *

         (Y / Z / ef_grid[:, np.newaxis] / k_grid[np.newaxis, :] ** alpha) ** (alpha / (1 - alpha))
         ) ** (1 / (1 + epsilon * alpha / (1 - alpha))) * P_index

    # mc(z, k)
    mc_ind = p / P_index * (epsilon - 1) / epsilon

    # calculation of other variables of interest
    n_ind = ((1 - alpha) * mc_ind * Z * ef_grid[:, np.newaxis] * k_grid[np.newaxis, :] ** alpha / w) ** (1 / alpha)

    y_ind = Z * ef_grid[:, np.newaxis] * k_grid[np.newaxis, :] ** alpha * n_ind ** (1 - alpha)

    k_adj_ind = 1 / delta / epsI / 2 * (k / k_grid[np.newaxis, :] - 1) ** 2 * k_grid[np.newaxis, :]

    i_ind = k - (1 - delta) * k_grid[np.newaxis, :] + k_adj_ind

    div_ind = p / P_index * y_ind - w * n_ind - i_ind

    markup_ind = p / P_index / mc_ind

    q_ind = (k / k_grid[np.newaxis, :] - 1) / (delta * epsI) + 1

    # envelope conditions and update of the value function guess
    Vk = (Z * mc_ind * ef_grid[:, np.newaxis] * alpha * k_grid[np.newaxis, :] ** (alpha - 1) * n_ind ** (1 - alpha) +
          1 - delta - 1 / delta / 2 / epsI * (k / k_grid[np.newaxis, :] - 1) ** 2 +
          1 / delta / epsI * (k / k_grid[np.newaxis, :] - 1) * k / k_grid[np.newaxis, :])

    V = W + Wd
    Vd = div_ind
    V_agg = V

    pp = p ** (1 - epsilon)

    yy_ind = y_ind ** ((epsilon - 1) / epsilon)

    return Vk, V, Vd, p, k, i_ind, div_ind, y_ind, n_ind, mc_ind, markup_ind, pp, yy_ind, k_adj_ind, V_agg, q_ind

@het(exogenous='Pif', policy=['k'], backward=['Vk', 'V', 'Vd'])  # order as in grid!
def firm_no_price_adj_costs_exit(Vk_p, V_p, Vd_p, Pif_p, k_grid, ef_grid, rf, w, Y, alpha, delta, epsI, epsilon, P_index, Z):
    # get grid dimensions
    nZ, nK = Vk_p.shape

    # step 2: calculate Wk(z, k'), W(z, k') for convenience
    Wk, W, Wd = post_decision_vfun(Vk_p, V_p, Vd_p, Pif_p, rf)

    # step 3: obtain k(z, k) and W(z, k) through Wk(z, k') = f(k', k)
    k, W, Wd = foc_capital_no_p_d(Wk - 1, 1 / delta / epsI * (k_grid[:, np.newaxis] / k_grid[np.newaxis, :] - 1),
                            k_grid, nZ, nK, W, Wd)

    # p(z, k)
    p = (epsilon / (epsilon - 1) * w / (1 - alpha) / Z / ef_grid[:, np.newaxis] / k_grid[np.newaxis, :] ** alpha *

         (Y / Z / ef_grid[:, np.newaxis] / k_grid[np.newaxis, :] ** alpha) ** (alpha / (1 - alpha))
         ) ** (1 / (1 + epsilon * alpha / (1 - alpha))) * P_index

    # mc(z, k)
    mc_ind = p / P_index * (epsilon - 1) / epsilon

    # calculation of other variables of interest
    n_ind = ((1 - alpha) * mc_ind * Z * ef_grid[:, np.newaxis] * k_grid[np.newaxis, :] ** alpha / w) ** (1 / alpha)

    y_ind = Z * ef_grid[:, np.newaxis] * k_grid[np.newaxis, :] ** alpha * n_ind ** (1 - alpha)

    k_adj_ind = 1 / delta / epsI / 2 * (k / k_grid[np.newaxis, :] - 1) ** 2 * k_grid[np.newaxis, :]

    i_ind = k - (1 - delta) * k_grid[np.newaxis, :] + k_adj_ind

    div_ind = p / P_index * y_ind - w * n_ind - i_ind

    markup_ind = p / P_index / mc_ind

    q_ind = (k / k_grid[np.newaxis, :] - 1) / (delta * epsI) + 1

    # envelope conditions and update of the value function guess
    Vk = (Z * mc_ind * ef_grid[:, np.newaxis] * alpha * k_grid[np.newaxis, :] ** (alpha - 1) * n_ind ** (1 - alpha) +
          1 - delta - 1 / delta / 2 / epsI * (k / k_grid[np.newaxis, :] - 1) ** 2 +
          1 / delta / epsI * (k / k_grid[np.newaxis, :] - 1) * k / k_grid[np.newaxis, :])

    k_adj_ind1 = 1 / delta / epsI / 2 * (k_grid[0] / k - 1) ** 2 * k

    sell_next = -k_grid[0] + (1 - delta) * k - k_adj_ind1

    #V = W + np.maximum(Wd, sell_next / (1 + rf))
    V = W + np.maximum(Wd, 0)
    Vj = div_ind + V

    k_adj_ind2 = 1 / delta / epsI / 2 * (k_grid[0] / k_grid[np.newaxis, :] - 1) ** 2 * k_grid[np.newaxis, :]

    sell_ind = (-k_grid[0] + (1 - delta) * k_grid[np.newaxis, :] - k_adj_ind2) * np.ones((nZ, nK))

    p[Vj < 0] = 0
    k[Vj < 0] = k_grid[0]
    # p[Vj < sell_ind] = 0
    # k[Vj < sell_ind] = k_grid[0]

    mc_ind = p / P_index * (epsilon - 1) / epsilon

    # calculation of other variables of interest
    n_ind = ((1 - alpha) * mc_ind * Z * ef_grid[:, np.newaxis] * k_grid[np.newaxis, :] ** alpha / w) ** (1 / alpha)

    y_ind = Z * ef_grid[:, np.newaxis] * k_grid[np.newaxis, :] ** alpha * n_ind ** (1 - alpha)

    k_adj_ind = 1 / delta / epsI / 2 * (k / k_grid[np.newaxis, :] - 1) ** 2 * k_grid[np.newaxis, :]

    i_ind = k - (1 - delta) * k_grid[np.newaxis, :] + k_adj_ind

    div_ind = p / P_index * y_ind - w * n_ind - i_ind

    markup_ind = p / P_index / mc_ind

    q_ind = (k / k_grid[np.newaxis, :] - 1) / (delta * epsI) + 1

    # envelope conditions and update of the value function guess
    Vk = (Z * mc_ind * ef_grid[:, np.newaxis] * alpha * k_grid[np.newaxis, :] ** (alpha - 1) * n_ind ** (1 - alpha) +
          1 - delta - 1 / delta / 2 / epsI * (k / k_grid[np.newaxis, :] - 1) ** 2 +
          1 / delta / epsI * (k / k_grid[np.newaxis, :] - 1) * k / k_grid[np.newaxis, :])

    pp = p ** (1 - epsilon)

    yy_ind = y_ind ** ((epsilon - 1) / epsilon)

    Vd = div_ind
    #V[Vj < sell_ind] = 0
    V[Vj < 0] = 0
    V_agg = V

    return Vk, V, Vd, p, k, i_ind, div_ind, y_ind, n_ind, mc_ind, markup_ind, pp, yy_ind, k_adj_ind, V_agg, q_ind


@het(exogenous='Pif', policy=['k'], backward=['Vk', 'V', 'Vd'])  # order as in grid!
def firm_no_price_adj_costs_exit_Bertrand_ss(Vk_p, V_p, Vd_p, Pif_p, k_grid, ef_grid, rf, w, Y, alpha, delta, epsI, epsilon, P_index, Z, D):
    # get grid dimensions
    nZ, nK = Vk_p.shape

    # step 2: calculate Wk(z, k'), W(z, k') for convenience
    Wk, W, Wd = post_decision_vfun(Vk_p, V_p, Vd_p, Pif_p, rf)

    # step 3: obtain k(z, k) and W(z, k) through Wk(z, k') = f(k', k)
    k, W, Wd = foc_capital_no_p_d(Wk - 1, 1 / delta / epsI * (k_grid[:, np.newaxis] / k_grid[np.newaxis, :] - 1),
                            k_grid, nZ, nK, W, Wd)

    # p(z, k) from rhs(p', z, k) = lhs(p', z, k)
    nP = 100
    p_grid = utils.agrid(amax=1.5, n=nP, amin=0.7)
    rhs = (epsilon * (1 - (p_grid[:, np.newaxis, np.newaxis] / P_index) ** (1 - epsilon) * D[np.newaxis, :, :]) / (
            epsilon * (1 - (p_grid[:, np.newaxis, np.newaxis] / P_index) ** (1 - epsilon)) * D[np.newaxis, :, :] - 1) *
           w / (1 - alpha) / Z / ef_grid[np.newaxis, :, np.newaxis] / k_grid[np.newaxis, np.newaxis, :] ** alpha * (
           Y / Z / ef_grid[np.newaxis, :, np.newaxis] / k_grid[np.newaxis, np.newaxis, :] ** alpha) ** (alpha / (1 - alpha))
         ) ** (1 / (1 + epsilon * alpha / (1 - alpha)))
    p = foc_price_RBC_Bertrand(rhs, p_grid[:, np.newaxis, np.newaxis] / P_index,  p_grid, nZ, nP, nK)

    epsilon_modified = epsilon * (1 - (p / P_index) ** (1 - epsilon) * D)

    # mc(z, k)
    mc_ind = p / P_index * (epsilon_modified - 1) / epsilon_modified

    # calculation of other variables of interest
    n_ind = ((1 - alpha) * mc_ind * Z * ef_grid[:, np.newaxis] * k_grid[np.newaxis, :] ** alpha / w) ** (1 / alpha)

    y_ind = Z * ef_grid[:, np.newaxis] * k_grid[np.newaxis, :] ** alpha * n_ind ** (1 - alpha)

    k_adj_ind = 1 / delta / epsI / 2 * (k / k_grid[np.newaxis, :] - 1) ** 2 * k_grid[np.newaxis, :]

    i_ind = k - (1 - delta) * k_grid[np.newaxis, :] + k_adj_ind

    div_ind = p / P_index * y_ind - w * n_ind - i_ind

    markup_ind = p / P_index / mc_ind

    q_ind = (k / k_grid[np.newaxis, :] - 1) / (delta * epsI) + 1

    # envelope conditions and update of the value function guess
    Vk = (Z * mc_ind * ef_grid[:, np.newaxis] * alpha * k_grid[np.newaxis, :] ** (alpha - 1) * n_ind ** (1 - alpha) +
          1 - delta - 1 / delta / 2 / epsI * (k / k_grid[np.newaxis, :] - 1) ** 2 +
          1 / delta / epsI * (k / k_grid[np.newaxis, :] - 1) * k / k_grid[np.newaxis, :])

    k_adj_ind1 = 1 / delta / epsI / 2 * (k_grid[0] / k - 1) ** 2 * k

    sell_next = -k_grid[0] + (1 - delta) * k - k_adj_ind1

    # V = W + np.maximum(Wd, sell_next / (1 + rf))
    V = W + np.maximum(Wd, 0)
    Vj = div_ind + V

    k_adj_ind2 = 1 / delta / epsI / 2 * (k_grid[0] / k_grid[np.newaxis, :] - 1) ** 2 * k_grid[np.newaxis, :]

    sell_ind = (-k_grid[0] + (1 - delta) * k_grid[np.newaxis, :] - k_adj_ind2) * np.ones((nZ, nK))

    p[Vj < 0] = 0
    k[Vj < 0] = k_grid[0]
    # p[Vj < sell_ind] = 0
    # k[Vj < sell_ind] = k_grid[0]

    mc_ind = p / P_index * (epsilon - 1) / epsilon

    # calculation of other variables of interest
    n_ind = ((1 - alpha) * mc_ind * Z * ef_grid[:, np.newaxis] * k_grid[np.newaxis, :] ** alpha / w) ** (1 / alpha)

    y_ind = Z * ef_grid[:, np.newaxis] * k_grid[np.newaxis, :] ** alpha * n_ind ** (1 - alpha)

    k_adj_ind = 1 / delta / epsI / 2 * (k / k_grid[np.newaxis, :] - 1) ** 2 * k_grid[np.newaxis, :]

    i_ind = k - (1 - delta) * k_grid[np.newaxis, :] + k_adj_ind

    div_ind = p / P_index * y_ind - w * n_ind - i_ind

    markup_ind = p / P_index / mc_ind

    q_ind = (k / k_grid[np.newaxis, :] - 1) / (delta * epsI) + 1

    # envelope conditions and update of the value function guess
    Vk = (Z * mc_ind * ef_grid[:, np.newaxis] * alpha * k_grid[np.newaxis, :] ** (alpha - 1) * n_ind ** (1 - alpha) +
          1 - delta - 1 / delta / 2 / epsI * (k / k_grid[np.newaxis, :] - 1) ** 2 +
          1 / delta / epsI * (k / k_grid[np.newaxis, :] - 1) * k / k_grid[np.newaxis, :])

    pp = p ** (1 - epsilon)

    yy_ind = y_ind ** ((epsilon - 1) / epsilon)

    Vd = div_ind
    # V[Vj < sell_ind] = 0
    V[Vj < 0] = 0
    V_agg = V

    return Vk, V, Vd, p, k, i_ind, div_ind, y_ind, n_ind, mc_ind, markup_ind, pp, yy_ind, k_adj_ind, V_agg, q_ind


@het(exogenous='Pif', policy=['k'], backward=['Vk', 'V', 'Vd'])  # order as in grid!
def firm_no_price_adj_costs_exit_Bertrand(Vk_p, V_p, Vd_p, Pif_p, k_grid, ef_grid, rf, w, Y, alpha, delta, epsI, epsilon, P_index, Z, Df):
    # get grid dimensions
    nZ, nK = Vk_p.shape

    # step 2: calculate Wk(z, k'), W(z, k') for convenience
    Wk, W, Wd = post_decision_vfun(Vk_p, V_p, Vd_p, Pif_p, rf)

    # step 3: obtain k(z, k) and W(z, k) through Wk(z, k') = f(k', k)
    k, W, Wd = foc_capital_no_p_d(Wk - 1, 1 / delta / epsI * (k_grid[:, np.newaxis] / k_grid[np.newaxis, :] - 1),
                            k_grid, nZ, nK, W, Wd)

    # p(z, k) from rhs(p', z, k) = lhs(p', z, k)
    nP = 100
    p_grid = utils.agrid(amax=1.5, n=nP, amin=0.7)
    rhs = (epsilon * (1 - (p_grid[:, np.newaxis, np.newaxis] / P_index) ** (1 - epsilon)) / (
            epsilon * (1 - (p_grid[:, np.newaxis, np.newaxis] / P_index) ** (1 - epsilon)) - 1) * w / (1 - alpha) / Z /
           ef_grid[np.newaxis, :, np.newaxis] / k_grid[np.newaxis, np.newaxis, :] ** alpha * (
           Y / Z / ef_grid[np.newaxis, :, np.newaxis] / k_grid[np.newaxis, np.newaxis, :] ** alpha) ** (alpha / (1 - alpha))
         ) ** (1 / (1 + epsilon * alpha / (1 - alpha)))
    p = foc_price_RBC_Bertrand(rhs, p_grid[:, np.newaxis, np.newaxis] / P_index,  p_grid, nZ, nP, nK)

    epsilon_modified = epsilon * (1 - (p / P_index) ** (1 - epsilon))

    # mc(z, k)
    mc_ind = p / P_index * (epsilon_modified - 1) / epsilon_modified

    # calculation of other variables of interest
    n_ind = ((1 - alpha) * mc_ind * Z * ef_grid[:, np.newaxis] * k_grid[np.newaxis, :] ** alpha / w) ** (1 / alpha)

    y_ind = Z * ef_grid[:, np.newaxis] * k_grid[np.newaxis, :] ** alpha * n_ind ** (1 - alpha)

    k_adj_ind = 1 / delta / epsI / 2 * (k / k_grid[np.newaxis, :] - 1) ** 2 * k_grid[np.newaxis, :]

    i_ind = k - (1 - delta) * k_grid[np.newaxis, :] + k_adj_ind

    div_ind = p / P_index * y_ind - w * n_ind - i_ind

    markup_ind = p / P_index / mc_ind

    q_ind = (k / k_grid[np.newaxis, :] - 1) / (delta * epsI) + 1

    # envelope conditions and update of the value function guess
    Vk = (Z * mc_ind * ef_grid[:, np.newaxis] * alpha * k_grid[np.newaxis, :] ** (alpha - 1) * n_ind ** (1 - alpha) +
          1 - delta - 1 / delta / 2 / epsI * (k / k_grid[np.newaxis, :] - 1) ** 2 +
          1 / delta / epsI * (k / k_grid[np.newaxis, :] - 1) * k / k_grid[np.newaxis, :])

    V = W + Wd
    Vd = div_ind
    V_agg = V

    pp = p ** (1 - epsilon)

    yy_ind = y_ind ** ((epsilon - 1) / epsilon)

    return Vk, V, Vd, p, k, i_ind, div_ind, y_ind, n_ind, mc_ind, markup_ind, pp, yy_ind, k_adj_ind, V_agg, q_ind


@het(exogenous='Pif', policy=['k'], backward=['Vk', 'V', 'Vd'])  # order as in grid!
def firm_no_price_adj_costs_open(Vk_p, V_p, Vd_p, Pif_p, k_grid, ef_grid, rf, w, Y, alpha, delta, epsI, epsilon, P_index, Z, p_busket):
    # get grid dimensions
    nZ, nK = Vk_p.shape

    # step 2: calculate Wk(z, k'), W(z, k') for convenience
    Wk, W, Wd = post_decision_vfun(Vk_p, V_p, Vd_p, Pif_p, rf)

    # step 3: obtain k(z, k) and W(z, k) through Wk(z, k') = f(k', k)
    k, W, Wd = foc_capital_no_p_d(Wk - 1, 1 / delta / epsI * (k_grid[:, np.newaxis] / k_grid[np.newaxis, :] - 1),
                            k_grid, nZ, nK, W, Wd)

    # p(z, k)
    p = (epsilon / (epsilon - 1) * w / (1 - alpha) / Z / ef_grid[:, np.newaxis] / k_grid[np.newaxis, :] ** alpha *

         (Y / Z / ef_grid[:, np.newaxis] / k_grid[np.newaxis, :] ** alpha) ** (alpha / (1 - alpha)) *
         P_index ** (epsilon * alpha / (1-alpha)) * p_busket) ** (1 / (1 + epsilon * alpha / (1 - alpha)))

    # mc(z, k)
    mc_ind = p / p_busket * (epsilon - 1) / epsilon

    # calculation of other variables of interest
    n_ind = ((1 - alpha) * mc_ind * Z * ef_grid[:, np.newaxis] * k_grid[np.newaxis, :] ** alpha / w) ** (1 / alpha)

    y_ind = Z * ef_grid[:, np.newaxis] * k_grid[np.newaxis, :] ** alpha * n_ind ** (1 - alpha)

    k_adj_ind = 1 / delta / epsI / 2 * (k / k_grid[np.newaxis, :] - 1) ** 2 * k_grid[np.newaxis, :]

    i_ind = k - (1 - delta) * k_grid[np.newaxis, :] + k_adj_ind

    div_ind = p / p_busket * y_ind - w * n_ind - i_ind

    markup_ind = p / P_index / mc_ind

    q_ind = (k / k_grid[np.newaxis, :] - 1) / (delta * epsI) + 1

    # envelope conditions and update of the value function guess
    Vk = (Z * mc_ind * ef_grid[:, np.newaxis] * alpha * k_grid[np.newaxis, :] ** (alpha - 1) * n_ind ** (1 - alpha) +
          1 - delta - 1 / delta / 2 / epsI * (k / k_grid[np.newaxis, :] - 1) ** 2 +
          1 / delta / epsI * (k / k_grid[np.newaxis, :] - 1) * k / k_grid[np.newaxis, :])

    V = W + Wd
    Vd = div_ind

    pp = p ** (1 - epsilon)

    yy_ind = y_ind ** ((epsilon - 1) / epsilon)

    V_agg = V

    return Vk, V, Vd, p, k, i_ind, div_ind, y_ind, n_ind, mc_ind, markup_ind, pp, yy_ind, k_adj_ind, V_agg, q_ind

def post_decision_vfun(Vp_p, Vk_p, V_p, Pi, rf):
    Wp = (Vp_p.T @ (1 / (1 + rf) * Pi.T)).T
    Wk = (Vk_p.T @ (1 / (1 + rf) * Pi.T)).T
    W = (V_p.T @ (1 / (1 + rf) * Pi.T)).T
    return Wp, Wk, W

def post_decision_vfun_no_p(Vk_p, V_p, Pi, rf):
    Wk = (Vk_p.T @ (1 / (1 + rf) * Pi.T)).T
    W = (V_p.T @ (1 / (1 + rf) * Pi.T)).T
    return Wk, W


@njit
def foc_capital(lhs, rhs, k_grid, nZ, nP, nK, Wp, W):
    k = np.empty((nZ, nP, nK))
    Wp_endo = np.empty((nZ, nP, nK))
    W_endo = np.empty((nZ, nP, nK))
    for iz in range(nZ):
        for ip in range(nP):
            ikp = 0  # use mononicity in k
            for ik in range(nK):
                while True:
                    if lhs[iz, ip, ikp] < rhs[ikp, ik]:
                        break
                    elif ikp < nK - 1:
                        ikp += 1
                    else:
                        break
                if ikp == 0:
                    k[iz, ip, ik] = k_grid[0]
                    Wp_endo[iz, ip, ik] = Wp[iz, ip, 0]
                    W_endo[iz, ip, ik] = W[iz, ip, 0]
                elif ikp == nK:
                    k[iz, ip, ik] = k_grid[ikp]
                    Wp_endo[iz, ip, ik] = Wp[iz, ip, ikp]
                    W_endo[iz, ip, ik] = W[iz, ip, ikp]
                else:
                    y0 = lhs[iz, ip, ikp - 1] - rhs[ikp - 1, ik]
                    y1 = lhs[iz, ip, ikp] - rhs[ikp, ik]
                    k[iz, ip, ik] = k_grid[ikp - 1] - y0 * (k_grid[ikp] - k_grid[ikp - 1]) / (y1 - y0)
                    Wp_endo[iz, ip, ik] = Wp[iz, ip, ikp - 1] + (
                            k[iz, ip, ik] - k_grid[ikp - 1]) * (
                                                   Wp[iz, ip, ikp] - Wp[iz, ip, ikp - 1]) / (
                                                       k_grid[ikp] - k_grid[ikp - 1])
                    W_endo[iz, ip, ik] = W[iz, ip, ikp - 1] + (
                            k[iz, ip, ik] - k_grid[ikp - 1]) * (
                                                  W[iz, ip, ikp] - W[iz, ip, ikp - 1]) / (
                                                  k_grid[ikp] - k_grid[ikp - 1])
    return k, Wp_endo, W_endo

@njit
def foc_capital_d(lhs, rhs, k_grid, nZ, nP, nK, Wp, W, Wd):
    k = np.empty((nZ, nP, nK))
    Wp_endo = np.empty((nZ, nP, nK))
    W_endo = np.empty((nZ, nP, nK))
    Wd_endo = np.empty((nZ, nP, nK))
    for iz in range(nZ):
        for ip in range(nP):
            ikp = 0  # use mononicity in k
            for ik in range(nK):
                while True:
                    if lhs[iz, ip, ikp] < rhs[ikp, ik]:
                        break
                    elif ikp < nK - 1:
                        ikp += 1
                    else:
                        break
                if ikp == 0:
                    k[iz, ip, ik] = k_grid[0]
                    Wp_endo[iz, ip, ik] = Wp[iz, ip, 0]
                    W_endo[iz, ip, ik] = W[iz, ip, 0]
                    Wd_endo[iz, ip, ik] = Wd[iz, ip, 0]
                elif ikp == nK:
                    k[iz, ip, ik] = k_grid[ikp]
                    Wp_endo[iz, ip, ik] = Wp[iz, ip, ikp]
                    W_endo[iz, ip, ik] = W[iz, ip, ikp]
                    Wd_endo[iz, ip, ik] = Wd[iz, ip, ikp]
                else:
                    y0 = lhs[iz, ip, ikp - 1] - rhs[ikp - 1, ik]
                    y1 = lhs[iz, ip, ikp] - rhs[ikp, ik]
                    k[iz, ip, ik] = k_grid[ikp - 1] - y0 * (k_grid[ikp] - k_grid[ikp - 1]) / (y1 - y0)
                    Wp_endo[iz, ip, ik] = Wp[iz, ip, ikp - 1] + (
                            k[iz, ip, ik] - k_grid[ikp - 1]) * (
                                                   Wp[iz, ip, ikp] - Wp[iz, ip, ikp - 1]) / (
                                                       k_grid[ikp] - k_grid[ikp - 1])
                    W_endo[iz, ip, ik] = W[iz, ip, ikp - 1] + (
                            k[iz, ip, ik] - k_grid[ikp - 1]) * (
                                                  W[iz, ip, ikp] - W[iz, ip, ikp - 1]) / (
                                                  k_grid[ikp] - k_grid[ikp - 1])
                    Wd_endo[iz, ip, ik] = Wd[iz, ip, ikp - 1] + (
                            k[iz, ip, ik] - k_grid[ikp - 1]) * (
                                                 Wd[iz, ip, ikp] - Wd[iz, ip, ikp - 1]) / (
                                                 k_grid[ikp] - k_grid[ikp - 1])
    return k, Wp_endo, W_endo, Wd_endo


@njit
def foc_capital_no_p(lhs, rhs, k_grid, nZ, nK, W):
    k = np.empty((nZ, nK))
    W_endo = np.empty((nZ, nK))
    for iz in range(nZ):
        ikp = 0  # use mononicity in k
        for ik in range(nK):
            while True:
                if lhs[iz, ikp] < rhs[ikp, ik]:
                    break
                elif ikp < nK - 1:
                    ikp += 1
                else:
                    break
            if ikp == 0:
                k[iz, ik] = k_grid[0]
                W_endo[iz, ik] = W[iz, 0]
            elif ikp == nK:
                k[iz, ik] = k_grid[ikp]
                W_endo[iz, ik] = W[iz, ikp]
            else:
                y0 = lhs[iz, ikp - 1] - rhs[ikp - 1, ik]
                y1 = lhs[iz, ikp] - rhs[ikp, ik]
                k[iz, ik] = k_grid[ikp - 1] - y0 * (k_grid[ikp] - k_grid[ikp - 1]) / (y1 - y0)
                W_endo[iz, ik] = W[iz, ikp - 1] + (
                        k[iz, ik] - k_grid[ikp - 1]) * (
                                              W[iz, ikp] - W[iz, ikp - 1]) / (
                                              k_grid[ikp] - k_grid[ikp - 1])
    return k, W_endo


@njit
def foc_capital_no_p_d(lhs, rhs, k_grid, nZ, nK, W, Wd):
    k = np.empty((nZ, nK))
    W_endo = np.empty((nZ, nK))
    Wd_endo = np.empty((nZ, nK))
    for iz in range(nZ):
        ikp = 0  # use mononicity in k
        for ik in range(nK):
            while True:
                if lhs[iz, ikp] < rhs[ikp, ik]:
                    break
                elif ikp < nK - 1:
                    ikp += 1
                else:
                    break
            if ikp == 0:
                k[iz, ik] = k_grid[0]
                W_endo[iz, ik] = W[iz, 0]
                Wd_endo[iz, ik] = Wd[iz, 0]
            elif ikp == nK:
                k[iz, ik] = k_grid[ikp]
                W_endo[iz, ik] = W[iz, ikp]
                Wd_endo[iz, ik] = Wd[iz, ikp]
            else:
                y0 = lhs[iz, ikp - 1] - rhs[ikp - 1, ik]
                y1 = lhs[iz, ikp] - rhs[ikp, ik]
                k[iz, ik] = k_grid[ikp - 1] - y0 * (k_grid[ikp] - k_grid[ikp - 1]) / (y1 - y0)
                W_endo[iz, ik] = W[iz, ikp - 1] + (
                        k[iz, ik] - k_grid[ikp - 1]) * (
                                              W[iz, ikp] - W[iz, ikp - 1]) / (
                                              k_grid[ikp] - k_grid[ikp - 1])
                Wd_endo[iz, ik] = Wd[iz, ikp - 1] + (
                        k[iz, ik] - k_grid[ikp - 1]) * (
                                         Wd[iz, ikp] - Wd[iz, ikp - 1]) / (
                                         k_grid[ikp] - k_grid[ikp - 1])
    return k, W_endo, Wd_endo



@njit
def foc_price(lhs, rhs, p_grid, nZ, nP, nK, k_endo, W_endo):
    p = np.empty((nZ, nP, nK))
    k = np.empty((nZ, nP, nK))
    W = np.empty((nZ, nP, nK))
    for iz in range(nZ):
        for ik in range(nK):
            ipp = 0  # use mononicity in p
            for ip in range(nP):
                while True:
                    if lhs[iz, ipp, ik] < rhs[ipp, ip, iz, ik]:
                        break
                    elif ipp < nP - 1:
                        ipp += 1
                    else:
                        break
                if ipp == 0:
                    p[iz, ip, ik] = p_grid[0]
                    k[iz, ip, ik] = k_endo[iz, 0, ik]
                    W[iz, ip, ik] = W_endo[iz, 0, ik]
                elif ipp == nP:
                    p[iz, ip, ik] = p_grid[ipp]
                    k[iz, ip, ik] = k_endo[iz, ipp, ik]
                    W[iz, ip, ik] = W_endo[iz, ipp, ik]
                else:
                    y0 = lhs[iz, ipp - 1, ik] - rhs[ipp - 1, ip, iz, ik]
                    y1 = lhs[iz, ipp, ik] - rhs[ipp, ip, iz, ik]
                    p[iz, ip, ik] = p_grid[ipp - 1] - y0 * (p_grid[ipp] - p_grid[ipp - 1]) / (y1 - y0)
                    k[iz, ip, ik] = k_endo[iz, ipp - 1, ik] + (
                            p[iz, ip, ik] - p_grid[ipp - 1]) * (
                                                   k_endo[iz, ipp, ik] - k_endo[iz, ipp - 1, ik]) / (
                                                       p_grid[ipp] - p_grid[ipp - 1])
                    W[iz, ip, ik] = W_endo[iz, ipp - 1, ik] + (
                            p[iz, ip, ik] - p_grid[ipp - 1]) * (
                                            W_endo[iz, ipp, ik] - W_endo[iz, ipp - 1, ik]) / (
                                            p_grid[ipp] - p_grid[ipp - 1])
    return p, k, W


@njit
def foc_price_d(lhs, rhs, p_grid, nZ, nP, nK, k_endo, W_endo, Wd_endo):
    p = np.empty((nZ, nP, nK))
    k = np.empty((nZ, nP, nK))
    W = np.empty((nZ, nP, nK))
    Wd = np.empty((nZ, nP, nK))
    for iz in range(nZ):
        for ik in range(nK):
            ipp = 0  # use mononicity in p
            for ip in range(nP):
                while True:
                    if lhs[iz, ipp, ik] < rhs[ipp, ip, iz, ik]:
                        break
                    elif ipp < nP - 1:
                        ipp += 1
                    else:
                        break
                if ipp == 0:
                    p[iz, ip, ik] = p_grid[0]
                    k[iz, ip, ik] = k_endo[iz, 0, ik]
                    W[iz, ip, ik] = W_endo[iz, 0, ik]
                    Wd[iz, ip, ik] = Wd_endo[iz, 0, ik]
                elif ipp == nP:
                    p[iz, ip, ik] = p_grid[ipp]
                    k[iz, ip, ik] = k_endo[iz, ipp, ik]
                    W[iz, ip, ik] = W_endo[iz, ipp, ik]
                    Wd[iz, ip, ik] = Wd_endo[iz, ipp, ik]
                else:
                    y0 = lhs[iz, ipp - 1, ik] - rhs[ipp - 1, ip, iz, ik]
                    y1 = lhs[iz, ipp, ik] - rhs[ipp, ip, iz, ik]
                    p[iz, ip, ik] = p_grid[ipp - 1] - y0 * (p_grid[ipp] - p_grid[ipp - 1]) / (y1 - y0)
                    k[iz, ip, ik] = k_endo[iz, ipp - 1, ik] + (
                            p[iz, ip, ik] - p_grid[ipp - 1]) * (
                                                   k_endo[iz, ipp, ik] - k_endo[iz, ipp - 1, ik]) / (
                                                       p_grid[ipp] - p_grid[ipp - 1])
                    W[iz, ip, ik] = W_endo[iz, ipp - 1, ik] + (
                            p[iz, ip, ik] - p_grid[ipp - 1]) * (
                                            W_endo[iz, ipp, ik] - W_endo[iz, ipp - 1, ik]) / (
                                            p_grid[ipp] - p_grid[ipp - 1])
                    Wd[iz, ip, ik] = Wd_endo[iz, ipp - 1, ik] + (
                            p[iz, ip, ik] - p_grid[ipp - 1]) * (
                                            Wd_endo[iz, ipp, ik] - Wd_endo[iz, ipp - 1, ik]) / (
                                            p_grid[ipp] - p_grid[ipp - 1])
    return p, k, W, Wd

@njit
def foc_price_RBC_Bertrand(rhs, lhs, p_grid, nZ, nP, nK):
    p = np.empty((nZ, nK))
    for iz in range(nZ):
        for ik in range(nK):
            ipp = 0  # use mononicity in p
            for ip in range(nP, 0, -1):
                while True:
                    if lhs[ipp, iz, ik] < rhs[ipp, iz, ik]:
                        break
                    elif ipp > 0:
                        ipp -= 1
                    else:
                        break
                if ipp == 0:
                    p[iz, ik] = p_grid[0]
                elif ipp == nP:
                    p[iz, ik] = p_grid[ipp]
                else:
                    y0 = lhs[ipp - 1, iz, ik] - rhs[ipp - 1, iz, ik]
                    y1 = lhs[ipp, iz, ik] - rhs[ipp, iz, ik]
                    p[iz, ik] = p_grid[ipp - 1] - y0 * (p_grid[ipp] - p_grid[ipp - 1]) / (y1 - y0)
    return p
