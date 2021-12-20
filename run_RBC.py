import numpy as np
import matplotlib.pyplot as plt

import utils
import jacobian as jac
from het_block import het
from simple_block import simple
from solved_block import solved
import steady_state_no_price_adj_costs_closed
import steady_state_RBC
import determinacy as det
import nonlinear
import het_firm

ss = steady_state_RBC.hank_ss(noisy=True)

from dynamics_RBC import arbitrage, firm_redefinition, share_value, \
              future_interest, fiscal, finance, union, mkt_clearing, pricing_het, markup


T = 700
block_list = [steady_state_RBC.household_inc, arbitrage, firm_redefinition, share_value,
              future_interest, fiscal, finance, union, mkt_clearing, pricing_het, markup]

exogenous = ['rstar', 'Z', 'G', 'mup']
unknowns = ['r', 'w', 'Y']
targets = ['asset_mkt', 'wnkpc', 'fisher']

# Representative household
# block_list = [RANK_hh, arbitrage, arbitrage, firm_redefinition, share_value,
#               future_interest, fiscal, finance, union, mkt_clearing, pricing_het, markup]
# unknowns = ['r', 'w', 'Y', 'C', 'A', 'B']
# targets = ['goods_mkt', 'wnkpc', 'fisher', 'euler_eq', 'nonliq_asset_eq', 'budget_const']

A = jac.get_H_U(block_list, unknowns, targets, T, ss, asymptotic=True, save=False)
wn = det.winding_criterion(A)
print(f'Winding number: {wn}')


G = jac.get_G(block_list, exogenous, unknowns, targets, T=T, ss=ss, use_saved=False)


