import numpy as np
import matplotlib.pyplot as plt

import utils
import jacobian as jac
from het_block import het
from simple_block import simple
from solved_block import solved
import steady_state_NK
import determinacy as det
import nonlinear
import het_firm

ss = steady_state_NK.hank_ss(noisy=True)

from dynamics_NK import arbitrage, firm_redefinition, share_value, pricing_het, markup, \
              future_interest, taylor, fiscal, finance, wage, union, mkt_clearing

T = 700
block_list = [steady_state_as_RF.household_inc, arbitrage, firm_redefinition, share_value,
              future_interest, taylor, fiscal, finance, wage, union, mkt_clearing, pricing_het, checks, markup]
exogenous = ['rstar', 'Z', 'G', 'mup']
unknowns = ['r', 'w', 'Y']
targets = ['asset_mkt', 'wnkpc', 'fisher']

A = jac.get_H_U(block_list, unknowns, targets, T, ss, asymptotic=True, save=False)
wn = det.winding_criterion(A)
print(f'Winding number: {wn}')

G = jac.get_G(block_list, exogenous, unknowns, targets, T=T, ss=ss, use_saved=False)

