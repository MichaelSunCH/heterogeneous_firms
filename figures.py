T = 700
# %% markup shock
rhos = np.array([0.7])
dmup = 0.01 * ss['mup'] * rhos ** (np.arange(T)[:, np.newaxis])
dY1 = 100 * G_linear['Y']['mup'] @ dmup
dI1 = 100 * G_linear['I_IND']['mup'] @ dmup
dC1 = 100 * G_linear['C']['mup'] @ dmup
di1 = np.zeros(T)
#di1 = 100 * G_linear['i']['mup'] @ dmup
dw1 = 100 * G_linear['w']['mup'] @ dmup
dN1 = 100 * G_linear['N']['mup'] @ dmup
dK1 = 100 * G_linear['K']['mup'] @ dmup
dmc1 = 100 * G_linear['MC_IND']['mup'] @ dmup
dpi1 = np.zeros(T)
#dpi1 = 100 * (G_linear['pi']['mup'] @ dmup) * (ss['pi'] + 1)

dA1 = 100 * G_linear['A']['mup'] @ dmup
dB1 = 100 * G_linear['B']['mup'] @ dmup
dWealth1 = 100 * (dA1 / 100 * ss['A'] + dB1 / 100 * ss['B']) / (ss['A'] + ss['B'])
dp_equity1 = 100 * G_linear['V_AGG']['mup'] @ dmup

dr1 = 100 * G_linear['r']['mup'] @ dmup
ddiv1 = 100 * G_linear['DIV_IND']['mup'] @ dmup

rhos = np.array([0.7])
dmup = 0.01 * ss2['mup'] * rhos ** (np.arange(T)[:, np.newaxis])

dY2 = 100 * G_linear_2['Y']['mup'] @ dmup
dI2 = 100 * G_linear_2['I_IND']['mup'] @ dmup
dC2 = 100 * G_linear_2['C']['mup'] @ dmup
di2 = np.zeros(T)
#di2 = 100 * G_linear_2['i']['mup'] @ dmup
dw2 = 100 * G_linear_2['w']['mup'] @ dmup
dN2 = 100 * G_linear_2['N']['mup'] @ dmup
dK2 = 100 * G_linear_2['K']['mup'] @ dmup
dmc2 = 100 * G_linear_2['MC_IND']['mup'] @ dmup
dpi2 = np.zeros(T)
#dpi2 = 100 * (G_linear_2['pi']['mup'] @ dmup) * (ss2['pi'] + 1)

dA2 = 100 * G_linear_2['A']['mup'] @ dmup
dB2 = 100 * G_linear_2['B']['mup'] @ dmup
dWealth2 = 100 * (dA2 / 100 * ss2['A'] + dB2 / 100 * ss2['B']) / (ss2['A'] + ss2['B'])
dp_equity2 = 100 * G_linear_2['V_AGG']['mup'] @ dmup

dr2 = 100 * G_linear_2['r']['mup'] @ dmup
ddiv2 = 100 * G_linear_2['DIV_IND']['mup'] @ dmup

# plot
fig, axs = plt.subplots(3, 4, figsize=(16, 16))
axs[0, 0].plot(dY1[:20], label='HH, HF, RBC', linestyle='-', linewidth=2)
axs[0, 0].plot(dY2[:20], label='HH, HF, RBC, endogeneous exit', linestyle='-', linewidth=2)
axs[0, 0].set_title('GDP')
axs[0, 1].plot(dr1[:20], linestyle='-', linewidth=2)
axs[0, 1].plot(dr2[:20], linestyle='-', linewidth=2)
axs[0, 1].set_title('Real Rate')
axs[0, 2].plot(dC1[:20], linestyle='-', linewidth=2)
axs[0, 2].plot(dC2[:20], linestyle='-', linewidth=2)
axs[0, 2].set_title('Consumption')
axs[0, 3].plot(di1[:20], linestyle='-', linewidth=2)
axs[0, 3].plot(di2[:20], linestyle='-', linewidth=2)
axs[0, 3].set_title("Central Bank's policy rate")
axs[1, 0].plot(dw1[:20], linestyle='-', linewidth=2)
axs[1, 0].plot(dw2[:20], linestyle='-', linewidth=2)
axs[1, 0].set_title('Wages')
axs[1, 1].plot(dN1[:20], linestyle='-', linewidth=2)
axs[1, 1].plot(dN2[:20], linestyle='-', linewidth=2)
axs[1, 1].set_title('Labor hours')
axs[1, 2].plot(dp_equity1[:20], linestyle='-', linewidth=2)
axs[1, 2].plot(dp_equity2[:20], linestyle='-', linewidth=2)
axs[1, 2].set_title('Price of equity')
axs[1, 3].plot(100 * dmup[:20, 0] / ss['mup'], linestyle='-', linewidth=2)
axs[1, 3].set_title('Markup shock')
axs[2, 0].plot(dK1[:20], linestyle='-', linewidth=2)
axs[2, 0].plot(dK2[:20], linestyle='-', linewidth=2)
axs[2, 0].set_title('Capital')

axs[2, 3].plot(ddiv1[:20], linestyle='-', linewidth=2)
axs[2, 3].plot(ddiv2[:20], linestyle='-', linewidth=2)
axs[2, 3].set_title('Dividends')

axs[2, 2].plot(dA1[:20], linestyle='-', linewidth=2)
axs[2, 2].plot(dA2[:20], linestyle='-', linewidth=2)
axs[2, 2].set_title('Non-liquid assets')
axs[2, 1].plot(dB1[:20], linestyle='-', linewidth=2)
axs[2, 1].plot(dB2[:20], linestyle='-', linewidth=2)
axs[2, 1].set_title('Liquid assets')
axs[0, 0].legend(bbox_to_anchor=(3.5, -2.75), loc='lower right', ncol=2)
plt.show()

# %% TFP comparison
rhos = np.array([0.7])
dZ = 0.01 * ss['Z'] * rhos ** (np.arange(T)[:, np.newaxis])
dY1 = 100 * G_linear['Y']['Z'] @ dZ
dI1 = 100 * G_linear['I_IND']['Z'] @ dZ
dC1 = 100 * G_linear['C']['Z'] @ dZ
#di1 = 100 * G_linear['i']['Z'] @ dZ
di1 = np.zeros(T)
dw1 = 100 * G_linear['w']['Z'] @ dZ
dN1 = 100 * G_linear['N']['Z'] @ dZ
dK1 = 100 * G_linear['K']['Z'] @ dZ
dmc1 = 100 * G_linear['MC_IND']['Z'] @ dZ
dpi1 = np.zeros(T)
#dpi1 = 100 * (G_linear['pi']['Z'] @ dZ) * (ss['pi'] + 1)

dA1 = 100 * G_linear['A']['Z'] @ dZ
dB1 = 100 * G_linear['B']['Z'] @ dZ
dWealth1 = 100 * (dA1 / 100 * ss['A'] + dB1 / 100 * ss['B']) / (ss['A'] + ss['B'])
dp_equity1 = 100 * G_linear['V_AGG']['Z'] @ dZ

dr1 = 100 * G_linear['r']['Z'] @ dZ
ddiv1 = 100 * G_linear['DIV_IND']['Z'] @ dZ

rhos = np.array([0.7])
dZ = 0.01 * ss['Z'] * rhos ** (np.arange(T)[:, np.newaxis])
dY2 = 100 * G_linear_2['Y']['Z'] @ dZ
dI2 = 100 * G_linear_2['I_IND']['Z'] @ dZ
dC2 = 100 * G_linear_2['C']['Z'] @ dZ
#di2 = 100 * G_linear_2['i']['Z'] @ dZ
di2 = np.zeros(T)
dw2 = 100 * G_linear_2['w']['Z'] @ dZ
dN2 = 100 * G_linear_2['N']['Z'] @ dZ
dK2 = 100 * G_linear_2['K']['Z'] @ dZ
dmc2 = 100 * G_linear_2['MC_IND']['Z'] @ dZ
dpi2 = np.zeros(T)
#dpi2 = 100 * (G_linear_2['pi']['Z'] @ dZ) * (ss2['pi'] + 1)

dA2 = 100 * G_linear_2['A']['Z'] @ dZ
dB2 = 100 * G_linear_2['B']['Z'] @ dZ
dWealth2 = 100 * (dA2 / 100 * ss2['A'] + dB2 / 100 * ss2['B']) / (ss2['A'] + ss2['B'])
dp_equity2 = 100 * G_linear_2['V_AGG']['Z'] @ dZ

dr2 = 100 * G_linear_2['r']['Z'] @ dZ
ddiv2 = 100 * G_linear_2['DIV_IND']['Z'] @ dZ


# plot
fig, axs = plt.subplots(3, 4, figsize=(16, 16))
axs[0, 0].plot(dY1[:20], label='HH, HF, RBC', linestyle='-', linewidth=2)
axs[0, 0].plot(dY2[:20], label='HH, HF, RBC, endogeneous exit', linestyle='-', linewidth=2)
axs[0, 0].set_title('GDP')
axs[0, 1].plot(dr1[:20], linestyle='-', linewidth=2)
axs[0, 1].plot(dr2[:20], linestyle='-', linewidth=2)
axs[0, 1].set_title('Real Rate')
axs[0, 2].plot(dC1[:20], linestyle='-', linewidth=2)
axs[0, 2].plot(dC2[:20], linestyle='-', linewidth=2)
axs[0, 2].set_title('Consumption')
axs[0, 3].plot(di1[:20], linestyle='-', linewidth=2)
axs[0, 3].plot(di2[:20], linestyle='-', linewidth=2)
axs[0, 3].set_title("Central Bank's policy rate")
axs[1, 0].plot(dw1[:20], linestyle='-', linewidth=2)
axs[1, 0].plot(dw2[:20], linestyle='-', linewidth=2)
axs[1, 0].set_title('Wages')
axs[1, 1].plot(dN1[:20], linestyle='-', linewidth=2)
axs[1, 1].plot(dN2[:20], linestyle='-', linewidth=2)
axs[1, 1].set_title('Labor hours')
axs[1, 2].plot(dp_equity1[:20], linestyle='-', linewidth=2)
axs[1, 2].plot(dp_equity2[:20], linestyle='-', linewidth=2)
axs[1, 2].set_title('Price of equity')
axs[1, 3].plot(100 * dZ[:50, 0] / ss['Z'], linestyle='-', linewidth=2)
axs[1, 3].set_title('Aggregate TFP shock')
axs[2, 0].plot(dK1[:20], linestyle='-', linewidth=2)
axs[2, 0].plot(dK2[:20], linestyle='-', linewidth=2)
axs[2, 0].set_title('Capital')

axs[2, 3].plot(ddiv1[:20], linestyle='-', linewidth=2)
axs[2, 3].plot(ddiv2[:20], linestyle='-', linewidth=2)
axs[2, 3].set_title('Dividends')

axs[2, 2].plot(dA1[:20], linestyle='-', linewidth=2)
axs[2, 2].plot(dA2[:20], linestyle='-', linewidth=2)
axs[2, 2].set_title('Non-liquid assets')
axs[2, 1].plot(dB1[:20], linestyle='-', linewidth=2)
axs[2, 1].plot(dB2[:20], linestyle='-', linewidth=2)
axs[2, 1].set_title('Liquid assets')
axs[0, 0].legend(bbox_to_anchor=(3.0, -2.75), loc='lower right', ncol=2)
plt.show()

# %% government spending comparison
rhos = np.array([0.7])
dG = 0.01 * ss['G'] * rhos ** (np.arange(T)[:, np.newaxis])
dY1 = 100 * G_linear['Y']['G'] @ dG
dI1 = 100 * G_linear['I_IND']['G'] @ dG
dC1 = 100 * G_linear['C']['G'] @ dG
#di1 = 100 * G_linear['i']['G'] @ dG
di1 = np.zeros(T)
dw1 = 100 * G_linear['w']['G'] @ dG
dN1 = 100 * G_linear['N']['G'] @ dG
dK1 = 100 * G_linear['K']['G'] @ dG
dmc1 = 100 * G_linear['MC_IND']['G'] @ dG
#dpi1 = 100 * (G_linear['pi']['G'] @ dG) * (ss['pi'] + 1)
dpi1 = np.zeros(T)

dA1 = 100 * G_linear['A']['G'] @ dG
dB1 = 100 * G_linear['B']['G'] @ dG
dWealth1 = 100 * (dA1 / 100 * ss['A'] + dB1 / 100 * ss['B']) / (ss['A'] + ss['B'])
dp_equity1 = 100 * G_linear['V_AGG']['G'] @ dG

dr1 = 100 * G_linear['r']['G'] @ dG
ddiv1 = 100 * G_linear['DIV_IND']['G'] @ dG

rhos = np.array([0.7])
dG = 0.01 * ss['G'] * rhos ** (np.arange(T)[:, np.newaxis])
dY2 = 100 * G_linear_2['Y']['G'] @ dG
dI2 = 100 * G_linear_2['I_IND']['G'] @ dG
dC2 = 100 * G_linear_2['C']['G'] @ dG
#di2 = 100 * G_linear_2['i']['G'] @ dG
di2 = np.zeros(T)
dw2 = 100 * G_linear_2['w']['G'] @ dG
dN2 = 100 * G_linear_2['N']['G'] @ dG
dK2 = 100 * G_linear_2['K']['G'] @ dG
dmc2 = 100 * G_linear_2['MC_IND']['G'] @ dG
#dpi2 = 100 * (G_linear_2['pi']['G'] @ dG) * (ss2['pi'] + 1)
dpi2 = np.zeros(T)

dA2 = 100 * G_linear_2['A']['G'] @ dG
dB2 = 100 * G_linear_2['B']['G'] @ dG
dWealth2 = 100 * (dA2 / 100 * ss2['A'] + dB2 / 100 * ss2['B']) / (ss2['A'] + ss2['B'])
dp_equity2 = 100 * G_linear_2['V_AGG']['G'] @ dG

dr2 = 100 * G_linear_2['r']['G'] @ dG
ddiv2 = 100 * G_linear_2['DIV_IND']['G'] @ dG

# plot
fig, axs = plt.subplots(3, 4, figsize=(16, 16))
axs[0, 0].plot(dY1[:20], label='HH, HF, RBC', linestyle='-', linewidth=2)
axs[0, 0].plot(dY2[:20], label='HH, HF, RBC, endogeneous exit', linestyle='-', linewidth=2)
axs[0, 0].set_title('GDP')
axs[0, 1].plot(dr1[:20], linestyle='-', linewidth=2)
axs[0, 1].plot(dr2[:20], linestyle='-', linewidth=2)
axs[0, 1].set_title('Real Rate')
axs[0, 2].plot(dC1[:20], linestyle='-', linewidth=2)
axs[0, 2].plot(dC2[:20], linestyle='-', linewidth=2)
axs[0, 2].set_title('Consumption')
axs[0, 3].plot(di1[:20], linestyle='-', linewidth=2)
axs[0, 3].plot(di2[:20], linestyle='-', linewidth=2)
axs[0, 3].set_title("Central Bank's policy rate")
axs[1, 0].plot(dw1[:20], linestyle='-', linewidth=2)
axs[1, 0].plot(dw2[:20], linestyle='-', linewidth=2)
axs[1, 0].set_title('Wages')
axs[1, 1].plot(dN1[:20], linestyle='-', linewidth=2)
axs[1, 1].plot(dN2[:20], linestyle='-', linewidth=2)
axs[1, 1].set_title('Labor hours')
axs[1, 2].plot(dp_equity1[:20], linestyle='-', linewidth=2)
axs[1, 2].plot(dp_equity2[:20], linestyle='-', linewidth=2)
axs[1, 2].set_title('Price of equity')
axs[1, 3].plot(100 * dG[:50, 0] / ss['G'], linestyle='-', linewidth=2)
axs[1, 3].set_title('Government spending shock')
axs[2, 0].plot(dK1[:20], linestyle='-', linewidth=2)
axs[2, 0].plot(dK2[:20], linestyle='-', linewidth=2)
axs[2, 0].set_title('Capital')

axs[2, 3].plot(ddiv1[:20], linestyle='-', linewidth=2)
axs[2, 3].plot(ddiv2[:20], linestyle='-', linewidth=2)
axs[2, 3].set_title('Dividends')

axs[2, 2].plot(dA1[:20], linestyle='-', linewidth=2)
axs[2, 2].plot(dA2[:20], linestyle='-', linewidth=2)
axs[2, 2].set_title('Non-liquid assets')
axs[2, 1].plot(dB1[:20], linestyle='-', linewidth=2)
axs[2, 1].plot(dB2[:20], linestyle='-', linewidth=2)
axs[2, 1].set_title('Liquid assets')
axs[0, 0].legend(bbox_to_anchor=(3.0, -2.75), loc='lower right', ncol=2)
plt.show()


# %% Monetary policy comparison
rhos = np.array([0.7])
drstar = 0.0025 * rhos ** (np.arange(T)[:, np.newaxis])
dY1 = 100 * G_linear['Y']['rstar'] @ drstar
dI1 = 100 * G_linear['I_IND']['rstar'] @ drstar
dC1 = 100 * G_linear['C']['rstar'] @ drstar
di1 = 100 * G_linear['i']['rstar'] @ drstar
dw1 = 100 * G_linear['w']['rstar'] @ drstar
dN1 = 100 * G_linear['N']['rstar'] @ drstar
dK1 = 100 * G_linear['K']['rstar'] @ drstar
dmc1 = 100 * G_linear['MC_IND']['rstar'] @ drstar
dpi1 = 100 * (G_linear['pi']['rstar'] @ drstar) * (ss['pi'] + 1)

dA1 = 100 * G_linear['A']['rstar'] @ drstar
dB1 = 100 * G_linear['B']['rstar'] @ drstar
dWealth1 = 100 * (dA1 / 100 * ss['A'] + dB1 / 100 * ss['B']) / (ss['A'] + ss['B'])
dp_equity1 = 100 * G_linear['V_AGG']['rstar'] @ drstar

dr1 = 100 * G_linear['r']['rstar'] @ drstar
ddiv1 = 100 * G_linear['DIV_IND']['rstar'] @ drstar

rhos = np.array([0.7])
drstar = 0.0025 * rhos ** (np.arange(T)[:, np.newaxis])
dY2 = 100 * G_linear_2['Y']['rstar'] @ drstar
dI2 = 100 * G_linear_2['I_IND']['rstar'] @ drstar
dC2 = 100 * G_linear_2['C']['rstar'] @ drstar
di2 = 100 * G_linear_2['i']['rstar'] @ drstar
dw2 = 100 * G_linear_2['w']['rstar'] @ drstar
dN2 = 100 * G_linear_2['N']['rstar'] @ drstar
dK2 = 100 * G_linear_2['K']['rstar'] @ drstar
dmc2 = 100 * G_linear_2['MC_IND']['rstar'] @ drstar
dpi2 = 100 * (G_linear_2['pi']['rstar'] @ drstar) * (ss2['pi'] + 1)


dA2 = 100 * G_linear_2['A']['rstar'] @ drstar
dB2 = 100 * G_linear_2['B']['rstar'] @ drstar
dWealth2 = 100 * (dA2 / 100 * ss2['A'] + dB2 / 100 * ss2['B']) / (ss2['A'] + ss2['B'])
dp_equity2 = 100 * G_linear_2['V_AGG']['rstar'] @ drstar

dr2 = 100 * G_linear_2['r']['rstar'] @ drstar
ddiv2 = 100 * G_linear_2['DIV_IND']['rstar'] @ drstar


# plot
fig, axs = plt.subplots(3, 4, figsize=(16, 16))
axs[0, 0].plot(dY1[:20], label='HH, HF, NK', linestyle='-', linewidth=2)
axs[0, 0].plot(dY2[:20], label='HH, HF, NK, Bertrand', linestyle='-', linewidth=2)
axs[0, 0].set_title('GDP')
axs[0, 1].plot(dr1[:20], linestyle='-', linewidth=2)
axs[0, 1].plot(dr2[:20], linestyle='-', linewidth=2)
axs[0, 1].set_title('Real Rate')
axs[0, 2].plot(dC1[:20], linestyle='-', linewidth=2)
axs[0, 2].plot(dC2[:20], linestyle='-', linewidth=2)
axs[0, 2].set_title('Consumption')
axs[0, 3].plot(di1[:20], linestyle='-', linewidth=2)
axs[0, 3].plot(di2[:20], linestyle='-', linewidth=2)
axs[0, 3].set_title("Central Bank's policy rate")
axs[1, 0].plot(dw1[:20], linestyle='-', linewidth=2)
axs[1, 0].plot(dw2[:20], linestyle='-', linewidth=2)
axs[1, 0].set_title('Wages')
axs[1, 1].plot(dN1[:20], linestyle='-', linewidth=2)
axs[1, 1].plot(dN2[:20], linestyle='-', linewidth=2)
axs[1, 1].set_title('Labor hours')
axs[1, 2].plot(dp_equity1[:20], linestyle='-', linewidth=2)
axs[1, 2].plot(dp_equity2[:20], linestyle='-', linewidth=2)
axs[1, 2].set_title('Price of equity')
axs[1, 3].plot(100 * drstar[:50, 0], linestyle='-', linewidth=2)
axs[1, 3].set_title('Monetary policy shock')
axs[2, 0].plot(dK1[:20], linestyle='-', linewidth=2)
axs[2, 0].plot(dK2[:20], linestyle='-', linewidth=2)
axs[2, 0].set_title('Capital')

axs[2, 3].plot(ddiv1[:20], linestyle='-', linewidth=2)
axs[2, 3].plot(ddiv2[:20], linestyle='-', linewidth=2)
axs[2, 3].set_title('Dividends')

axs[2, 2].plot(dA1[:20], linestyle='-', linewidth=2)
axs[2, 2].plot(dA2[:20], linestyle='-', linewidth=2)
axs[2, 2].set_title('Non-liquid assets')
axs[2, 1].plot(dB1[:20], linestyle='-', linewidth=2)
axs[2, 1].plot(dB2[:20], linestyle='-', linewidth=2)
axs[2, 1].set_title('Liquid assets')
axs[0, 0].legend(bbox_to_anchor=(3.0, -2.75), loc='lower right', ncol=2)
plt.show()

