# heterogeneous_firms #
Heterogeneous firms (monopolistic, endogeneous exit, Bertrand competition, RBC/NK)

# Structure of the code #
## Naming conventions ##
- "run_...py": this file runs linear solutions of the models
- "dynamamics_...py": these files contain all dynamic equations needed for the models except equations for households and firms
- "steady_state...py": these files solve for the steady states and contain households problems for HA models
- "het_firm.py": this file contains heterogeneous firms problems

## Available models ##
- Heterogeneous firms with monopolistic competition, with endogeneous exit, with Bertrand competition. All models are available as RBC and as NK versions.

## Figures ##
All codes for figures could be found in "figures.py".

# Source code #
The solution method and codes are taken from Auclert, Bardóczy, Rognlie, Straub (2021): "Using the Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models" and their github folder https://github.com/shade-econ/sequence-jacobian

# Comments #
To choose between different heterogeneous firms' problems change function name (from firm to firm_exit, for example) in three places: two times in the "steady_state...py" file where heterogeneous firms' problem is solved and one time in "dynamics...py" file in "pricing_het" block definition.


