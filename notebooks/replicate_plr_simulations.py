"""Script-style notebook alternative: run small simulation and save summary.

This file can be executed as a script if a full notebook isn't required.
"""
from dmlkappa.simulation import run_plr_simulation_grid

def main():
    n_list = [200]
    overlap_list = ["moderate"]
    rho_list = [0.5]
    results, summary = run_plr_simulation_grid(n_list, overlap_list, rho_list, n_rep=3, random_state=1)
    print(summary)

if __name__ == "__main__":
    main()
