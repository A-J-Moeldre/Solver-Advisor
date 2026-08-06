import argparse
from solver_advisor.run import run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix")
    args = parser.parse_args()

    A, result = run(args.matrix)

    print("Form:", A.shape)
    print("Symmetrie:", result["symmetric"])
    print("SPD:", result["spd"])
    print("Blockstruktur:", result["has_blocks"])
    print("Kondition:", result["kappa"])
    print("Solver:", result["solver"])
    print("Preconditioner:", result["preconditioner"])

