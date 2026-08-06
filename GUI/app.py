import tkinter as tk
from tkinter import scrolledtext, ttk
import os

# ------------------------------------------------------------
# Import backend (clean, modular)
# ------------------------------------------------------------

from solver_advisor.io import load_matrix
from solver_advisor.diagnostics import solver_recommendation

# ------------------------------------------------------------
# Colors
# ------------------------------------------------------------

COLOR_GOOD = "#4CAF50"
COLOR_WARN = "#FFC107"
COLOR_BAD  = "#F44336"
COLOR_INFO = "#2196F3"

# ------------------------------------------------------------
# Find available matrices
# ------------------------------------------------------------

MATRIX_FOLDER = os.path.join(os.path.dirname(__file__), "..", "matrices")

available_matrices = []

for root, dirs, files in os.walk(MATRIX_FOLDER):
    for file in files:
        if file.endswith(".mtx"):
            available_matrices.append(os.path.join(root, file))

# ------------------------------------------------------------
# GUI Functions
# ------------------------------------------------------------

def analyze_matrix():
    file_path = matrix_choice.get()

    if not file_path:
        return

    try:
        A = load_matrix(file_path)
        result = solver_recommendation(A)
        show_result(result)

    except Exception as e:
        output_box.delete("1.0", tk.END)
        output_box.insert(tk.END, f"Error loading matrix:\n{e}", "bad")

def show_result(result):
    output_box.delete("1.0", tk.END)

    # Symmetry
    output_box.insert(
        tk.END,
        f"Symmetry: {'Yes' if result['symmetric'] else 'No'}\n",
        "good" if result["symmetric"] else "bad"
    )

    # SPD
    output_box.insert(
        tk.END,
        f"SPD: {'Yes' if result['spd'] else 'No'}\n",
        "good" if result["spd"] else "warn"
    )

    # Blocks
    if result["has_blocks"]:
        output_box.insert(tk.END, "Block structure: Yes\n", "info")
        output_box.insert(tk.END, f"Clusters: {result['clusters']}\n", "info")
    else:
        output_box.insert(tk.END, "Block structure: No\n", "info")

    # Condition number
    output_box.insert(
        tk.END,
        f"\nCondition number estimate: {result['kappa']}\n",
        "warn"
    )

    # Eigenvalues
    output_box.insert(
        tk.END,
        f"\nEigenvalues:\n{result['eigvals']}\n",
        "info"
    )

    # Solver
    output_box.insert(
        tk.END,
        f"\nRecommended solver: {result['solver']}\n",
        "good"
    )

    # Preconditioner
    output_box.insert(
        tk.END,
        f"Recommended preconditioner: {result['preconditioner']}\n",
        "good"
    )

# ------------------------------------------------------------
# Create GUI
# ------------------------------------------------------------

root = tk.Tk()
root.title("Solver Advisor")
root.geometry("700x550")

# Matrix dropdown
matrix_choice = ttk.Combobox(root, values=available_matrices, width=80)
matrix_choice.pack(pady=10)

if available_matrices:
    matrix_choice.current(0)

# Analyze button
analyze_button = tk.Button(
    root,
    text="Analyze Matrix",
    command=analyze_matrix,
    bg="#2196F3",
    fg="white",
    width=20
)
analyze_button.pack(pady=10)

# Output box
output_box = scrolledtext.ScrolledText(root, width=80, height=25)
output_box.pack(padx=10, pady=10)

# Text colors
output_box.tag_config("good", foreground=COLOR_GOOD)
output_box.tag_config("warn", foreground=COLOR_WARN)
output_box.tag_config("bad", foreground=COLOR_BAD)
output_box.tag_config("info", foreground=COLOR_INFO)

root.mainloop()
