import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = "data/output/eval_workspace.xlsx" 
OUTPUT_FILE = "data/output/eval.xls"

def calculate_fleiss_kappa(ratings_matrix):
    """
    Calculates Fleiss' Kappa for a list of ratings.
    ratings_matrix: List of lists, e.g., [['Pos', 'Pos', 'Neu'], ['Neg', 'Neg', 'Neg'], ...]
    """
    n_items = len(ratings_matrix)
    n_raters = len(ratings_matrix[0])
    
    # Find all unique categories used
    categories = list(set(val for row in ratings_matrix for val in row if pd.notna(val)))
    k_categories = len(categories)
    
    # Create the Subject-Category matrix (items x categories)
    mat = np.zeros((n_items, k_categories))
    for i, row in enumerate(ratings_matrix):
        for val in row:
            if pd.notna(val):
                mat[i, categories.index(val)] += 1
                
    # Calculate P_i (extent to which raters agree on the i-th subject)
    P_i = (np.sum(mat**2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = np.mean(P_i)
    
    # Calculate P_j (proportion of all assignments to the j-th category)
    p_j = np.sum(mat, axis=0) / (n_items * n_raters)
    P_e_bar = np.sum(p_j**2)
    
    # Calculate Kappa
    if P_e_bar == 1:
        return 1.0
    kappa = (P_bar - P_e_bar) / (1 - P_e_bar)
    return kappa

def format_and_calculate_iaa():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_excel(INPUT_FILE, engine="openpyxl")
    df.columns = df.columns.str.strip()

    # --- 1. CALCULATE IAA (Pairwise & Fleiss' Kappa) ---
    print("\n" + "="*50)
    print("📊 INTER-ANNOTATOR AGREEMENT (FOR REPORT)")
    print("="*50)
    
    subj_cols =["Annotator_1_Subj", "Annotator_2_Subj", "Annotator_3_Subj"]
    pol_cols =["Annotator_1_Pol", "Annotator_2_Pol", "Annotator_3_Pol"]
    
    if all(c in df.columns for c in subj_cols + pol_cols):
        # A. Average Pairwise Percentage (To satisfy the literal 80% rubric)
        # Compare 1&2, 1&3, and 2&3
        s_12 = (df[subj_cols[0]] == df[subj_cols[1]]).mean()
        s_13 = (df[subj_cols[0]] == df[subj_cols[2]]).mean()
        s_23 = (df[subj_cols[1]] == df[subj_cols[2]]).mean()
        avg_subj_pairwise = (s_12 + s_13 + s_23) / 3 * 100

        p_12 = (df[pol_cols[0]] == df[pol_cols[1]]).mean()
        p_13 = (df[pol_cols[0]] == df[pol_cols[2]]).mean()
        p_23 = (df[pol_cols[1]] == df[pol_cols[2]]).mean()
        avg_pol_pairwise = (p_12 + p_13 + p_23) / 3 * 100

        # B. Fleiss' Kappa (The Academic Innovation)
        subj_matrix = df[subj_cols].values.tolist()
        pol_matrix = df[pol_cols].values.tolist()
        
        kappa_subj = calculate_fleiss_kappa(subj_matrix)
        kappa_pol = calculate_fleiss_kappa(pol_matrix)

        print(f"Subjectivity -> Avg Pairwise Agreement: {avg_subj_pairwise:.1f}% | Fleiss' Kappa: {kappa_subj:.3f}")
        print(f"Polarity     -> Avg Pairwise Agreement: {avg_pol_pairwise:.1f}% | Fleiss' Kappa: {kappa_pol:.3f}")
        
    else:
        print("❌ Missing Annotator columns. Ensure Annotator 1, 2, and 3 columns exist.")
        return

    # --- 2. FORMAT AND EXCEL EXPORT ---
    df = df.rename(columns={
        "Annotator_3_Subj": "subjectivity",
        "Annotator_3_Pol": "polarity"
    })

    final_cols =["id", "text", "subjectivity", "polarity"]
    df_final = df[final_cols].copy()

    # Drop blank rows from the final ground truth
    df_final = df_final.replace("", pd.NA).dropna(subset=["subjectivity", "polarity"])

    df_final["subjectivity"] = df_final["subjectivity"].astype(str).str.lower()
    df_final["polarity"] = df_final["polarity"].astype(str).str.lower()

    df_final.to_excel(OUTPUT_FILE, index=False, engine="openpyxl")
    print(f"\n✅ Cleaned ground-truth saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    format_and_calculate_iaa()