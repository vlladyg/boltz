import os
import re
import json
import warnings
import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser
#import matplotlib.pyplot as plt
#from matplotlib.patches import Rectangle
from collections import Counter
from scipy.spatial.distance import pdist, squareform

warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore')

def transform_pae_matrix(pae_matrix, pae_cutoff):
    # Initialize the transformed matrix with zeros
    transformed_pae = np.zeros_like(pae_matrix)

    # Apply transformation: pae = 0 -> score = 1, pae = cutoff -> score = 0, above cutoff -> score = 0
    # Linearly scale values between 0 and cutoff to fall between 1 and 0
    within_cutoff = pae_matrix < pae_cutoff
    transformed_pae[within_cutoff] = 1 - (pae_matrix[within_cutoff] / pae_cutoff)
    
    return transformed_pae

def calculate_mean_lis(transformed_pae, subunit_number):
    # Calculate the cumulative sum of protein lengths to get the end indices of the submatrices
    cum_lengths = np.cumsum(subunit_number)
    
    # Add a zero at the beginning of the cumulative lengths to get the start indices
    start_indices = np.concatenate(([0], cum_lengths[:-1]))
    
    # Initialize an empty matrix to store the mean LIS
    mean_lis_matrix = np.zeros((len(subunit_number), len(subunit_number)))
    
    # Iterate over the start and end indices
    for i in range(len(subunit_number)):
        for j in range(len(subunit_number)):
            # Get the start and end indices of the interaction submatrix
            start_i, end_i = start_indices[i], cum_lengths[i]
            start_j, end_j = start_indices[j], cum_lengths[j]
            
            # Get the interaction submatrix
            submatrix = transformed_pae[start_i:end_i, start_j:end_j]
            
            # Calculate the mean LIS, considering only non-zero values
            mean_lis = submatrix[submatrix > 0].mean()
            
            # Store the mean LIS in the matrix
            mean_lis_matrix[i, j] = mean_lis
    
    return mean_lis_matrix

def calculate_contact_map_my(coordinates, distance_threshold=8):
    
    distances = squareform(pdist(coordinates))

    # Assuming the column for atom names is at index 3 after insertion
    #has_phosphorus = df.iloc[:, 3].apply(lambda x: 'P' in str(x)).to_numpy()

    # Adjust the threshold for phosphorus-containing residues
    #adjusted_distances = np.where(has_phosphorus[:, np.newaxis] | has_phosphorus[np.newaxis, :], 
     #                             distances - 4, distances)

    contact_map = np.where(distances < distance_threshold, 1, 0)
    return contact_map

def afm3_plot_average_to_df_my(pae_matrix, token_chain_ids, chain_pair_iptm, coordinates, pae_cutoff=12, distance_cutoff=8, result_save="True"):
    sum_pae_matrix = None
    sum_transformed_pae_matrix = None
    sum_mean_lis_matrix = None
    sum_contact_lia_map = None
    sum_iptm_matrix = None
    all_interactions = []


    # ----------------------------------------------
    # 1) Read JSON data
    # ----------------------------------------------

    chain_residue_counts = Counter(token_chain_ids)
    subunit_number = list(chain_residue_counts.values())

    # ----------------------------------------------
    # 2) Transform PAE matrix => LIS
    # ----------------------------------------------
    transformed_pae_matrix = transform_pae_matrix(pae_matrix, pae_cutoff)
    transformed_pae_matrix = np.nan_to_num(transformed_pae_matrix)

    # A binary map (1 where LIS>0, else 0)
    lia_map = np.where(transformed_pae_matrix > 0, 1, 0)

    mean_lis_matrix = calculate_mean_lis(transformed_pae_matrix, subunit_number)
    mean_lis_matrix = np.nan_to_num(mean_lis_matrix)

    # ----------------------------------------------
    # 3) Contact map => cLIA
    # ----------------------------------------------
    contact_map = calculate_contact_map_my(coordinates, distance_cutoff)

    combined_map = np.where(
        (transformed_pae_matrix > 0) & (contact_map == 1),
        transformed_pae_matrix, 
        0
    )
    mean_clis_matrix = calculate_mean_lis(combined_map, subunit_number)
    mean_clis_matrix = np.nan_to_num(mean_clis_matrix)

    # ----------------------------------------------
    # 4) Count-based metrics: LIA, LIR, cLIA, cLIR
    #    plus local (per-subunit) residue indices
    # ----------------------------------------------
    subunit_count = len(subunit_number)
    lia_matrix = np.zeros((subunit_count, subunit_count), dtype=int)
    lir_matrix = np.zeros((subunit_count, subunit_count), dtype=int)
    clia_matrix = np.zeros((subunit_count, subunit_count), dtype=int)
    clir_matrix = np.zeros((subunit_count, subunit_count), dtype=int)

    # For extracting submatrices
    cum_lengths = np.cumsum(subunit_number)
    starts = np.concatenate(([0], cum_lengths[:-1]))

    for i in range(subunit_count):
        for j in range(subunit_count):
            # subunit i spans [start_i, end_i), subunit j spans [start_j, end_j)
            start_i, end_i = starts[i], cum_lengths[i]
            start_j, end_j = starts[j], cum_lengths[j]

            # Submatrix for LIS-based local interactions (binary)
            interaction_submatrix = lia_map[start_i:end_i, start_j:end_j]
            lia_matrix[i, j] = np.count_nonzero(interaction_submatrix)

            # *Local* residue indices for subunit i and j => add +1 to be 1-based
            # but do NOT add +start_i or +start_j, so each subunit is 1..(subunit_number[i])
            residues_i_LIR = np.unique(np.where(interaction_submatrix > 0)[0]) + 1
            residues_j_LIR = np.unique(np.where(interaction_submatrix > 0)[1]) + 1
            lir_matrix[i, j] = len(residues_i_LIR) + len(residues_j_LIR)

            # Submatrix for contact-based local interactions
            combined_submatrix = combined_map[start_i:end_i, start_j:end_j]
            clia_matrix[i, j] = np.count_nonzero(combined_submatrix)

            residues_i_cLIR = np.unique(np.where(combined_submatrix > 0)[0]) + 1
            residues_j_cLIR = np.unique(np.where(combined_submatrix > 0)[1]) + 1
            clir_matrix[i, j] = len(residues_i_cLIR) + len(residues_j_cLIR)

    # ----------------------------------------------
    # 5) ipTM
    # ----------------------------------------------
    iptm_matrix = np.array(chain_pair_iptm, dtype=float)
    iptm_matrix = np.nan_to_num(iptm_matrix)

    # ----------------------------------------------
    # 6) Accumulate sums for averaging across models
    # ----------------------------------------------
    if sum_pae_matrix is None:
        sum_pae_matrix = pae_matrix
        sum_transformed_pae_matrix = transformed_pae_matrix
        sum_mean_lis_matrix = mean_lis_matrix
        sum_contact_lia_map = combined_map
        sum_iptm_matrix = iptm_matrix
    else:
        sum_pae_matrix += pae_matrix
        sum_transformed_pae_matrix += transformed_pae_matrix
        sum_mean_lis_matrix += mean_lis_matrix
        sum_contact_lia_map += combined_map
        sum_iptm_matrix += iptm_matrix

    # ----------------------------------------------
    # 7) Build model-level rows
    #    with the local (per-subunit) indices
    # ----------------------------------------------
    model_number = 0
    #folder_name = os.path.basename(os.path.dirname(af3_json))
    ilis_matrix = np.zeros((subunit_count, subunit_count))
    
    for i in range(subunit_count):
        for j in range(subunit_count):
            iLIS_value = np.sqrt(mean_lis_matrix[i, j] * mean_clis_matrix[i, j])
            ilis_matrix[i, j] = iLIS_value
            # We re-derive the local submatrices to get the final index arrays
            start_i, end_i = starts[i], cum_lengths[i]
            start_j, end_j = starts[j], cum_lengths[j]

            interaction_submatrix = lia_map[start_i:end_i, start_j:end_j]
            residues_i_LIR = np.unique(np.where(interaction_submatrix > 0)[0]) + 1
            residues_j_LIR = np.unique(np.where(interaction_submatrix > 0)[1]) + 1

            combined_submatrix = combined_map[start_i:end_i, start_j:end_j]
            residues_i_cLIR = np.unique(np.where(combined_submatrix > 0)[0]) + 1
            residues_j_cLIR = np.unique(np.where(combined_submatrix > 0)[1]) + 1

            row_dict = {
                'folder_name': './',
                'model_number': model_number,
                'protein_1': i + 1,
                'protein_2': j + 1,
                'iLIS': iLIS_value,
                'LIS': mean_lis_matrix[i, j],
                'LIA': lia_matrix[i, j],
                'LIR': lir_matrix[i, j],
                'cLIS': mean_clis_matrix[i, j],
                'cLIA': clia_matrix[i, j],
                'cLIR': clir_matrix[i, j],
                'iptm': iptm_matrix[i, j],
                'LIR_indices_A': residues_i_LIR.tolist(),
                'LIR_indices_B': residues_j_LIR.tolist(),
                'cLIR_indices_A': residues_i_cLIR.tolist(),
                'cLIR_indices_B': residues_j_cLIR.tolist()
            }
            all_interactions.append(row_dict)

    # -----------------------------------------------------------
    # 8) Create average matrices (optional but consistent)
    # -----------------------------------------------------------
    n = 1
    avg_pae_matrix = np.nan_to_num(sum_pae_matrix / n)
    avg_transformed_pae_matrix = np.nan_to_num(sum_transformed_pae_matrix / n)
    avg_mean_lis_matrix = np.nan_to_num(sum_mean_lis_matrix / n)
    avg_contact_lia_map = np.nan_to_num(sum_contact_lia_map / n)
    avg_iptm_matrix = np.nan_to_num(sum_iptm_matrix / n)

    # -----------------------------------------------------------
    # 9) Build final DataFrame, group, reorder columns
    # -----------------------------------------------------------
    df_interactions = pd.DataFrame(all_interactions)

    # Sort the chain pair into a tuple so (1,2) and (2,1) are recognized as the same
    df_interactions['interaction'] = df_interactions.apply(
        lambda row: tuple(sorted((row['protein_1'], row['protein_2']))),
        axis=1
    )

    # Use an explicit aggregator to keep columns properly
    df_merged = df_interactions.groupby(
        ['folder_name', 'model_number', 'interaction'], as_index=False
    ).agg({
        'protein_1': 'first',
        'protein_2': 'first',
        'iLIS': 'mean',
        'LIS': 'mean',
        'LIA': 'mean',
        'LIR': 'mean',
        'cLIS': 'mean',
        'cLIA': 'mean',
        'cLIR': 'mean',
        'iptm': 'mean',
        'LIR_indices_A': 'first',
        'LIR_indices_B': 'first',
        'cLIR_indices_A': 'first',
        'cLIR_indices_B': 'first'
    })

    # Now add "average" rows, per (protein_1, protein_2) across all models
    avg_rows = []
    for (p1, p2), sub_df in df_merged.groupby(['protein_1', 'protein_2']):
        first_row = sub_df.iloc[0]
        avg_rows.append({
            'folder_name': './',
            'model_number': 'average',
            'interaction': (p1, p2),
            'protein_1': p1,
            'protein_2': p2,
            'iLIS': sub_df['iLIS'].mean(),
            'LIS': sub_df['LIS'].mean(),
            'LIA': sub_df['LIA'].mean(),
            'LIR': sub_df['LIR'].mean(),
            'cLIS': sub_df['cLIS'].mean(),
            'cLIA': sub_df['cLIA'].mean(),
            'cLIR': sub_df['cLIR'].mean(),
            'iptm': sub_df['iptm'].mean(),
            'LIR_indices_A': None,
            'LIR_indices_B': None,
            'cLIR_indices_A': None,
            'cLIR_indices_B': None
        })

    df_avg = pd.DataFrame(avg_rows)
    df_merged = pd.concat([df_merged, df_avg], ignore_index=True)

    # Convert selected columns to integer
    for col in ['LIA', 'LIR', 'cLIA', 'cLIR']:
        df_merged[col] = df_merged[col].astype(int)

    # We can now drop 'interaction' if we like
    if 'interaction' in df_merged.columns:
        df_merged.drop(columns=['interaction'], inplace=True)

    # Reorder columns so folder_name, model_number are first
    desired_order = [
        'folder_name', 'model_number',
        'protein_1', 'protein_2',
        'iLIS', 'LIS', 'LIA', 'LIR',
        'cLIS', 'cLIA', 'cLIR',
        'iptm',
        'LIR_indices_A', 'LIR_indices_B',
        'cLIR_indices_A', 'cLIR_indices_B'
    ]
    df_merged = df_merged[desired_order]

    # -----------------------------------------------------------
    # 10) Save DataFrame to CSV
    # -----------------------------------------------------------
    #output_folder = os.path.dirname(af3_jsons[0])
    #folder_name = os.path.basename(output_folder)
    #output_path = os.path.join(output_folder, f"{folder_name}_lis_analysis.csv")

    #if result_save == "True":
    #    df_merged.to_csv(output_path, index=False)
    #    print("Results saved to:", output_path)
    #    print(f"{folder_name}_lis_analysis.csv")

    return df_merged, ilis_matrix, lia_matrix
