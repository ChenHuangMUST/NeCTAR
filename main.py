import os
import copy
import pickle
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

# Import required modules from herb_network_predictor
from nectar.modules import (
    data_preprocessing,
    herb_ratio_optimization,
    herb_filter,
    dosage_to_weight,
    calculateScore,
    weight_to_dosage
)
from nectar.modules.data_io import (
    load_herb_info,
    load_herb_nes,
    load_dosage_info,
    load_disease_data
)
from nectar.modules.data_preprocessing import prepare_input_data
from nectar.modules.seed_utils import set_random_seeds
from nectar.modules.utils import create_result_folders
from nectar.modules.plotting import plot_optimization_results

# Global variable for herbs to exclude
stopherbs = []


def compute_scores_for_top_list(input_data, dosage_array, dosage_range_array):
    """
    Generate a series of 'top' values in the range [0, 1] with decreasing step sizes.
    Each dosage (converted to weight) is normalized within the range [0, top] and the corresponding score is calculated.
    
    Parameters:
        input_data: Prepared input data for score calculation.
        dosage_array: Array of initial dosages.
        dosage_range_array: Array containing dosage ranges for each herb.
    
    Returns:
        A tuple (min_score, best_weights) where:
        - min_score: The minimum score obtained.
        - best_weights: The weight list corresponding to the minimum score.
    """
    n = 10
    top_list = [1 - i / n for i in range(n)] + [0]
    score_list = []
    weight_list = []

    for top in tqdm(top_list, desc="Computing scores for top values"):
        normalized_list = [
            weight_to_dosage.normalize_to_range([0, 1, usage], 0, top)[-1]
            for usage in dosage_to_weight.dosage_to_weight(dosage_array, dosage_range_array)
        ]
        weight_list.append(normalized_list)
        _, score = calculateScore.calculateScore(input_data, normalized_list)
        score_list.append(score)

    min_score = min(score_list)
    best_weights = weight_list[score_list.index(min_score)]
    return min_score, best_weights


def remove_zero_dosage(formula, weight_list, dosage_list, weights, normalized_weight_list):
    """
    Remove herbs with a zero dosage from all related lists (herb names, weights, dosages).
    
    Parameters:
        formula: List of herb names.
        weight_list: List of corresponding weights.
        dosage_list: List of dosages.
        weights: Torch tensor of weights.
        normalized_weight_list: List of normalized weights.
    
    Returns:
        A tuple with updated (formula, weight_list, dosage_list, weights as list, normalized_weight_list).
    """
    weights_cpu = weights.cpu().tolist()
    for i in range(len(dosage_list) - 1, -1, -1):
        if dosage_list[i] == 0:
            dosage_list.pop(i)
            weight_list.pop(i)
            formula.pop(i)
            weights_cpu.pop(i)
            normalized_weight_list.pop(i)
    return formula, weight_list, dosage_list, weights_cpu, normalized_weight_list


def add_top_herbs(result, df_herb_nes, top_num=50):
    """
    Sort herbs based on the logits provided in the result and return the names of the top herbs.
    
    Parameters:
        result: Output from herb_filter containing logits.
        df_herb_nes: DataFrame of herb NES information.
        top_num: Number of top herbs to select.
    
    Returns:
        List of herb names corresponding to the top logits.
    """
    logits_list = list(result[0])
    sorted_indices = sorted(range(len(logits_list)), key=lambda i: logits_list[i], reverse=True)
    column_names = df_herb_nes.columns.tolist()[1:]  # Exclude the first column (NES identifier)
    add_herb_list = [column_names[i] for i in sorted_indices[:top_num]]
    return add_herb_list


def optimize_herb_ratio(
    df_herb_nes, 
    formula, 
    df_disease, 
    herb_count, 
    result_folder, 
    dosage_range_array, 
    dosage_info
):
    """
    Core function for the herb ratio optimization process. This function:
      1. Prepares the input data.
      2. Optimizes herb weights.
      3. Converts weights to dosages and removes herbs with zero dosage.
      4. Resets dosage ranges based on the updated formula.
      5. Computes the final combined NES and score.
    
    Parameters:
        df_herb_nes: DataFrame containing herb NES information.
        formula: List of initial herb names.
        df_disease: DataFrame containing disease-related data.
        herb_count: Number of herbs in the formula.
        result_folder: Directory path to save results.
        dosage_range_array: Array of dosage ranges for each herb.
        dosage_info: DataFrame containing dosage information.
    
    Returns:
        A tuple containing:
            (formula, weight_list, dosage_list, dosage_range_array, combined_nes, score, herb_count, normalized_weight_list)
    """
    # Prepare input data for scoring
    input_data, _ = prepare_input_data(df_herb_nes, formula, df_disease)

    # Optimize herb weights
    weights, loss_values = herb_ratio_optimization.optimize_weights(input_data, herb_count, result_folder)

    # Convert weights to dosages
    weight_list, dosage_list, normalized_weight_list = weight_to_dosage.weightToDosage(weights, dosage_range_array)

    # Remove herbs with zero dosage
    formula, weight_list, dosage_list, weights_cpu_list, normalized_weight_list = remove_zero_dosage(
        formula, weight_list, dosage_list, weights, normalized_weight_list
    )

    # Reset dosage ranges based on the new formula
    min_limit_array = np.array(dosage_info[formula].iloc[0])
    max_limit_array = np.array(dosage_info[formula].iloc[1])
    herb_count = len(formula)
    dosage_range_array = np.array([[min_limit_array[i], max_limit_array[i]] for i in range(herb_count)])
    
    # Recalculate the final score with the updated data
    input_data, _ = prepare_input_data(df_herb_nes, formula, df_disease)
    combined_nes, score = calculateScore.calculateScore(input_data, weights_cpu_list)

    print("Final combined NES:", combined_nes)

    return (
        formula,
        weight_list,
        dosage_list,
        dosage_range_array,
        combined_nes,
        score,
        herb_count,
        normalized_weight_list
    )


def optimize_filter(formula, combined_nes, df_herb_nes, dosage_info):
    """
    Filter and expand the formula based on the current combined NES score.
    Adds the top potential herbs and resets the dosage limits.
    
    Parameters:
        formula: Current list of herb names.
        combined_nes: Current combined NES score.
        df_herb_nes: DataFrame of herb NES data.
        dosage_info: DataFrame containing dosage limits.
    
    Returns:
        A tuple (formula, herb_count, dosage_range_array) after filtering and updating.
    """
    result = herb_filter.herb_filter(combined_nes)
    formula += add_top_herbs(result, df_herb_nes, top_num=50)
    formula = list(set(formula))
    formula = sorted(formula)

    # Remove herbs that are forced to be excluded
    for herb in stopherbs:
        if herb in formula:
            formula.remove(herb)

    # Reset dosage limits based on the updated formula
    min_limit_array = np.array(dosage_info[formula].iloc[0])
    max_limit_array = np.array(dosage_info[formula].iloc[1])
    herb_count = len(formula)
    dosage_range_array = np.array([[min_limit_array[i], max_limit_array[i]] for i in range(herb_count)])
    return formula, herb_count, dosage_range_array


def optimize_herb_ratio_loop(
    df_herb_nes, 
    formula, 
    df_disease, 
    herb_count, 
    result_folder, 
    dosage_range_array, 
    dosage_info
):
    """
    Perform multiple rounds of herb ratio optimization.
    The loop continues until there is no improvement in the score for 5 consecutive iterations.
    
    Parameters:
        df_herb_nes: DataFrame containing herb NES information.
        formula: List of herb names.
        df_disease: DataFrame containing disease-related data.
        herb_count: Number of herbs.
        result_folder: Directory path to save results.
        dosage_range_array: Array of dosage ranges.
        dosage_info: DataFrame containing dosage information.
    
    Returns:
        A tuple containing the optimized values:
            (formula, weight_list, dosage_list, dosage_range_array, combined_nes, score, herb_count, normalized_weight_list)
    """
    best_score = float('inf')
    best_result = None
    no_improve_count = 0
    count = 0
    judge_set = set()

    while judge_set != set(formula) or count == 0:
        count += 1
        judge_set = set(formula)

        result = optimize_herb_ratio(
            df_herb_nes,
            formula,
            df_disease,
            herb_count,
            result_folder,
            dosage_range_array,
            dosage_info
        )

        (formula, weight_list, dosage_list, dosage_range_array,
         combined_nes, score, herb_count, normalized_weight_list) = result

        if score < best_score:
            best_score = score
            best_result = {
                'formula': copy.deepcopy(formula),
                'weight_list': copy.deepcopy(weight_list),
                'dosage_list': copy.deepcopy(dosage_list),
                'dosage_range_array': copy.deepcopy(dosage_range_array),
                'combined_nes': copy.deepcopy(combined_nes),
                'score': score,
                'herb_count': copy.deepcopy(herb_count),
                'normalized_weight_list': copy.deepcopy(normalized_weight_list)
            }
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= 5:
            print("Score did not decrease for 5 consecutive iterations. Terminating loop.")
            break

    if best_result:
        formula = best_result['formula']
        weight_list = best_result['weight_list']
        dosage_list = best_result['dosage_list']
        dosage_range_array = best_result['dosage_range_array']
        combined_nes = best_result['combined_nes']
        score = best_result['score']
        herb_count = best_result['herb_count']
        normalized_weight_list = best_result['normalized_weight_list']

    return (
        formula, 
        weight_list, 
        dosage_list, 
        dosage_range_array, 
        combined_nes, 
        score, 
        herb_count, 
        normalized_weight_list
    )


def nectar(herb_info_path, disease_data_path):
    """
    Execute the complete pipeline for optimizing Traditional Chinese Medicine formulas.
    
    Parameters:
        herb_info_path: Path to the Excel file containing initial herb information.
        disease_data_path: Path to the disease data file (pkl or other format).
    
    Returns:
        A dictionary containing:
            - final_formula: Optimized list of herb names.
            - dosage: Corresponding dosages.
            - final_score: Final score after optimization.
            - result_folder: Folder path where the results are saved.
    """
    set_random_seeds()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Load required data files
    herb_nes_path = os.path.join(BASE_DIR, "data", "df_herb_nes_mini.txt")
    dosage_info_path = os.path.join(BASE_DIR, "data", "dosage_info.txt")
    df_herb_nes = load_herb_nes(herb_nes_path)
    dosage_info = load_dosage_info(dosage_info_path)
    df_disease = load_disease_data(disease_data_path)

    # Load initial formula information (herb names and dosages)
    df_herb_info_excel = pd.read_excel(herb_info_path)
    formula = list(df_herb_info_excel["name"])
    dosage_array = np.array(df_herb_info_excel["dosage"])

    # Retrieve dosage limits
    min_limit_array = np.array(dosage_info[formula].iloc[0])
    max_limit_array = np.array(dosage_info[formula].iloc[1])
    herb_count = len(formula)
    dosage_range_array = np.array([[min_limit_array[i], max_limit_array[i]] for i in range(herb_count)])

    # Create folder for saving results
    result_folder = create_result_folders()

    # Calculate initial score
    input_data, no_nor_input_data = prepare_input_data(df_herb_nes, formula, df_disease)
    min_score, best_weights = compute_scores_for_top_list(input_data, dosage_array, dosage_range_array)

    # Record initial formula information (preserving full details)
    df_info_formula = pd.DataFrame(columns=["Cycle Count", "Formula", "weight", "Dosage", "Score"])
    df_info_formula.loc[0] = [0, formula[:], best_weights[:], dosage_array.tolist(), min_score]

    # First round of weight optimization
    (formula,
     weight_list,
     dosage_list,
     dosage_range_array,
     combined_nes,
     score,
     herb_count,
     normalized_weight_list) = optimize_herb_ratio_loop(
        df_herb_nes, 
        formula, 
        df_disease, 
        herb_count, 
        result_folder, 
        dosage_range_array, 
        dosage_info
    )

    # Visualize the preliminary result
    if formula:
        _, no_nor_input_data = prepare_input_data(df_herb_nes, formula, df_disease)
        calculateScore.calculateScore_plot(formula, no_nor_input_data, weight_list, result_folder, plot=0)

    # Save the first round optimization results
    df_info_formula.loc[1] = [1, formula[:], normalized_weight_list[:], dosage_list[:], score]

    # Loop for further optimization ("add herbs + ratio optimization")
    all_results = []
    best_score = float('inf')
    no_improve_count = 0
    count = 0
    judge_set = set()

    while judge_set != set(formula) or count == 0:
        count += 1
        judge_set = set(formula)

        # Filter and expand the formula based on current results
        formula, herb_count, dosage_range_array = optimize_filter(
            formula, combined_nes, df_herb_nes, dosage_info
        )

        # Perform multiple rounds of weight optimization
        (formula,
         weight_list,
         dosage_list,
         dosage_range_array,
         combined_nes,
         score,
         herb_count,
         normalized_weight_list) = optimize_herb_ratio_loop(
            df_herb_nes, 
            formula, 
            df_disease, 
            herb_count, 
            result_folder, 
            dosage_range_array, 
            dosage_info
        )

        current_result = {
            'formula': copy.deepcopy(formula),
            'weight_list': copy.deepcopy(weight_list),
            'dosage_list': copy.deepcopy(dosage_list),
            'dosage_range_array': copy.deepcopy(dosage_range_array),
            'combined_nes': copy.deepcopy(combined_nes),
            'score': score,
            'herb_count': copy.deepcopy(herb_count),
            'normalized_weight_list': copy.deepcopy(normalized_weight_list)
        }
        all_results.append(current_result)

        if score < best_score:
            best_score = score
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= 3:
            print("Score did not decrease for 3 consecutive iterations. Terminating loop.")
            break

    # Select the best result among all iterations
    if all_results:
        best_result = min(all_results, key=lambda x: x['score'])
    else:
        best_result = None

    if best_result:
        formula = best_result['formula']
        weight_list = best_result['weight_list']
        dosage_list = best_result['dosage_list']
        dosage_range_array = best_result['dosage_range_array']
        combined_nes = best_result['combined_nes']
        score = best_result['score']
        herb_count = best_result['herb_count']
        normalized_weight_list = best_result['normalized_weight_list']

    # Visualize the final optimization result
    _, no_nor_input_data = prepare_input_data(df_herb_nes, formula, df_disease)
    calculateScore.calculateScore_plot(formula, no_nor_input_data, weight_list, result_folder, plot=0)

    # Save the final result information
    df_info_formula.loc[2] = [2, formula[:], normalized_weight_list[:], dosage_list[:], score]
    output_path_full = os.path.join(result_folder, "df_info_formula.xlsx")
    df_info_formula.to_excel(output_path_full, index=False)

    # Save a simplified Excel file with only herb names and dosages
    df_final = pd.DataFrame({
        "herb": formula,
        "dosage": dosage_list
    })
    output_path_simple = os.path.join(result_folder, "final_herb_dosage.xlsx")
    df_final.to_excel(output_path_simple, index=False)

    # Optional: Visualize all historical optimization results
    if all_results:
        plot_optimization_results(all_results, result_folder)

    return {
        "final_formula": formula,
        "dosage": dosage_list,
        "final_score": score,
        "result_folder": result_folder
    }


if __name__ == "__main__":
    import argparse
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    set_random_seeds()

    parser = argparse.ArgumentParser(description="Herb predictor pipeline.")
    parser.add_argument(
        "--herb_info_path",
        type=str,
        default=os.path.join(BASE_DIR, "data", "info_input_herbs.xlsx"),
        help="Path to the Excel file containing initial herb information."
    )
    parser.add_argument(
        "--disease_data_path",
        type=str,
        default=os.path.join(BASE_DIR, "data", "disease_nes.pkl"),
        help="Path to the disease data file (pkl or other format)."
    )
    
    args = parser.parse_args()

    # Execute the main pipeline
    result = nectar(args.herb_info_path, args.disease_data_path)

    # Further process or display the final results
    herbList = result["final_formula"]
    dosage_list = result["dosage"]
    resultList = [f"{herbList[i]} {dosage_list[i]}" for i in range(len(herbList))]
    print("Final formula:", resultList)
    print("Final score:", result["final_score"])
