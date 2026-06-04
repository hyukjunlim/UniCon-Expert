import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_annotations(
    annotations_path='annotation_llm.csv', 
    input_data_path='annotation_input_data.csv',
    output_dir='figs/annotation_analysis'
):
    """
    Analyzes human preferences vs. model scores.
    """
    # 1. Load Data
    if not os.path.exists(annotations_path):
        print(f"Error: {annotations_path} not found. Run the annotation app first.")
        return

    df_human = pd.read_csv(annotations_path)
    df_model = pd.read_csv(input_data_path)
    
    # 2. Preprocessing & Merging
    # Ensure reaction_smiles is the key. Drop duplicates in model data just in case.
    df_model = df_model.drop_duplicates(subset=['reaction_smiles'])
    
    # Merge: We only analyze reactions that have been annotated
    df = pd.merge(df_human, df_model, on='reaction_smiles', how='inner')
    
    if len(df) == 0:
        print("No matching records found between annotations and input data.")
        return

    print(f"Loaded {len(df)} annotated samples.")

    # 3. Decode Human Preference
    # Logic: 
    # - If User picked Option 1 AND Option 1 was GT -> Human picked GT
    # - If User picked Option 2 AND Option 1 was GT -> Human picked Baseline (since Opt 2 is Baseline)
    # - If User picked Option 1 AND Option 1 was Baseline -> Human picked Baseline
    # - ...
    
    def get_human_choice(row):
        choice = row['user_choice'] # 'option1', 'option2', 'tie', 'bad'
        is_opt1_gt = row['is_option_1_GT']
        
        if choice == 'Tie':
            return 'Tie'
        if choice == 'Bad':
            return 'Bad'
        
        if choice == 'Option 1':
            return 'GT' if is_opt1_gt else 'Baseline'
        elif choice == 'Option 2':
            return 'Baseline' if is_opt1_gt else 'GT'
        return 'Unknown'

    df['human_preference'] = df.apply(get_human_choice, axis=1)

    # 4. Decode Model Preference
    # Assuming 'winner' column contains the text of the winning condition.
    # Check if winner matches condition_a (GT) or condition_b (Baseline)
    def get_model_choice(row):
        # Handle potential float/nan issues or whitespace
        winner = str(row['winner']).strip()
        if winner == 'condition_a':
            return 'GT'
        elif winner == 'condition_b':
            return 'Baseline'
        elif winner == 'tie':
            return 'Tie'
        else:
            raise ValueError(f"Unknown winner: {winner}")
            
    df['model_preference'] = df.apply(get_model_choice, axis=1)

    # 5. Core Metrics Calculation
    
    # A. General Distribution
    print("\n=== 1. Human Preference Distribution ===")
    human_counts = df['human_preference'].value_counts()
    print(human_counts)
    
    # B. Model Preference Distribution (on this subset)
    print("\n=== 2. Model Preference Distribution ===")
    model_counts = df['model_preference'].value_counts()
    print(model_counts)

    # C. Alignment (Agreement) Analysis
    # We filter out 'Tie' and 'Bad' for strict accuracy, but keep them for overall analysis
    valid_decisions = df[~df['human_preference'].isin(['Tie', 'Bad'])]
    
    if len(valid_decisions) > 0:
        agreement = (valid_decisions['human_preference'] == valid_decisions['model_preference']).sum()
        accuracy = agreement / len(valid_decisions)
        print(f"\n=== 3. Strict Alignment Accuracy ===")
        print(f"Agreement on {len(valid_decisions)} decisive samples: {agreement} ({accuracy:.1%})")
        print("(Excludes Ties and Bad/Invalid reactions)")
    else:
        print("\n=== 3. Strict Alignment Accuracy ===")
        print("Not enough decisive samples.")

    # D. The "Discovery" Rate (Crucial for your paper)
    # Case: Model preferred Baseline. Did Human agree?
    discovery_subset = df[df['model_preference'] == 'Baseline']
    n_discovery_attempts = len(discovery_subset)
    
    if n_discovery_attempts > 0:
        # Success = Human chose Baseline OR Tie (meaning Baseline is valid)
        confirmed_discovery = discovery_subset[discovery_subset['human_preference'].isin(['Baseline', 'Tie'])]
        success_rate = len(confirmed_discovery) / n_discovery_attempts
        
        print(f"\n=== 4. 'Valid Discovery' Rate ===")
        print(f"Model proposed Baseline > GT in {n_discovery_attempts} cases.")
        print(f"Human confirmed (Baseline or Tie) in {len(confirmed_discovery)} cases.")
        print(f"Success Rate: {success_rate:.1%}")
    else:
        print("\n=== 4. 'Valid Discovery' Rate ===")
        print("Model did not prefer Baseline in any annotated samples.")

    # 6. Visualization
    os.makedirs(output_dir, exist_ok=True)
    
    # Confusion Matrix-style Heatmap
    labels = ['GT', 'Baseline', 'Tie', 'Bad']
    ct = pd.crosstab(
        df['model_preference'],
        df['human_preference']
    )
    ct = ct.reindex(index=labels, columns=labels, fill_value=0)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(ct, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Model vs Human Preference Alignment')
    plt.ylabel('Model Choice')
    plt.xlabel('Human Choice')
    plt.savefig(f'{output_dir}/preference_heatmap.png')
    print(f"\nSaved heatmap to {output_dir}/preference_heatmap.png")
    
    # 7. Save merged analysis file
    output_csv = f'{output_dir}/full_analysis_report.csv'
    df.to_csv(output_csv, index=False)
    print(f"Detailed analysis saved to {output_csv}")

    # 8. Spearman Rank Correlation
    from scipy.stats import spearmanr
    df['model_margin'] = df['gt_score'] - df['baseline_score']
    human_map = {'GT': 1, 'Tie': 0.0, 'Baseline': -1.0}
    corr_df = df[df['human_preference'].isin(['GT', 'Baseline', 'Tie'])].copy()
    corr_df['human_rank'] = corr_df['human_preference'].map(human_map)

    corr_df = corr_df.dropna(subset=['human_rank', 'model_margin'])
    corr_df = corr_df[corr_df['human_rank'] != 0]

    if len(corr_df) >= 3:
        rho, pval = spearmanr(corr_df['human_rank'], corr_df['model_margin'])
        print("\n=== 5. Spearman Rank Correlation ===")
        print(f"N={len(corr_df)}  rho={rho:.3f}  p-value={pval:.3g}")
    else:
        print("\n=== 5. Spearman Rank Correlation ===")
        print("Not enough samples after filtering for correlation.")
        
    corr_df['abs_margin'] = corr_df['model_margin'].abs()
    corr_df['abs_margin'] = corr_df['abs_margin'].replace(np.inf, 100)
    bins = pd.qcut(corr_df['abs_margin'], q=3)
    grouped = corr_df.groupby(bins, observed=False)

    for b, g in grouped:
        agree = (np.sign(g['model_margin']) == g['human_rank']).mean()
        print(f"{b}: agreement={agree:.2f}, N={len(g)}")

if __name__ == "__main__":
    analyze_annotations()