import os
import pandas as pd

def merge_csv_metrics(root_path, method):
    # Determine the input sub-directory and target column based on the method name
    if method.lower().startswith("vwcd"):
        input_sub_folder = "tail_probability_theta_ge_Tstar"
        value_col = "P_theta_ge_Tstar"
    else:
        input_sub_folder = ""
        value_col = "CP"
    
    # Define the output directory
    output_base_dir = os.path.join(root_path, "full", method)
    os.makedirs(output_base_dir, exist_ok=True)

    # Identify metric folders (pl, rtt_down, etc.), excluding the "full" folder
    metrics = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d)) and d != "full"]
    
    if not metrics:
        return

    # Use the first metric folder as a reference to list CSV files
    reference_metric_path = os.path.join(root_path, metrics[0], method, input_sub_folder)
    if not os.path.exists(reference_metric_path):
        return
        
    csv_files = [f for f in os.listdir(reference_metric_path) if f.endswith('.csv')]

    for file_name in csv_files:
        merged_df = None

        for metric in metrics:
            file_path = os.path.join(root_path, metric, method, input_sub_folder, file_name)
            
            if os.path.exists(file_path):
                # Read as string to preserve exact decimal representation
                df = pd.read_csv(file_path, dtype=str)
                
                if 'timestamp' in df.columns and value_col in df.columns:
                    # Clean potential whitespace
                    df['timestamp'] = df['timestamp'].str.strip()
                    df[value_col] = df[value_col].str.strip()
                    
                    # New column name with 'P_' prefix
                    new_col_name = f"P_{metric}"
                    
                    if merged_df is None:
                        # Initialize the DataFrame with timestamp and rename the value column
                        merged_df = df[['timestamp', value_col]].copy()
                        merged_df.rename(columns={value_col: new_col_name}, inplace=True)
                    else:
                        # Join subsequent metrics based on the timestamp column
                        temp_df = df[['timestamp', value_col]].copy()
                        temp_df.rename(columns={value_col: new_col_name}, inplace=True)
                        merged_df = pd.merge(merged_df, temp_df, on='timestamp', how='outer')

        if merged_df is not None:
            output_file = os.path.join(output_base_dir, file_name)
            # Save the final merged CSV
            merged_df.to_csv(output_file, index=False)

def merge_probabilities_with_series(root_path, method, series_path):
    # Paths for the probability CSVs and the real series CSVs
    prob_base_dir = os.path.join(root_path, "full", method)
    series_base_dir = os.path.join(series_path, "full")
    
    if not os.path.exists(prob_base_dir):
        return

    # Map full column names to the abbreviated names used in the probability files
    # Probability columns: P_rtt_down, P_tp_down, P_rtt_up, P_tp_up, P_pl
    # Series columns: rtt_download, throughput_download, rtt_upload, throughput_upload, packet_loss
    column_mapping = {
        'rtt_download': 'rtt_down',
        'throughput_download': 'tp_down',
        'rtt_upload': 'rtt_up',
        'throughput_upload': 'tp_up',
        'packet_loss': 'pl'
    }

    # List all CSV files in the probability directory
    csv_files = [f for f in os.listdir(prob_base_dir) if f.endswith('.csv')]

    for file_name in csv_files:
        prob_file_path = os.path.join(prob_base_dir, file_name)
        series_file_path = os.path.join(series_base_dir, file_name)

        if os.path.exists(series_file_path):
            # Read both as string to maintain precision and identity
            df_prob = pd.read_csv(prob_file_path, dtype=str)
            df_series = pd.read_csv(series_file_path, dtype=str)

            # Clean whitespace and normalize timestamps for merging
            df_prob['timestamp'] = df_prob['timestamp'].str.strip()
            df_series['timestamp'] = df_series['timestamp'].str.strip()

            # Rename series columns to abbreviated versions
            df_series.rename(columns=column_mapping, inplace=True)

            # Merge the probability data with the real values
            # how='inner' ensures we only keep timestamps present in both (probabilities are usually a subset)
            merged_df = pd.merge(df_series, df_prob, on='timestamp', how='inner')

            # Overwrite the CSV in the method folder with the combined data
            # The result will contain: timestamp, abbreviated_series_cols, P_abbreviated_metrics
            merged_df.to_csv(prob_file_path, index=False)

if __name__ == "__main__":
    root_path = "results/NDT"  # Substitua pelo caminho real dos seus dados
    method = "vwcd"  # Substitua pelo nome do método que você deseja processar
    merge_csv_metrics(root_path, method)
    series_path = "series/NDT"  # Substitua pelo caminho real dos seus dados de séries temporais
    merge_probabilities_with_series(root_path, method, series_path)
