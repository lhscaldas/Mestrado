import os
import pandas as pd
from pathlib import Path

def segment_series_by_changepoints(scenario: str, method: str, threshold: float = None):  # type: ignore
    scenario_path = Path(os.path.join('results', scenario))
    series_path = Path(os.path.join('series', scenario))
    
    if not scenario_path.exists():
        raise FileNotFoundError(f"The scenario folder '{scenario}' was not found.")

    for variable_dir in scenario_path.iterdir():
        if variable_dir.is_dir():
            variable = variable_dir.name
            
            if method == 'vwcd':
                target_dir = variable_dir / method / "tail_probability_theta_ge_Tstar"
            else:
                target_dir = variable_dir / method
            
            if target_dir.exists():
                records = []
                for csv_file in target_dir.glob("*.csv"):
                    name_parts = csv_file.stem.split('_', 1)
                    if len(name_parts) == 2:
                        client, server = name_parts
                        
                        df = pd.read_csv(csv_file)
                        
                        cp_indices = []
                        if method == 'vwcd':
                            if 'P_theta_ge_Tstar' in df.columns and threshold is not None:
                                cp_indices = df.index[df['P_theta_ge_Tstar'] > threshold].tolist()
                        else:
                            if 'CP' in df.columns:
                                cp_indices = df.index[df['CP'] == 1].tolist()
                        
                        series_csv_path = series_path / variable / csv_file.name
                        if series_csv_path.exists():
                            df_series = pd.read_csv(series_csv_path)
                            
                            col_name = variable if variable in df_series.columns else df_series.columns[-1]
                            series_data = df_series[col_name]
                            
                            cp_indices = sorted([cp for cp in cp_indices if cp < len(series_data)])
                            boundaries = [0] + cp_indices + [len(series_data)]
                            
                            segment_order = 1
                            for i in range(len(boundaries) - 1):
                                start = boundaries[i]
                                end = boundaries[i+1]
                                
                                if start < end:
                                    segment = series_data.iloc[start:end]
                                    
                                    records.append({
                                        'client': client,
                                        'server': server,
                                        'segment_order': segment_order,
                                        'mean': segment.mean(),
                                        'min': segment.min(),
                                        'max': segment.max(),
                                        'std': segment.std() if len(segment) > 1 else 0.0
                                    })
                                    segment_order += 1

                if records:
                    final_df = pd.DataFrame(records)
                    
                    out_dir = Path(f"segments/{scenario}/{variable}")
                    out_dir.mkdir(parents=True, exist_ok=True)
                    
                    if method == 'vwcd':
                        out_path = out_dir / f"{method}_{threshold}.csv"
                    else:
                        out_path = out_dir / f"{method}.csv"
                        
                    final_df.to_csv(out_path, index=False)

def segment_series_by_threshold(scenario, method, threshold, ref_metric):
    input_dir = os.path.join("results", scenario, "full", method)
    output_dir = os.path.join("segments", scenario, "full")
    os.makedirs(output_dir, exist_ok=True)
    
    ref_col = f"P_{ref_metric}"
    # List of all probability columns to check for the threshold count
    prob_cols = ['P_rtt_down', 'P_tp_down', 'P_rtt_up', 'P_tp_up', 'P_pl']
    # Value columns for statistics
    value_metrics = ['rtt_down', 'tp_down', 'rtt_up', 'tp_up', 'pl']
    
    all_segments_data = []

    if not os.path.exists(input_dir):
        return

    for file_name in [f for f in os.listdir(input_dir) if f.endswith('.csv')]:
        name_part = file_name.replace('.csv', '')
        client, server = name_part.split('_', 1) if '_' in name_part else (name_part, "unknown")

        df = pd.read_csv(os.path.join(input_dir, file_name))
        if ref_col not in df.columns:
            continue

        # Convert probability columns to float to ensure comparison works
        for col in prob_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Segmentation logic based ONLY on ref_metric
        df['is_change'] = df[ref_col] >= threshold
        
        # Shift the cumsum so that the row where 'is_change' is True 
        # still belongs to the PREVIOUS segment (representing the end of it)
        df['segment_id'] = df['is_change'].shift(fill_value=False).cumsum()

        groups = list(df.groupby('segment_id'))
        for i, (seg_id, group) in enumerate(groups):
            if group.empty:
                continue
                
            # The 'timestamp' now reflects the END of the segment (the moment of change)
            # For the last segment of the file, it will be the last available timestamp
            current_timestamp = group['timestamp'].iloc[-1]
            
            # Feature: Count how many metrics exceeded the threshold AT THIS TIMESTAMP
            # We look at the last row of the group, which is the changepoint row
            metrics_above_threshold = 0
            last_row = group.iloc[-1]
            for col in prob_cols:
                if col in last_row and last_row[col] >= threshold:
                    metrics_above_threshold += 1

            segment_stats = {
                'client': client,
                'server': server,
                'segment_number': seg_id,
                'timestamp': current_timestamp,
                'start_timestamp': group['timestamp'].iloc[0],
                'metrics_at_threshold': metrics_above_threshold,
                'n_points': len(group)
            }

            for metric in value_metrics:
                if metric in group.columns:
                    values = pd.to_numeric(group[metric], errors='coerce')
                    segment_stats[f'{metric}_mean'] = values.mean()
                    segment_stats[f'{metric}_median'] = values.median()
                    segment_stats[f'{metric}_std'] = values.std()

            all_segments_data.append(segment_stats)

    if all_segments_data:
        output_df = pd.DataFrame(all_segments_data)
        output_path = os.path.join(output_dir, f"{method}_{ref_metric}_{threshold}.csv")
        output_df.to_csv(output_path, index=False)

def feature_extraction(scenario, method, threshold, ref_metric):
    folder_path = os.path.join("segments", scenario, "full")
    file_name = f"{method}_{ref_metric}_{threshold}.csv"
    input_path = os.path.join(folder_path, file_name)
    output_path = os.path.join(folder_path, f"features_{file_name}")
    
    if not os.path.exists(input_path):
        return

    df = pd.read_csv(input_path)
    
    # 1. Filtro de pares sem changepoint
    df['pair'] = df['client'] + "_" + df['server']
    pairs_with_changes = df.groupby('pair')['segment_number'].max()
    pairs_with_changes = pairs_with_changes[pairs_with_changes > 0].index
    df_filtered = df[df['pair'].isin(pairs_with_changes)].copy()
    
    df_filtered = df_filtered.sort_values(['client', 'server', 'segment_number'])
    
    eps = 1e-9
    features_list = []

    for (client, server), group in df_filtered.groupby(['client', 'server']):
        group = group.reset_index(drop=True)
        
        # Iteramos até len - 1 pois o último segmento não tem um "próximo" para comparar
        for i in range(len(group) - 1):
            curr_seg = group.iloc[i]     # Ex: Segmento 0
            next_seg = group.iloc[i+1]   # Ex: Segmento 1
            
            feat = {
                'client': curr_seg['client'],
                'server': curr_seg['server'],
                'segment_number': curr_seg['segment_number'],
                'timestamp': curr_seg['timestamp'] # Timestamp do changepoint
            }
            
            # RTT Deltas: Próximo (1) - Atual (0)
            feat['delta_rtt_up'] = next_seg['rtt_up_median'] - curr_seg['rtt_up_median']
            feat['delta_rtt_down'] = next_seg['rtt_down_median'] - curr_seg['rtt_down_median']
            
            # Throughput Deltas: (Próximo - Atual) / (Atual + eps)
            feat['delta_tp_up'] = (next_seg['tp_up_median'] - curr_seg['tp_up_median']) / (curr_seg['tp_up_median'] + eps)
            feat['delta_tp_down'] = (next_seg['tp_down_median'] - curr_seg['tp_down_median']) / (curr_seg['tp_down_median'] + eps)
            
            # Packet Loss Delta: Próximo (1) - Atual (0)
            feat['delta_pl'] = next_seg['pl_mean'] - curr_seg['pl_mean']
            
            # Sync_score: Usa o valor do momento da quebra (o metrics_at_threshold do ponto de transição)
            feat['Sync_score'] = curr_seg['metrics_at_threshold'] / 5.0
            
            features_list.append(feat)

    # 3. Salva o resultado
    df_features = pd.DataFrame(features_list)
    df_features.to_csv(output_path, index=False)
    print(f"Features salvas em: {output_path}")
        
if __name__ == "__main__":
    scenario = "NDT_AGO_OUT"
    # method = "cusum"  # or "cpd"
    # methods = ["cusum", "pelt", "vwcd"]
    threshold = 0.95  # Only needed for vwcd
    # for method in methods:
    #     result_df = segment_series_by_changepoints(scenario, method, threshold)
    method = "vwcd"
    # segment_series_by_threshold(scenario, method, threshold, ref_metric="rtt_down")
    feature_extraction(scenario, method, threshold, ref_metric="rtt_down")