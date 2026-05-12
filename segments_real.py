import os
import pandas as pd
from pathlib import Path

def segment_series(scenario, method, threshold, ref_metric):
    input_dir = os.path.join("results", scenario, "full", method)
    output_dir = os.path.join("segments", scenario)
    os.makedirs(output_dir, exist_ok=True)
    
    prob_cols = ['P_rtt_down', 'P_tp_down', 'P_rtt_up', 'P_tp_up', 'P_pl']
    value_metrics = ['rtt_down', 'tp_down', 'rtt_up', 'tp_up', 'pl']
    
    ref_col = f"P_{ref_metric}" if ref_metric != "full" else None
    
    all_segments_data = []

    if not os.path.exists(input_dir):
        return

    for file_name in [f for f in os.listdir(input_dir) if f.endswith('.csv')]:
        name_part = file_name.replace('.csv', '')
        
        parts = name_part.split('_', 2)
        client = parts[0] if len(parts) > 0 else name_part
        server = parts[1] if len(parts) > 1 else "unknown"
        block = parts[2] if len(parts) > 2 else "unknown"

        df = pd.read_csv(os.path.join(input_dir, file_name))
        
        if ref_metric != "full" and ref_col not in df.columns:
            continue

        for col in prob_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if ref_metric == "full":
            existing_prob_cols = [col for col in prob_cols if col in df.columns]
            if not existing_prob_cols:
                continue
            df['is_change'] = df[existing_prob_cols].ge(threshold).any(axis=1)
        else:
            df['is_change'] = df[ref_col] >= threshold
        
        df['segment_id'] = df['is_change'].shift(fill_value=False).cumsum()

        groups = list(df.groupby('segment_id'))
        for i, (seg_id, group) in enumerate(groups):
            if group.empty:
                continue
                
            current_timestamp = group['timestamp'].iloc[-1]
            
            metrics_above_threshold = 0
            last_row = group.iloc[-1]
            for col in prob_cols:
                if col in last_row and last_row[col] >= threshold:
                    metrics_above_threshold += 1

            segment_stats = {
                'client': client,
                'server': server,
                'block': block,
                'segment_number': f"{i + 1:02d}",
                'final_timestamp': current_timestamp,
                'start_timestamp': group['timestamp'].iloc[0],
                'metrics_at_threshold': metrics_above_threshold,
                'n_points': len(group)
            }

            # verificar se pl varia entre 0 e 1 ou 0 e 100
            if 'pl' in group.columns:
                pl_values = pd.to_numeric(group['pl'], errors='coerce')
                if pl_values.max() > 1.0:
                    group['pl'] = pl_values / 100.0
            

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
    folder_path = os.path.join("segments", scenario)
    file_name = f"{method}_{ref_metric}_{threshold}.csv"
    input_path = os.path.join(folder_path, file_name)
    output_path = os.path.join("features", scenario, f"features_{file_name}")
    
    if not os.path.exists(input_path):
        print(f"Erro: Arquivo {input_path} not found.")
        return
    
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.read_csv(input_path)
    
    df = df[df['n_points'] >= 6].copy()
    
    df['series_id'] = df['client'] + "_" + df['server'] + "_" + df['block'].astype(str)
    
    series_counts = df.groupby('series_id').size()
    series_with_changes = series_counts[series_counts > 1].index
    df_filtered = df[df['series_id'].isin(series_with_changes)].copy()
    
    df_filtered = df_filtered.sort_values(['client', 'server', 'block', 'segment_number'])
    
    eps = 1e-9
    features_list = []

    for (client, server, block), group in df_filtered.groupby(['client', 'server', 'block']):
        group = group.reset_index(drop=True)
        
        for i in range(len(group) - 1):
            curr_seg = group.iloc[i]
            next_seg = group.iloc[i+1]
            
            feat = {
                'client': curr_seg['client'],
                'server': curr_seg['server'],
                'block': curr_seg['block'],
                'segment_number': curr_seg['segment_number'],
                'timestamp': curr_seg['final_timestamp']
            }
            
            feat['d_rtt_up_rel'] = (next_seg['rtt_up_median'] - curr_seg['rtt_up_median']) / (curr_seg['rtt_up_median'] + eps)
            feat['d_rtt_down_rel'] = (next_seg['rtt_down_median'] - curr_seg['rtt_down_median'])  / (curr_seg['rtt_down_median'] + eps)

            feat['d_rtt_up_abs'] = (next_seg['rtt_up_median'] - curr_seg['rtt_up_median'])
            feat['d_rtt_down_abs'] = (next_seg['rtt_down_median'] - curr_seg['rtt_down_median'])
            
            feat['d_tp_up'] = (next_seg['tp_up_median'] - curr_seg['tp_up_median']) / (curr_seg['tp_up_median'] + eps)
            feat['d_tp_down'] = (next_seg['tp_down_median'] - curr_seg['tp_down_median']) / (curr_seg['tp_down_median'] + eps)
            
            feat['d_pl'] = next_seg['pl_mean'] - curr_seg['pl_mean']
            
            feat['sync_score'] = curr_seg['metrics_at_threshold'] / 5.0
            
            features_list.append(feat)

    if features_list:
        df_features = pd.DataFrame(features_list)
        df_features.to_csv(output_path, index=False)
        print(f"Features salvas em: {output_path}")
    else:
        print("Nenhuma feature gerada.")
        
if __name__ == "__main__":
    scenario = "NDT"
    threshold = 0.95
    method = "vwcd_fp1"
    ref_metric = "rtt_down"
    segment_series(scenario, method, threshold, ref_metric)
    feature_extraction(scenario, method, threshold, ref_metric)