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

if __name__ == "__main__":
    scenario = "NDT_OUT"
    # method = "cusum"  # or "cpd"
    methods = ["cusum", "pelt", "vwcd"]
    threshold = 0.95  # Only needed for vwcd
    for method in methods:
        result_df = segment_series_by_changepoints(scenario, method, threshold)