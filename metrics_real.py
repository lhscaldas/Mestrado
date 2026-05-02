import os
import pandas as pd
from pathlib import Path
from collections import defaultdict

def count_changepoints(scenario: str, method: str, threshold: float = None):  # type: ignore
    data = defaultdict(dict)
    scenario_path = Path(os.path.join('results',scenario))
    series_path = Path(os.path.join('series', scenario))
    measured_variables = set()
    
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
                measured_variables.add(variable)
                for csv_file in target_dir.glob("*.csv"):
                    name_parts = csv_file.stem.split('_', 1)
                    if len(name_parts) == 2:
                        client, server = name_parts
                        
                        df = pd.read_csv(csv_file)
                        
                        if method == 'vwcd':
                            if 'P_theta_ge_Tstar' in df.columns and threshold is not None:
                                count = (df['P_theta_ge_Tstar'] > threshold).sum()
                                data[(client, server)][variable] = count
                        else:
                            if 'CP' in df.columns:
                                count = (df['CP'] == 1).sum()
                                data[(client, server)][variable] = count

                        series_csv_path = series_path / variable / csv_file.name
                        if series_csv_path.exists():
                            df_series = pd.read_csv(series_csv_path)
                            
                            col_name = variable if variable in df_series.columns else df_series.columns[-1]
                            
                            data[(client, server)][f"{variable}_min"] = df_series[col_name].min()
                            data[(client, server)][f"{variable}_mean"] = df_series[col_name].mean()
                            data[(client, server)][f"{variable}_max"] = df_series[col_name].max()
                            data[(client, server)][f"{variable}_std"] = df_series[col_name].std() if len(df_series[col_name]) > 1 else 0.0

    records = []
    for (client, server), var_counts in data.items():
        row = {'client': client, 'server': server}
        row.update(var_counts)
        records.append(row)
        
    final_df = pd.DataFrame(records)

    if not final_df.empty:
        column_order = ['client', 'server']
        for var in sorted(measured_variables):
            column_order.extend([var, f"{var}_min", f"{var}_mean", f"{var}_max", f"{var}_std"])
            
        for col in column_order:
            if col not in final_df.columns:
                final_df[col] = 0
                
        final_df = final_df[column_order].fillna(0)
        
        out_dir = Path(f"metrics/{scenario_path.name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        if method == 'vwcd':
            out_path = out_dir / f"{method}_{threshold}.csv"
        else:
            out_path = out_dir / f"{method}.csv"
            
        final_df.to_csv(out_path, index=False)
        
    return final_df

if __name__ == "__main__":
    scenario = "NDT_NOV_ABR"
    # method = "cusum"  # or "cpd"
    # methods = ["cusum", "pelt", "vwcd"]
    methods = ["vwcd"]
    threshold = 0.95  # Only needed for vwcd
    for method in methods:
        result_df = count_changepoints(scenario, method, threshold)