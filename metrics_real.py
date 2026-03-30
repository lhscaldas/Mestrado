import os
import pandas as pd
from pathlib import Path
from collections import defaultdict

def count_changepoints(scenario: str, method: str, threshold: float = None):  # type: ignore
    data = defaultdict(dict)
    scenario_path = Path(os.path.join('results',scenario))
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

    records = []
    for (client, server), var_counts in data.items():
        row = {'client': client, 'server': server}
        row.update(var_counts)
        records.append(row)
        
    final_df = pd.DataFrame(records)
    
    if not final_df.empty:
        for var in measured_variables:
            if var not in final_df.columns:
                final_df[var] = 0
                
        column_order = ['client', 'server'] + list(measured_variables)
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
    scenario = "NDT_AGO_OUT"
    # method = "cusum"  # or "cpd"
    methods = ["cusum", "pelt", "vwcd"]
    threshold = 0.95  # Only needed for vwcd
    for method in methods:
        result_df = count_changepoints(scenario, method, threshold)