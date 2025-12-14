import json
import os
from .config import RESULTS_DIR
from .visualization import plot_metrics_vs_eps

def main():
    json_path = os.path.join(RESULTS_DIR, "experiment_results.json")
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path, "r") as f:
        results = json.load(f)
    
    print(f"Loaded {len(results)} records.")
    
    try:
        plot_path = plot_metrics_vs_eps(results)
        print(f"Successfully generated plot at: {plot_path}")
    except Exception as e:
        print(f"Error generating plot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
