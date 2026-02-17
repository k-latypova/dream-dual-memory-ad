import os
import json
import numpy as np
import argparse

def aggregate_results(path: str):
    #seeds = os.listdir(os.path.join(path, 'train'))
    seeds = os.listdir(path)
    aucs = []
    auprs = []
    f1s = []
    for seed in seeds:
        #seed_path = os.path.join(path, 'train', seed)
        seed_path = os.path.join(path, seed)
        if not os.path.isdir(seed_path):
            print(f"Skipping {seed_path}, not a directory. Weird!")
            continue
        # Check if the test_results.json file exists
        if not os.path.exists(os.path.join(seed_path, 'test_results.json')):
            print(f"Skipping {seed_path}, test_results.json not found.")
            continue
        with open(os.path.join(seed_path, 'test_results.json'), 'r') as f:
            test_results = json.load(f)
            aucs.append(test_results['auc'])
            auprs.append(test_results['aupr'])
            f1s.append(test_results['f1'])
   
    aucs = np.array(aucs)
    auprs = np.array(auprs)
    f1s = np.array(f1s)
    test_results = {
        "auc": {
            "mean": np.mean(aucs),
            "std": np.std(aucs)
        },
        "aupr": {
            "mean": np.mean(auprs),
            "std": np.std(auprs)
        },
        "f1": {
            "mean": np.mean(f1s),
            "std": np.std(f1s)
        }
    }
    # Save the aggregated results to a JSON file
    with open(os.path.join(path, 'aggregated_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)
    print(f"Aggregated results saved to {os.path.join(path, 'aggregated_results.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate results from multiple seeds.')
    parser.add_argument('--path', type=str, required=True, help='Path to the directory containing seed directories.')
    args = parser.parse_args()
    
    aggregate_results(args.path)