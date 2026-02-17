import os
import json
import numpy as np
import argparse

def aggregate_by_class(path: str):
    classes = os.listdir(path)

    with open(os.path.join(path, 'test_results.csv'), 'w') as f:
        f.write('class,auc_mean,auc_std,aupr_mean,aupr_std,f1_mean,f1_std\n')
    for class_idx in classes:
        class_path = os.path.join(path, class_idx)
        if not os.path.isdir(class_path):
            print(f"Skipping {class_path}, not a directory. Weird!")
            continue
        # Check if the test_results.json file exists
        if not os.path.exists(os.path.join(class_path, 'aggregated_results.json')):
            print(f"Skipping {class_path}, aggregated_results.json not found.")
            continue
        with open(os.path.join(class_path, 'aggregated_results.json'), 'r') as f:
            test_results = json.load(f)
            with open(os.path.join(path, 'test_results.csv'), 'a') as csv_file:
                csv_file.write(f"{class_idx},{test_results['auc']['mean']},{test_results['auc']['std']},"
                               f"{test_results['aupr']['mean']},{test_results['aupr']['std']},"
                               f"{test_results['f1']['mean']},{test_results['f1']['std']}\n")
            # aucs.append(test_results['auc'])
            # auprs.append(test_results['aupr'])
            # f1s.append(test_results['f1'])
   
    
    print(f"Aggregated results saved to {os.path.join(path, 'test_results.csv')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate results from multiple seeds.')
    parser.add_argument('--path', type=str, required=True, help='Path to the directory containing seed directories.')
    args = parser.parse_args()
    
    aggregate_by_class(args.path)