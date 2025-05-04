# FALSE POSITIVES
# PREFIX = "checker_1_"
# NAME = "img_cls_1"
global_parent_dir = "./"
MAX_LEN = 10000
import os
import pandas as pd
import yaml

from traincheck.checker import parse_checker_results
from traincheck.invariant.base_cls import read_inv_file, Invariant

def get_applied_results(result_folders):
    # get all the inv files under the folder

    all_fp_rates = []
    for folder in result_folders:
        files_in_folder = os.listdir(os.path.join(global_parent_dir, folder))
        inv_file = "invariants.json"
        invs = read_inv_file(os.path.join(global_parent_dir, folder, inv_file))

        sub_folder = [f for f in files_in_folder if f != inv_file][0]
        result_files = os.listdir(os.path.join(global_parent_dir, folder, sub_folder))
        applied_file = [f for f in result_files if "passed" in f][0]
        applied_results = parse_checker_results(os.path.join(global_parent_dir, folder, sub_folder, applied_file))
        
        applied_rate = len(applied_results) / len(invs)
        all_fp_rates.append(applied_rate)

    return all_fp_rates

if __name__ == "__main__":
    avg_applied_rates = []
    applied_rate = get_applied_results(["checker_gcn-torch251"])
    avg_applied_rates.append({
        "setup": "gcn-torch251",
        "applied_rate": sum(applied_rate) / len(applied_rate),
    })

    # now write the results to a csv file
    df = pd.DataFrame(avg_applied_rates)
    df.to_csv("applied_rates.csv", index=False)



    