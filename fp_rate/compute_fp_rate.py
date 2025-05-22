# FALSE POSITIVES
# PREFIX = "checker_1_"
# NAME = "img_cls_1"
global_parent_dir = "./workloads"
MAX_LEN = 10000
import os
import pandas as pd
import yaml

SETUP_FILE = "setups.yml"

from traincheck.checker import parse_checker_results
from traincheck.invariant.base_cls import read_inv_file, Invariant
from ae_fp import get_checker_output_dir, get_setup_key

def get_fp_results(result_folders):
    # get all the inv files under the folder

    all_fp_rates = []
    for folder in result_folders:
        complete_path = os.path.join(global_parent_dir, folder)
        if not os.path.exists(complete_path):
            continue
        files_in_folder = os.listdir(os.path.join(global_parent_dir, folder))
        inv_file = "invariants.json"
        invs = read_inv_file(os.path.join(global_parent_dir, folder, inv_file))

        sub_folder = [f for f in files_in_folder if f != inv_file][0]
        result_files = os.listdir(os.path.join(global_parent_dir, folder, sub_folder))
        failed_file = [f for f in result_files if "failed" in f][0]
        failed_results = parse_checker_results(os.path.join(global_parent_dir, folder, sub_folder, failed_file))
        
        fp_rate = len(failed_results) / len(invs)
        all_fp_rates.append(fp_rate)

    return all_fp_rates

if __name__ == "__main__":
    config= yaml.load(open(os.path.join(global_parent_dir, SETUP_FILE), "r"), Loader=yaml.FullLoader)
    training_setups = config["training_setups"]
    validation_setups = config["validation_setups"]
    # now find the corresponding checker results
    setup_to_checker_results = {}
    for setup_config in training_setups:
        setup_name = get_setup_key(setup_config)
        setup_to_checker_results[setup_name] = []
        for validation_program in validation_setups:
            setup_to_checker_results[setup_name].append(get_checker_output_dir(setup_config, validation_program))

    all_checker_results = set()
    for setup_name, checker_results in setup_to_checker_results.items():
        all_checker_results.update(checker_results)

    assert len(all_checker_results) == len(training_setups) * len(validation_setups), "Not all checker results are unique, something is wrong"

    avg_fp_rates = []
    for setup, checker_results in setup_to_checker_results.items():
        fp_rate = get_fp_results(checker_results)
        if len(fp_rate) == 0:
            print(f"Warning: No results found for {setup}")
            continue
        print(f"FP rates for {setup} on different validation programs: {fp_rate}")
        avg_fp_rates.append({
            "setup": f"{len(setup)}-input",
            "fp_rate": sum(fp_rate) / len(fp_rate),
        })

    # now write the results to a csv file
    df = pd.DataFrame(avg_fp_rates)
    df.to_csv("fp_rates.csv", index=False)



    