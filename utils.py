import random
import pandas as pd

def train_test_split(X, test_size, seed):
    random.seed(seed)
    random.shuffle(X)
    test_start_idx = int(len(X) - test_size*len(X))
    print("Length of profiles:", len(X), "Index of test", test_start_idx)
    return X[:test_start_idx], X[test_start_idx:]


def write_output(run_name, train_baseline_res, test_baseline_res, train_avnet_res, test_avnet_res):
    df_dict = {"Voting Rule": [],
               "Train Condorcet fraction": [],
               "Train Condorcet rate": [],
               "Train Majority fraction": [],
               "Train Majority rate": [],
               "Train Plurality fraction": [],
               "Train Plurality rate": [],
               "Train Mean IM fraction": [],
               "Train Mean IM rate": [],
               "Train Mean IM score": [],
               "Test Condorcet fraction": [],
               "Test Condorcet rate": [],
               "Test Majority fraction": [],
               "Test Majority rate": [],
               "Test Plurality fraction": [],
               "Test Plurality rate": [],
               "Test Mean IM fraction": [],
               "Test Mean IM rate": [],
               "Test Mean IM score": [],
               }

    # Store the baseline results
    for rule, res in train_baseline_res.items():
        df_dict["Voting Rule"].append(rule)
        df_dict["Train Condorcet fraction"].append(res["Condorcet_fraction"])
        df_dict["Train Condorcet rate"].append(res["Condorcet_Score"])
        df_dict["Train Majority fraction"].append(res["Majority_fraction"])
        df_dict["Train Majority rate"].append(res["Majority_Score"])
        df_dict["Train Plurality fraction"].append(res["Plurality_fraction"])
        df_dict["Train Plurality rate"].append(res["Plurality_Score"])
        df_dict["Train Mean IM fraction"].append(res["IM_score_fraction"])
        df_dict["Train Mean IM rate"].append(res["IM_mean"])
        df_dict["Train Mean IM score"].append(res["IM_CCE_mean"])

    for rule, res in test_baseline_res.items():
        df_dict["Test Condorcet fraction"].append(res["Condorcet_fraction"])
        df_dict["Test Condorcet rate"].append(res["Condorcet_Score"])
        df_dict["Test Majority fraction"].append(res["Majority_fraction"])
        df_dict["Test Majority rate"].append(res["Majority_Score"])
        df_dict["Test Plurality fraction"].append(res["Plurality_fraction"])
        df_dict["Test Plurality rate"].append(res["Plurality_Score"])
        df_dict["Test Mean IM fraction"].append(res["IM_score_fraction"])
        df_dict["Test Mean IM rate"].append(res["IM_mean"])
        df_dict["Test Mean IM score"].append(res["IM_CCE_mean"])

    # Store the network results
    df_dict["Voting Rule"].append("AVNet")

    df_dict["Train Condorcet fraction"].append(train_avnet_res["Condorcet Score"][0])
    df_dict["Train Condorcet rate"].append(train_avnet_res["Condorcet Score"][1])
    df_dict["Train Majority fraction"].append(train_avnet_res["Majority Score"][0])
    df_dict["Train Majority rate"].append(train_avnet_res["Majority Score"][1])
    df_dict["Train Plurality fraction"].append(train_avnet_res["Plurality Score"][0])
    df_dict["Train Plurality rate"].append(train_avnet_res["Plurality Score"][1])
    df_dict["Train Mean IM fraction"].append(train_avnet_res["IM Score"][0])
    df_dict["Train Mean IM rate"].append(round(train_avnet_res["IM Score"][1], 3))
    df_dict["Train Mean IM score"].append(round(train_avnet_res["IM CCE Score"], 3))

    df_dict["Test Condorcet fraction"].append(test_avnet_res["Condorcet Score"][0])
    df_dict["Test Condorcet rate"].append(test_avnet_res["Condorcet Score"][1])
    df_dict["Test Majority fraction"].append(test_avnet_res["Majority Score"][0])
    df_dict["Test Majority rate"].append(test_avnet_res["Majority Score"][1])
    df_dict["Test Plurality fraction"].append(test_avnet_res["Plurality Score"][0])
    df_dict["Test Plurality rate"].append(test_avnet_res["Plurality Score"][1])
    df_dict["Test Mean IM fraction"].append(test_avnet_res["IM Score"][0])
    df_dict["Test Mean IM rate"].append(round(test_avnet_res["IM Score"][1], 3))
    df_dict["Test Mean IM score"].append(round(test_avnet_res["IM CCE Score"], 3))

    df = pd.DataFrame(df_dict)
    df.to_csv("Results_" + run_name + ".csv")