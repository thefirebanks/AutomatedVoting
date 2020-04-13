"""
@author: Daniel Firebanks-Quevedo

python main.py -f data/cubic_nC3_nV20_nP100_imC40.profiles -c 3 -e 200 -opt Adam -lr 0.001 -tp 0.2 -s 69420

"""

from automated_voting.algorithms.neural_network import AVNet
from automated_voting.voting.profiles import load_dataset, AVProfile
from automated_voting.voting.election import evaluate_baselines
import random
import pandas as pd

import time
import argparse

parser = argparse.ArgumentParser(description='Define parameters for AVNet neural network')
parser.add_argument('-f', '--file_name', metavar='fn', type=str,
                        default='../../automated_voting/voting/profile_data/spheroid_nC5_nV500_nP100.profiles',
                        # required=True,
                        help='Name of the file to load dataset from')

parser.add_argument('-c', '--n_candidates', metavar='cand', type=int,
                        required=True,
                        help='Number of candidates')

parser.add_argument('-v', '--n_voters', metavar='votrs', type=int,
                        required=True,
                        help='Number of voters')

parser.add_argument('-e', '--epochs', metavar='epochs', type=int,
                    default=20,
                    help="Number of epochs")

parser.add_argument('-opt', '--optimizer', metavar='opt', type=str,
                    default="SGD",
                    help="choose between SGD, Adam or Adagrad")

parser.add_argument('-lr', '--l_rate', metavar='lrate', type=float,
                    default=0.001,
                    help="learning rate for network")

parser.add_argument('-tp', '--t_perc', metavar='tperc', type=float,
                    default=0.2,
                    help="test percentage")

parser.add_argument('-s', '--seed', metavar='seed', type=float,
                    default=69420,
                    help="seed")

parser.add_argument('-arch', '--nn_architecture', metavar='arch', type=int,
                    default=1,
                    help="different architecture types range between [1-5]")

args = parser.parse_args()


def train_test_split(X, test_size, seed):
    random.seed(seed)
    random.shuffle(X)
    test_start_idx = int(len(X) - test_size*len(X))
    print("Length of profiles:", len(X), "Index of test", test_start_idx)
    return X[:test_start_idx], X[test_start_idx:]

def write_output(run_name, baseline_res, train_avnet_res, test_avnet_res):
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
    for rule, res in baseline_res.items():
        df_dict["Voting Rule"].append(rule)
        df_dict["Train Condorcet fraction"].append(0)
        df_dict["Train Condorcet rate"].append(res["Condorcet"])
        df_dict["Train Majority fraction"].append(0)
        df_dict["Train Majority rate"].append(res["Majority"])
        df_dict["Train Plurality fraction"].append(0)
        df_dict["Train Plurality rate"].append(0)
        df_dict["Train Mean IM fraction"].append(0)
        df_dict["Train Mean IM rate"].append(0)
        df_dict["Train Mean IM score"].append(0)

        df_dict["Test Condorcet fraction"].append(0)
        df_dict["Test Condorcet rate"].append(0)
        df_dict["Test Majority fraction"].append(0)
        df_dict["Test Majority rate"].append(0)
        df_dict["Test Plurality fraction"].append(0)
        df_dict["Test Plurality rate"].append(0)
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

def main():
    # Step 1: Define basic parameters
    n_candidates = args.n_candidates
    n_features = n_candidates * n_candidates
    n_voters = args.n_voters
    epochs = args.epochs
    l_rate = args.l_rate
    opt = args.optimizer
    seed = args.seed
    t_perc = args.t_perc
    nn_arch = args.nn_architecture

    run_name = args.file_name.replace("data/", "").replace(".profiles", "")
    run_name += f"_ep={epochs}_arch={nn_arch}"

    # Step 2: Load a dataset - a list of AVProfiles
    profiles = load_dataset(args.file_name)

    # Step 2.1: Split into train test sets
    profiles_train, profiles_test = train_test_split(profiles, test_size=t_perc, seed=seed)

    # Step 3: Run baseline models to compare later
    start_b = time.time()

    baseline_results = evaluate_baselines(profiles_test)

    elapsed_b = time.time() - start_b
    print("Baselines time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_b)))

    # Step 4: Define model and start training loop
    start_n = time.time()
    av_model = AVNet(n_features, n_candidates, n_voters, inp_shape=(1, n_features),
                     opt=opt, l_rate=l_rate, arch=nn_arch)

    # This is the equivalent to model.fit() or model.train()
    av_model.train(profiles_train, epochs)
    train_avnet_results = av_model.get_results()

    # This is more of the equivalent of model.predict() - the reason being that the train method
    av_model.train(profiles_test, epochs=1)

    test_avnet_results = av_model.get_results()

    elapsed_n = time.time() - start_n
    print("Train time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_n)))

    write_output(run_name, baseline_results, train_avnet_results, test_avnet_results)


    print("===================================================")
    print("Baselines results")
    print("---------------------------------------------------")

    for rule, res in baseline_results.items():
        print("Rule:", rule)
        print("IM Mean Score:", res["IM_score_fraction"], res["IM_mean"])
        print("IM Mean CCE Score:", res["IM_CCE_mean"])
        print("-------")
    print("===================================================")

if __name__ == "__main__":
    main()
