"""
@author: Daniel Firebanks-Quevedo

python main.py -f data/cubic_nC3_nV10_nP100_imC40.profiles -c 3 -e 100 -opt Adam -lr 0.01

"""

from automated_voting.algorithms.neural_network import AVNet
from automated_voting.voting.profiles import load_dataset, AVProfile
from automated_voting.voting.election import evaluate_baselines
import random


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
parser.add_argument('-e', '--epochs', metavar='epochs', type=int,
                    default=20,
                    help="Number of epochs")

parser.add_argument('-opt', '--optimizer', metavar='opt', type=str,
                    default="SGD",
                    help="choose between SGD, Adam or Adagrad")

parser.add_argument('-lr', '--l_rate', metavar='lrate', type=float,
                    default=0.01,
                    help="learning rate for network")

parser.add_argument('-tp', '--t_perc', metavar='tperc', type=float,
                    default=0.2,
                    help="test percentage")

parser.add_argument('-s', '--seed', metavar='seed', type=float,
                    default=69420,
                    help="seed")

args = parser.parse_args()


def train_test_split(X, test_size, seed):
    random.shuffle(X, seed=seed)
    test_start_idx = len(X) - test_size*len(X)
    return X[:test_start_idx], X[test_start_idx]


def main():
    # Step 1: Define basic parameters
    n_candidates = args.n_candidates
    n_features = n_candidates * n_candidates
    epochs = args.epochs
    l_rate = args.l_rate
    opt = args.optimizer
    seed = args.seed
    t_perc = args.t_perc

    # Step 2: Load a dataset - a list of AVProfiles
    profiles = load_dataset(args.file_name)

    # Step 2.1: Split into train test sets
    profiles_train, profiles_test = train_test_split(profiles, test_size=t_perc, seed=seed)

    # Step 3: Run baseline models to compare later
    start_b = time.time()

    baseline_results = evaluate_baselines(profiles)

    elapsed_b = time.time() - start_b
    print("Baselines time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_b)))

    # Step 4: Define model and start training loop
    start_n = time.time()
    av_model = AVNet(n_features, n_candidates, inp_shape=(1, n_features), opt=opt, l_rate=l_rate)

    # This is the equivalent to model.fit() or model.train()
    av_model.train(profiles_train, epochs)

    # This is more of the equivalent of model.predict() - the reason being that the train method
    av_model.train(profiles_test, epochs=1)

    _ = av_model.get_results()

    elapsed_n = time.time() - start_n
    print("Train time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_n)))

    print("===================================================")
    print("Baselines results")
    print("---------------------------------------------------")

    for rule, res in baseline_results.items():
        print("Rule:", rule)
        print("IM Mean Score:", res["IM_mean"])
        print("IM Mean CCE Score:", res["IM_CCE_mean"])
        print("-------")
    print("===================================================")

if __name__ == "__main__":
    main()
