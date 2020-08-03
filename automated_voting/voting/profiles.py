"""
@author: Daniel Firebanks-Quevedo

For one distribution
- python profiles.py -p 20 -c 3 -v 10 -nd 1 -d gaussian -s 69420

For all distributions
- python profiles.py -p 100 -c 5 -v 40 -nd 7 -s 69420

"""
import numpy as np
from itertools import product
from pandas import DataFrame
from svvamp import PopulationCubicUniform, PopulationEuclideanBox, PopulationGaussianWell, \
    PopulationLadder, PopulationVMFHypercircle, PopulationVMFHypersphere, PopulationSpheroid
from whalrus.profile.Profile import Profile
from whalrus.rule.RuleCondorcet import RuleCondorcet
# from whalrus.rule.RuleBorda import RuleBorda
from pickle import dump, load
from tqdm import tqdm
from whalrus.ballot.BallotOrder import BallotOrder
import argparse
import re
import ast
import cProfile
import random

DATASET_FOLDER = "../../data"

# Add this method so that we can sort ballot objects
def __lt__(ballot1, ballot2):
    return list(ballot1) < list(ballot2)


def __gt__(ballot1, ballot2):
    return list(ballot1) > list(ballot2)


setattr(BallotOrder, "__lt__", __lt__)
setattr(BallotOrder, "__gt__", __gt__)


class AVProfile(Profile):
    def __init__(self, n_voters, origin="distribution", params="spheroid", candidates=None, IM_tuples=None):

        # Make sure we have the right input
        if origin == "distribution" and candidates is None:
            raise Exception("Candidates shouldn't be empty if we are creating a profile from distribution!")
        elif origin == "data" and params == "spheroid":
            raise Exception("Params should include csv file name, such as sample_data_profile.csv!")

        # Generate the ballots
        if origin == "distribution":

            # Create a population object
            if params == "cubic":
                pop = PopulationCubicUniform(V=n_voters, C=len(candidates))
            elif params == "euclidean":
                pop = PopulationEuclideanBox(V=n_voters, C=len(candidates), box_dimensions=[1])
            elif params == "gaussian":
                pop = PopulationGaussianWell(V=n_voters, C=len(candidates), sigma=[1], shift=[10])
            elif params == "ladder":
                pop = PopulationLadder(V=n_voters, C=len(candidates), n_rungs=5)
            elif params == "VMFHypercircle":
                pop = PopulationVMFHypercircle(V=n_voters, C=len(candidates), vmf_concentration=[10],
                                               vmf_pole=np.random.randint(0, len(candidates),
                                                                                  len(candidates)))
            elif params == "VMFHypersphere":
                pop = PopulationVMFHypersphere(V=n_voters, C=len(candidates),
                                               vmf_concentration=[50, 50],
                                               vmf_probability=None,
                                               vmf_pole=[
                                                   np.random.randint(0, len(candidates), len(candidates)),
                                                   np.random.randint(0, len(candidates), len(candidates))],
                                               stretching=1)
            else:
                pop = PopulationSpheroid(V=n_voters, C=len(candidates))

            self._preferences = pop.preferences_rk
            pop.labels_candidates = self._labels_candidates = candidates

        elif origin == "data":  # params = filename
            preferences = np.genfromtxt(params, delimiter=',', dtype=str)
            self._labels_candidates = preferences[0, :]
            self._preferences = preferences[1:, :].astype(int)
        else:
            raise Exception("Insert either distribution or data as the parameter origin!")

        # Basic properties
        self._n_candidates = len(self._labels_candidates)
        self._n_voters = n_voters
        self._candidate_map = dict(zip(self.candidates, list(range(self.n_candidates))))

        # Create a labeled version of the ranking from the population -> Allows us to initialize a Profile object
        self._candidate_rank_map = self.get_candidate_rank_map()
        self._name_ballots = self.get_named_ballots()
        self._idx_ballots = [[self.candidate_map[c] for c in ballot] for ballot in self._name_ballots]

        # Create a profile object - automatically creates ordered ballots
        super().__init__(self._name_ballots)

        # Create a dataframe of ballots and how many people voted for them
        self._ballot_df = self.to_ballot_dataframe()

        # Create a dataframe representation of the profile so we can print it out
        self._rank_matrix = self.to_count_matrix()
        self._rank_df = self.to_count_dataframe()
        # self._tournament_matrix = self.to_tournament_matrix()

        # Create one-hot vectors of established candidates (Condorcet, Majority, etc.)
        self._condorcet_w, self._condorcet_w_vector = self.get_condorcet()
        self._majority_w, self._majority_w_vector = self.get_majority()
        self._plurality_w, self._plurality_w_vector = self.get_plurality()

        # Uncomment this if we want to store all the IM ballots in here as a 2d array - IM_ballots[IM_matrix][ballot_i]
        self._IM_ballots = dict()
        self._all_possible_IM_alterations = IM_tuples

        # Create a sample of simulated profiles for IM
        self._IM_rank_matrices = self.generate_IM_rank_matrices()

        # Create a sample of simulated profiles for IIA
        # self._IIA_profiles = self.create_IIA_dict()

    def flatten_rank_matrix(self, rank_matrix=None) -> np.array:
        """
            Input matrix for neural network
            - Flatten dataset for network input layer purposes
            - Reshape to (n_features, 1) for tensor input, then transpose to make it n_features columns,
               where n_features = n_candidates**2 """

        if rank_matrix is None:
            return self._rank_matrix.flatten('F').reshape(self.n_candidates * self.n_candidates, 1).T
        else:
            return rank_matrix.flatten('F').reshape(self.n_candidates * self.n_candidates, 1).T

    def get_condorcet(self) -> (str, np.array):
        """ Check if a profile contains a condorcet winner and returns a one-hot vector representing them.
            Returns:
                - The name of the winner as a string, or "No Condorcet Winner" otherwise
                - A numpy array of size n_candidates, where candidate i is 1 if they are a Condorcet winner,
                and the rest are 0. If there is no Condorcet winner, then all elements are 0
                 """

        condorcet = RuleCondorcet(self)

        if len(condorcet.cowinners_) == 1:
            winner = next(iter(condorcet.cowinners_))
            winner_idx = self.candidates.index(winner)

            one_hot = np.zeros(shape=(self.n_candidates,))
            one_hot[winner_idx] = 1

            return winner, one_hot

        return "No Condorcet Winner", np.zeros(shape=(self.n_candidates,))

    def get_majority(self) -> (str, np.array):
        """ Check if a profile contains a majority winner and returns a one-hot vector representing them.
            Returns:
                - The name of the winner as a string, or "No Majority Winner" otherwise
                - A numpy array of size n_candidates, where candidate i is 1 if they are a majority winner,
                and the rest are 0. If there is no majority winner, then all elements are 0 """

        # Get the first row as those are the counts for how many times a candidate was ranked first
        first_candidates = self._rank_matrix[0]

        for candidate_idx, candidate_votes in enumerate(first_candidates):
            if candidate_votes >= 0.5 * self.n_voters:
                one_hot = np.zeros(shape=(self.n_candidates,))
                one_hot[candidate_idx] = 1
                return self.candidates[candidate_idx], one_hot

        return "No Majority Winner", np.zeros(shape=(self.n_candidates,))

    def get_plurality(self) -> (str, np.array):
        """ Returns the plurality winner of a given profile. If there are ties, it randomly returns one of the candidates
            with equal probability

            Returns:
                - The name of the winner
                - A numpy array of size n_candidates, where candidate i is 1 if they are a majority winner,
                and the rest are 0 """

        # Get the plurality winner from the first row of the rank matrix, break ties randomly
        winner_idx = np.random.choice(np.flatnonzero(self._rank_matrix[0] == self._rank_matrix[0].max()))
        winner = self.candidates[winner_idx]

        # Get the one-hot vector
        one_hot = np.zeros(shape=(self.n_candidates,))
        one_hot[winner_idx] = 1

        return winner, one_hot

    def get_candidate_rank_map(self) -> list:
        """ Convert profile to a list of {candidate: rank} per voter
            This method should only be used at initialization of the profile """

        ordered_prof = []

        for ballot in self._preferences:
            ordered_ballot = {}
            for candidate, rank in zip(self._labels_candidates, ballot):
                ordered_ballot[candidate] = rank

            ordered_prof.append(ordered_ballot)

        sorted_dicts = [dict(sorted(profile.items(), key=lambda kv: kv[1])) for profile in ordered_prof]

        return sorted_dicts

    def get_named_ballots(self) -> list:
        return [list(m.keys()) for m in self._candidate_rank_map]

    # def to_tournament_matrix(self) -> array:
    #     return array([])

    def to_count_matrix(self) -> np.array:
        """ Create a matrix representation of the profile,
        where matrix[c][r] represent the frequency of candidate c in rank r """

        self._idx_ballots = [[self.candidate_map[c] for c in ballot] for ballot in self.name_ballots]

        matrix_rank = [[0] * len(self._idx_ballots[0]) for _ in self._idx_ballots[0]]

        for ballot in self._idx_ballots:
            for candidate, rank in enumerate(ballot):
                matrix_rank[candidate][rank] += 1

        return np.array(matrix_rank).T

    def to_count_dataframe(self) -> DataFrame:
        """ Creates a dataframe representation of the profile from the matrix representation """
        data_dict = dict()
        self._rank_matrix = self.to_count_matrix()

        # Create dictionary for dataframe
        for i in range(len(self._rank_matrix)):
            data_dict[f"Rank {i + 1}"] = self._rank_matrix[:, i]

        df = DataFrame(data_dict, index=self.candidates)

        return df

    def to_ballot_dataframe(self) -> DataFrame:
        """ Creates a dataframe where the columns represent the ballots and the top row represents
        the number of candidates that voted for a particular candiate """

        # {Ballot: # of people that have it}
        ballot_counts = dict()
        for ballot in self.name_ballots:
            ballot_str = tuple(ballot)
            if ballot_str in ballot_counts:
                ballot_counts[ballot_str] += 1
            else:
                ballot_counts[ballot_str] = 1

        # Now, build the dataframe
        header = []
        rows = []

        # Get the candidates as their index
        for ballot, count in ballot_counts.items():
            # The count will be the header
            header.append(count)
            rows.append(ballot)

        # We will transpose the rows to format ballots properly
        columns = np.array(rows).T
        df = DataFrame(columns, columns=header)

        return df

    def new_ballots(self, c_down, c_up, c_down_rank, c_up_rank):
        """
            Given two candidates c_down, c_up and two ranks c_down_rank, c_up_rank, we want to find the first ballot
            that has c_down in c_down rank and c_up in c_up_rank, and switch them. This will update the list of ballots
            with a new BallotOrder object.
        """
        self._name_ballots = [list(ballot) for ballot in self.ballots]
        new_ballots = [ballot for ballot in self.ballots]
        c_down, c_up = self.candidates[c_down], self.candidates[c_up]

        # For every ballot that we have, check if it is in the ordering that we want
        for i in range(len(self._name_ballots)):

            if self._candidate_rank_map[i][c_up] == c_up_rank and self._candidate_rank_map[i][c_down] == c_down_rank:
                new_ballot_l = [c for c in self._name_ballots[i]]

                # Swap candidates at a given position
                new_ballot_l[c_up_rank], new_ballot_l[c_down_rank] = c_down, c_up

                # Create a new ballot order obj and replace!
                new_ballots[i] = BallotOrder(new_ballot_l)
                break

        return new_ballots

    def generate_IM_rank_matrices(self):
        # 1. Generate all possible IM combinations from the cached IM tuples
        IM_matrices = []
        i = 0

        for (candidate_up, candidate_up_rank), (candidate_down, candidate_down_rank) in self._all_possible_IM_alterations:
            new_rank_matrix = self._rank_matrix.copy()
            new_rank_matrix[candidate_up, candidate_up_rank] -= 1
            new_rank_matrix[candidate_down, candidate_down_rank] -= 1

            new_rank_matrix[candidate_up, candidate_down_rank] += 1
            new_rank_matrix[candidate_down, candidate_up_rank] += 1


            IM_matrices.append(new_rank_matrix)

            # Update the ballots for that particular IM matrix
            self._IM_ballots[f"IM_{i}"] = (self.new_ballots(candidate_down, candidate_up, candidate_down_rank, candidate_up_rank))
            i +=1

        # 2. Remove duplicate IM matrices from the list
        # Step 1: Convert arrays to a string
        string_repr = [str(matrix) for matrix in IM_matrices]

        # Step 2: Associate them with a hash - it will automatically remove duplicates
        IM_hashmap = dict(zip(map(hash, string_repr), string_repr))

        # Step 3: Convert the str representations back to a numpy array
        final_IM_matrices = list(map(str2array, list(IM_hashmap.values())))

        return final_IM_matrices

    # Basic properties
    @property
    def n_candidates(self):
        return self._n_candidates

    @property
    def candidate_map(self):
        return self._candidate_map

    @property
    def n_voters(self):
        return self._n_voters

    @property
    def candidates(self):
        return self._labels_candidates

    # Ballots and Profile matrix/dataframe formats
    @property
    def rank_matrix(self):
        # We first call the to_count_matrix in case we have altered the ballots somehow
        self._rank_matrix = self.to_count_matrix()
        return self._rank_matrix

    # @property
    # def tournament_matrix(self):
    #     self._tournament_matrix = self.to_tournament_matrix()
    #     return self._tournament_matrix

    @property
    def rank_df(self):
        self._rank_df = self.to_count_dataframe()
        return self._rank_df

    # Ballot descriptors
    @property
    def name_ballots(self):
        self._name_ballots = [list(ballot) for ballot in self.ballots]
        return self._name_ballots

    @property
    def idx_ballots(self):
        self._idx_ballots = [[self.candidate_map[c] for c in ballot] for ballot in self.name_ballots]
        return self._idx_ballots

    @property
    def ranked_ballots_map(self):
        return self._candidate_rank_map

    @property
    def ballot_df(self):
        self._ballot_df = self.to_ballot_dataframe()
        return self._ballot_df

    # Profile special candidates (strings and vectors)
    @property
    def condorcet_w(self):
        return self._condorcet_w

    @property
    def condorcet_w_vector(self):
        return self._condorcet_w_vector

    @property
    def majority_w(self):
        return self._majority_w

    @property
    def majority_w_vector(self):
        return self._majority_w_vector

    @property
    def plurality_w(self):
        return self._plurality_w

    @property
    def plurality_w_vector(self):
        return self._plurality_w_vector

    # Simulated profiles for IM
    @property
    def IM_rank_matrices(self):
        return self._IM_rank_matrices

    @property
    def IM_ballots(self):
        return self._IM_ballots

def get_cartesian_product(all_candidates_idx) -> list:
    """
        Compute the cartesian product of all the possible IM profiles, as they will be the same per each new profile
    """

    # Compute the cartesian product of C x C[1:], a candidate in first place can't go up
    candidates_up = list(product(all_candidates_idx, all_candidates_idx[1:]))

    # Compute the cartesian product of C x C[:-1], a candidate in last place can't go down
    candidates_down = list(product(all_candidates_idx, all_candidates_idx[:-1]))

    # Compute the cartesian product of both combinations
    cartesian_product = list(product(candidates_up, candidates_down))

    # Delete the tuples that have the same candidate
    cartesian_product = [tuples for tuples in cartesian_product if tuples[0][0] != tuples[1][0]]

    return cartesian_product

##########################################################################
# Utility functions from: https://www.peijun.me/convert-strarray-back-to-numpy-array.html
def str2array(s):
    # Remove space after [
    s = re.sub('\[ +', '[', s.strip())
    # Replace commas and spaces
    s = re.sub('[,\s]+', ', ', s)
    s = ast.literal_eval(s)
    return np.array(s)

def number_of_duplicates(arrays):
    print("Original length is", len(arrays))
    # Step 1: Convert arrays to a string
    string_repr = [str(array) for array in arrays]

    # Step 2: Associate them with a hash - it will automatically remove duplicates
    IM_hashmap = dict(zip(map(hash, string_repr), string_repr))

    # Step 3: Convert the str array back to a numpy array
    final_IM_matrices = list(map(str2array, list(IM_hashmap.values())))

    print("Number of duplicates were", len(arrays) - len(final_IM_matrices))

    return len(final_IM_matrices)


def is_duplicate(matrix, dataset):
    for profile in dataset:
        if np.array_equal(profile._rank_matrix, matrix._rank_matrix):
            return True
    return False

##########################################################################

def generate_profile_dataset(num_profiles, n_voters, candidates, origin, params, r_seed):
    # Set the seed!
    np.random.seed(r_seed)
    dataset = []

    # We want to cache all the possible IM profiles - tuples of ((c_up, c_up_rank), (c_down, c_down_rank))
    all_candidates_idx = list(range(len(candidates)))
    IM_tuples = get_cartesian_product(all_candidates_idx)

    # for i in tqdm(range(num_profiles)):
    i = 0
    repeated = 0
    while i < num_profiles:
        new_profile = AVProfile(n_voters, origin, params, candidates, IM_tuples)
        if not is_duplicate(new_profile, dataset):
            dataset.append(new_profile)
            i += 1
        else:
            if repeated == i:
                break
            repeated += 1
            print(f"Duplicate found. Total duplicates: {repeated}. Re-generating...")

        if i % 50 == 0:
            print("Generating:", i, "profiles...")

    print(f"Generated {i} profiles, with {repeated} duplicates avoided.")
    return dataset


def store_dataset(dataset, n_candidates, n_voters, n_profiles, distr_name, im_count):
    with open(f"{DATASET_FOLDER}/{distr_name}_nC{n_candidates}_nV{n_voters}_nP{n_profiles}_imC{im_count}.profiles", "wb") as fp:
        dump(dataset, fp)
        print(f"Stored: {distr_name}_nC{n_candidates}_nV{n_voters}_nP{n_profiles}_imC{im_count}.profiles")


def load_dataset(file_name):
    with open(file_name, "rb") as rf:
        pickled_dataset = load(rf)

    print(f"Loaded: {file_name}")
    return pickled_dataset


def main():
    parser = argparse.ArgumentParser(description='Define parameters for generation of Profile datasets')

    parser.add_argument('-p', '--n_profiles', metavar='profiles', type=int,
                        required=True,
                        help='Number of profiles')

    parser.add_argument('-c', '--n_candidates', metavar='candidates', type=int,
                        required=True,
                        help='Number of candidates')

    parser.add_argument('-v', '--n_voters', metavar='voters', type=int,
                        required=True,
                        help='Number of voters')

    parser.add_argument('-nd', '--n_distributions', metavar='n_dists', type=int,
                        default="1",
                        help='Number of distributions if we want to generate dataset from multiple. Else just 1 and specify -d')

    parser.add_argument('-d', '--distribution', metavar='dists', type=str,
                        default="spheroid",
                        help='Name of distribution if we only want to generate data from one distribution. If this is the case, -nd should be set to 1')

    parser.add_argument('-s', '--r_seed', metavar='seed', type=int, default=69420,
                        help="Random seed to shuffle the list of random seeds :D ")

    args = parser.parse_args()

    candidates = ["Austin", "Brock", "Chad", "Derek", "Ethan", "Gabe", "Jack", "Liam", "Mike", "Tyler"]
    distributions = ["spheroid", "cubic", "euclidean", "gaussian", "ladder", "VMFHypercircle", "VMFHypersphere"]

    n_profiles = args.n_profiles
    n_voters = args.n_voters
    n_candidates = args.n_candidates
    candidates = candidates[:n_candidates]
    n_dists = args.n_distributions
    seed = args.r_seed

    if n_dists == 1:
        dists = args.distribution
        if dists not in distributions:
            raise Exception("Please enter spheroid, cubic, euclidean, gaussian, ladder, VMFHypercircle or VMFHypersphere for distribution name")

        print("==========================================")
        print(f"\nGenerating {dists} dataset...\n")
        print("==========================================")

        dataset = generate_profile_dataset(n_profiles, n_voters, candidates, "distribution", dists, seed)

        # Extract number of IM
        n_im = len(dataset[0].IM_rank_matrices)

        store_dataset(dataset, n_candidates, n_voters, n_profiles, dists, n_im)
        print()

    else:
        dists = distributions[:n_dists]

        for distribution in dists:
            print("==========================================")
            print(f"\nGenerating {distribution} dataset...\n")
            print("==========================================")

            dataset = generate_profile_dataset(n_profiles, n_voters, candidates,
                                               "distribution", distribution, seed)

            # Extract number of IM
            n_im = len(dataset[0].IM_rank_matrices)
            
            store_dataset(dataset, n_candidates, n_voters, n_profiles, distribution, n_im)
            print()

    # profile = AVProfile(5, origin="distribution",
    #               params="spheroid", candidates=["Adam", "Bert", "Chad"])

    # dataset = generate_profile_dataset(50, 10, ["Austin", "Brad", "Chad"], "distribution", "gaussian")
    # store_dataset(dataset, 3, 10, 50, "gaussian")


if __name__ == "__main__":
    main()
