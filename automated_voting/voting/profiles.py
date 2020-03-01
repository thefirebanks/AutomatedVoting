from numpy import random, array, zeros, flatnonzero, genfromtxt, array_equal
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

parser.add_argument('-d', '--distribution', metavar='dists', type=str,
                        default="spheroid",
                        help='Name of distribution if we only want to generate data from one distribution. If this is the case, -nd should be set to 1')

parser.add_argument('-nd', '--n_distributions', metavar='n_dists', type=int,
                        default="1",
                        help='Number of distributions if we want to generate dataset from multiple. Else just 1 and specify -d')


args = parser.parse_args()


# Add this method so that we can sort ballot objects
def __lt__(ballot1, ballot2):
    return list(ballot1) < list(ballot2)


def __gt__(ballot1, ballot2):
    return list(ballot1) > list(ballot2)


setattr(BallotOrder, "__lt__", __lt__)
setattr(BallotOrder, "__gt__", __gt__)


class AVProfile(Profile):
    def __init__(self, n_voters, origin="distribution", params="spheroid", candidates=None):

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
                                               vmf_pole=random.random_integers(0, len(candidates),
                                                                               len(candidates)))
            elif params == "VMFHypersphere":
                pop = PopulationVMFHypersphere(V=n_voters, C=len(candidates),
                                               vmf_concentration=[50, 50],
                                               vmf_probability=None,
                                               vmf_pole=[
                                                   random.random_integers(0, len(candidates), len(candidates)),
                                                   random.random_integers(0, len(candidates), len(candidates))],
                                               stretching=1)
            else:
                pop = PopulationSpheroid(V=n_voters, C=len(candidates))

            self._preferences = pop.preferences_rk
            pop.labels_candidates = self._labels_candidates = candidates

        elif origin == "data":  # params = filename
            preferences = genfromtxt(params, delimiter=',', dtype=str)
            self._labels_candidates = preferences[0, :]
            self._preferences = preferences[1:, :].astype(int)
        else:
            raise Exception("Insert either distribution or data as the parameter origin!")

        # Basic properties
        self._n_candidates = len(self._labels_candidates)
        self._n_voters = n_voters
        self._candidate_map = dict(zip(self.candidates, list(range(self.n_candidates))))

        # TODO: This could be the same array, technically!!!!!! Depending on the profile generation method
        # Get a list of possible candidates whose score we can increase, that are not the winner or the wanted candidate
        self.all_candidates_idx = [i for i in range(self.n_candidates)]

        # Get possible ranks to increase/decrease
        self.all_ranks = [i for i in range(self.n_candidates)]

        # Create a labeled version of the ranking from the population -> Allows us to initialize a Profile object
        self._candidate_rank_map = self.get_candidate_rank_map()
        self._name_ballots = self.get_named_ballots()
        self._idx_ballots = [[self.candidate_map[c] for c in ballot] for ballot in self._name_ballots]

        # Create a profile object - automatically creates ordered ballots
        super().__init__(self._name_ballots)

        # Sort everything we have so far for ease of index accesing
        #         self.ballots.sort()
        #         self._name_ballots.sort()

        # Create a dataframe of ballots and how many people voted for them
        self._ballot_df = self.to_ballot_dataframe()

        # TODO: Modify all of these so that they use BALLOTS instead of preferences
        # Create a dataframe representation of the profile so we can print it out
        self._rank_matrix = self.to_count_matrix()
        self._rank_df = self.to_count_dataframe()
        # self._tournament_matrix = self.to_tournament_matrix()

        # Create one-hot vectors of established candidates (Condorcet, Majority, etc.)
        self._condorcet_w, self._condorcet_w_vector = self.get_condorcet()
        self._majority_w, self._majority_w_vector = self.get_majority()
        self._plurality_w, self._plurality_w_vector = self.get_plurality()

        # Extra variable to keep track of IM modified profiles - A set of tuples
        self._IM_pairs_set = set()

        # Extra variable to keep track of whether we have repeated matrices within the simulations
        self._repeated_rank_matrices = 0

        # Create a sample of simulated profiles for IM
        # TODO This could be simply the list of ballots and we update rank matrix later!!!!!!!!!!!
        self._IM_rank_matrices, self._IM_ballots = self.create_IM()

        # Create a sample of simulated profiles for IIA
        # self._IIA_profiles = self.create_IIA_dict()

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

    # Profile special candidates (strings and vectors
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

    def flatten_rank_matrix(self, rank_matrix=None) -> array:
        """
            Input matrix for neural network
            - Flatten dataset for network input layer purposes
            - Reshape to (n_features, 1) for tensor input, then transpose to make it n_features columns,
               where n_features = n_candidates**2 """

        if rank_matrix is None:
            return self.rank_matrix.flatten('F').reshape(self.n_candidates * self.n_candidates, 1).T
        else:
            return rank_matrix.flatten('F').reshape(self.n_candidates * self.n_candidates, 1).T

    def get_condorcet(self) -> (str, array):
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

            one_hot = zeros(shape=(self.n_candidates,))
            one_hot[winner_idx] = 1

            return winner, one_hot

        return "No Condorcet Winner", zeros(shape=(self.n_candidates,))

    def get_majority(self) -> (str, array):
        """ Check if a profile contains a majority winner and returns a one-hot vector representing them.
            Returns:
                - The name of the winner as a string, or "No Majority Winner" otherwise
                - A numpy array of size n_candidates, where candidate i is 1 if they are a majority winner,
                and the rest are 0. If there is no majority winner, then all elements are 0 """

        # Get the first row as those are the counts for how many times a candidate was ranked first
        first_candidates = self.rank_matrix[0]

        for candidate_idx, candidate_votes in enumerate(first_candidates):
            if candidate_votes >= 0.5 * self.n_voters:
                one_hot = zeros(shape=(self.n_candidates,))
                one_hot[candidate_idx] = 1
                return self.candidates[candidate_idx], one_hot

        return "No Majority Winner", zeros(shape=(self.n_candidates,))

    # TODO: Instead of returning one of the candidates with equal probability,
    #  return the one with the most second choice votes!!!!!! OR most votes recursively
    def get_plurality(self) -> (str, array):
        """ Returns the plurality winner of a given profile. If there are ties, it randomly returns one of the candidates
            with equal probability

            Returns:
                - The name of the winner
                - A numpy array of size n_candidates, where candidate i is 1 if they are a majority winner,
                and the rest are 0 """

        # Get the plurality winner from the first row of the rank matrix, break ties randomly
        winner_idx = random.choice(flatnonzero(self.rank_matrix[0] == self.rank_matrix[0].max()))
        winner = self.candidates[winner_idx]

        # Get the one-hot vector
        one_hot = zeros(shape=(self.n_candidates,))
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

    def to_count_matrix(self) -> array:
        """ Create a matrix representation of the profile,
        where matrix[c][r] represent the frequency of candidate c in rank r """

        self._idx_ballots = [[self.candidate_map[c] for c in ballot] for ballot in self.name_ballots]

        matrix_rank = [[0] * len(self._idx_ballots[0]) for _ in self._idx_ballots[0]]

        for ballot in self._idx_ballots:
            for candidate, rank in enumerate(ballot):
                matrix_rank[candidate][rank] += 1

        return array(matrix_rank).T

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
        columns = array(rows).T
        df = DataFrame(columns, columns=header)

        return df

    def update_ballots(self, c_down, c_up, c_down_rank, c_up_rank):
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

                # Create a new ballotorder obj and replace!
                new_ballots[i] = BallotOrder(new_ballot_l)
                break

        return new_ballots

    def create_IM(self, count=20) -> (dict, dict):
        IM_rank_matrices = dict()
        IM_ballots = dict()

        for possible_winner in range(self.n_candidates):
            # print(f"Generating alternative IM profiles for candidate {self.candidates[possible_winner]}...")

            self._IM_pairs_set = set()
            IM_rank_matrices[possible_winner], IM_ballots[possible_winner] = self.generate_IM_profiles(possible_winner,
                                                                                                       count)

        return IM_rank_matrices, IM_ballots

    def generate_IM_profiles(self, winner_idx, count) -> (list, list):
        IM_rank_matrices = []
        IM_ballots = []

        wanted_candidates = [i for i in range(self.n_candidates) if i != winner_idx]
        self._rank_matrix = self.to_count_matrix()

        for alt_candidate in wanted_candidates:
            for i in range(count):
                # TODO If we have exhausted possibilities, we should change our option - for now we will return
                # if len(self._IM_pairs_set) >= (self.n_candidates-1)*(self.n_candidates) - 2:
                #     return IM_profiles

                IM_profile, c_up, c_up_rank, c_down, c_down_rank = self.generate_IM_rank_matrix(alt_candidate,
                                                                                                winner_idx)
                if not array_equal(IM_profile, self._rank_matrix):
                    IM_rank_matrices.append(IM_profile)
                    IM_ballots.append(self.update_ballots(c_down, c_up, c_down_rank, c_up_rank))

        return IM_rank_matrices, IM_ballots

    def generate_IM_rank_matrix(self, wanted_idx, winner_idx):

        # Deep copy the rank matrix to avoid modifying the original one
        new_rank_matrix = self._rank_matrix.copy()
        option, i = 1, 0

        while True:
            random.seed(random.choice(range(9999)))

            # Generate rank/candidate_idx pairs to modify the rank matrix
            (candidate_down_rank, candidate_down), (candidate_up_rank, candidate_up) = \
                self.generate_IM_rank_pairs(wanted_idx, winner_idx, option=option)

            # TODO: We may have the same candidate_down, rank tuples as candidate up and might diminish our possibilities
            # If we have found a good alternative profile, great! Else we just keep looking for one
            if (candidate_down_rank, candidate_down) not in self._IM_pairs_set and (
                    candidate_up_rank, candidate_up) not in self._IM_pairs_set:
                self._IM_pairs_set.add((candidate_down_rank, candidate_down))
                self._IM_pairs_set.add((candidate_up_rank, candidate_up))
                break

            # This is to make sure we don't loop forever if we run out of alternative profiles to generate
            i += 1
            if i % 50 == 0:
                option += 1
            if option > 3:
                self._repeated_rank_matrices += 1
                return new_rank_matrix, -1, -1, -1, -1

        # Perform operations
        new_rank_matrix[candidate_up_rank, candidate_up] -= 1
        new_rank_matrix[candidate_down_rank, candidate_down] -= 1

        new_rank_matrix[candidate_up_rank, candidate_down] += 1
        new_rank_matrix[candidate_down_rank, candidate_up] += 1

        return new_rank_matrix, candidate_up, candidate_up_rank, candidate_down, candidate_down_rank

    def generate_IM_rank_pairs(self, wanted_idx, winner_idx, option=1) -> (tuple, tuple):
        """ Given a winner candidate, and a wanted candidate != winner candidate for a particular voter,
            return an alternative rank_matrix reflecting a non-truthful ballot from that voter

            - wanted_candidate = int, the index of a non-winner preferred candidate
            - winner_candidate = int, the index of the winner candidate for a particular rank_matrix
            @param: option = int representing the type of modification to the profile we will make

            @returns: alt_rank_matrix = 2D np.array representing an alternative rank_matrix """

        # TODO: FOR NOW: 3 IM matrices per possible winner!!!!!!!!!!!!!

        # TODO: DEAL WITH ZEROS IN THE MATRIX!!!!!!!!!!!!!!!!!!!!!!!!!!!!! So we don't end up with negative ranks
        # TODO Maybe after a certain number of iterations, if we see that the length of set pair does not change, we break!

        # Choose random non-winner/non-wanted candidates to alter their rank - Make sure we don't choose the same candidate
        candidate_down = random.choice(self.all_candidates_idx)
        candidate_up = random.choice(
            self.all_candidates_idx[:candidate_down] + self.all_candidates_idx[candidate_down + 1:])

        # Get the random ranks to modify scores from - Also make sure we don't choose the same number
        candidate_down_rank = random.choice(self.all_ranks)
        candidate_up_rank = random.choice(
            self.all_ranks[:candidate_down_rank] + self.all_ranks[candidate_down_rank + 1:])

        # Exclude candidate W from the list of candidates going up
        if option == 1:
            # C > O_1 > O_2 > W ===> C > O_2 > O_1 > W
            candidate_up = random.choice(
                self.all_candidates_idx[:winner_idx] + self.all_candidates_idx[winner_idx + 1:])
            candidate_down = random.choice(
                self.all_candidates_idx[:candidate_up] + self.all_candidates_idx[candidate_up + 1:])

        # Select candidate W to go down
        elif option == 2:
            # C > W > O ===> C > O > W
            candidate_down = winner_idx
            candidate_up = random.choice(
                self.all_candidates_idx[:winner_idx] + self.all_candidates_idx[winner_idx + 1:])

        # Exclude candidate C from the list of candidates going down
        elif option == 3:
            candidate_down = random.choice(
                self.all_candidates_idx[:wanted_idx] + self.all_candidates_idx[wanted_idx + 1:])
            candidate_up = random.choice(
                self.all_candidates_idx[:candidate_down] + self.all_candidates_idx[candidate_down + 1:])

        return (candidate_down_rank, candidate_down), (candidate_up_rank, candidate_up)

    def generate_IIA_profiles(self, count) -> array:
        return array


def generate_profile_dataset(num_profiles, n_voters, candidates, origin="distribution", params="spheroid"):
    dataset = []

    for i in tqdm(range(num_profiles)):
        dataset.append(AVProfile(n_voters, origin=origin, params=params, candidates=candidates))
    return dataset


def store_dataset(dataset, n_candidates, n_voters, n_profiles, distr_name):
    with open(f"{distr_name}_nC{n_candidates}_nV{n_voters}_nP{n_profiles}.profiles", "wb") as fp:
        dump(dataset, fp)
        print(f"Stored: {distr_name}_nC{n_candidates}_nV{n_voters}_nP{n_profiles}.profiles")


def load_dataset(file_name):
    with open(file_name, "rb") as rf:
        pickled_dataset = load(rf)

    print(f"Loaded: {file_name}")

    return pickled_dataset

def main():

    candidates = ["Austin", "Brock", "Chad", "Derek", "Ethan", "Gabe", "Jack", "Liam", "Mike", "Tyler"]
    distributions = ["spheroid", "cubic", "euclidean", "gaussian", "ladder", "VMFHypercircle", "VMFHypersphere"]

    n_profiles = args.n_profiles
    n_voters = args.n_voters
    n_candidates = args.n_candidates
    candidates = candidates[:n_candidates]
    n_dists = args.n_distributions

    if n_dists == 1:
        dists = args.distribution
        if dists not in distributions:
            raise Exception("Please enter spheroid, cubic, euclidean, gaussian, ladder, VMFHypercircle or VMFHypersphere for distribution name")

        print("==========================================")
        print(f"\nGenerating {dists} dataset...\n")
        print("==========================================")
        dataset = generate_profile_dataset(n_profiles, n_voters, candidates, "distribution", dists)
        store_dataset(dataset, n_candidates, n_voters, n_profiles, dists)
        print()

    else:
        dists = distributions[:n_dists]

        for distribution in dists:
            print("==========================================")
            print(f"\nGenerating {distribution} dataset...\n")
            print("==========================================")
            dataset = generate_profile_dataset(n_profiles, n_voters, candidates, "distribution", distribution)
            store_dataset(dataset, n_candidates, n_voters, n_profiles, distribution)
            print()

    # profile = AVProfile(5, origin="distribution",
    #               params="spheroid", candidates=["Adam", "Bert", "Chad"])

    # dataset = generate_profile_dataset(50, 10, ["Austin", "Brad", "Chad"], "distribution", "gaussian")
    # store_dataset(dataset, 3, 10, 50, "gaussian")


if __name__ == "__main__":
    main()
