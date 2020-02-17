from numpy import random, array, zeros, flatnonzero, genfromtxt, array_equal
from pandas import DataFrame
import svvamp as sv
from whalrus.profile.Profile import Profile
from whalrus.rule.RuleCondorcet import RuleCondorcet
from tqdm import tqdm

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
                pop = sv.PopulationCubicUniform(V=n_voters, C=len(candidates))
            elif params == "euclidean":
                pop = sv.PopulationEuclideanBox(V=n_voters, C=len(candidates), box_dimensions=[1])
            elif params == "gaussian":
                pop = sv.PopulationGaussianWell(V=n_voters, C=len(candidates), sigma=[1], shift=[10])
            elif params == "ladder":
                pop = sv.PopulationLadder(V=n_voters, C=len(candidates), n_rungs=5)
            elif params == "VMFHypercircle":
                pop = sv.PopulationVMFHypercircle(V=n_voters, C=len(candidates), vmf_concentration=[10],
                                                  vmf_pole=random.random_integers(0, len(candidates),
                                                                                  len(candidates)))
            elif params == "VMFHypersphere":
                pop = sv.PopulationVMFHypersphere(V=n_voters, C=len(candidates),
                                                  vmf_concentration=[50, 50],
                                                  vmf_probability=None,
                                                  vmf_pole=[
                                                      random.random_integers(0, len(candidates), len(candidates)),
                                                      random.random_integers(0, len(candidates), len(candidates))],
                                                  stretching=1)
            else:
                pop = sv.PopulationSpheroid(V=n_voters, C=len(candidates))

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

        # Create a dataframe representation of the profile so we can print it out
        self._rank_matrix = self.to_count_matrix()
        self._tournament_matrix = self.to_tournament_matrix()
        self._df_rank = self.to_count_dataframe()

        # Create a labeled version of the ranking from the population
        self._profile_dicts, self._labeled_ranks = self.label_profile()

        # Create a dataframe of ballots and how many people voted for them
        self._ballot_df = self.to_ballot_dataframe()

        # Create a profile object - automatically creates ordered ballots
        super().__init__(self._labeled_ranks)

        # Create one-hot vectors of established candidates (Condorcet, Majority, etc.)
        self._condorcet_w, self._condorcet_w_vector = self.get_condorcet()
        self._majority_w, self._majority_w_vector = self.get_majority()
        self._plurality_w, self._plurality_w_vector = self.get_plurality()

        # Extra variable to keep track of IM modified profiles - A set of tuples
        self._IM_pairs_set = set()

        # Create a sample of simulated1 profiles for IM
        self._IM_profiles = self.create_IM()

        # Create a sample of simulated profiles for IIA
        # self._IIA_profiles = self.create_IIA_dict()

    # Basic properties
    @property
    def n_candidates(self):
        return self._n_candidates

    @property
    def n_voters(self):
        return self._n_voters

    @property
    def candidates(self):
        return self._labels_candidates

    @property
    def preferences(self):
        return self._preferences

    # Ballots and Profile matrix/dataframe formats
    @property
    def rank_matrix(self):
        return self._rank_matrix

    @property
    def tournament_matrix(self):
        return self._tournament_matrix

    @property
    def rank_df(self):
        return self._df_rank

    # Ballot descriptors
    @property
    def labeled_ballots(self):
        return self._labeled_ranks

    @property
    def ranked_ballots(self):
        return self._profile_dicts

    @property
    def ballot_df(self):
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
    def IM_profiles(self):
        return self._IM_profiles

    def flatten_rank_matrix(self, rank_matrix=None) -> array:
        """
            Input matrix for neural network
            - Flatten dataset for network input layer purposes
            - Reshape to (n_features, 1) for tensor input, then transpose to make it n_features columns,
               where n_features = n_candidates**2 """

        if rank_matrix is None:
            return self.rank_matrix.flatten('F').reshape(self.n_candidates*self.n_candidates, 1).T
        else:
            return rank_matrix.flatten('F').reshape(self.n_candidates*self.n_candidates, 1).T

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

    def label_profile(self) -> (list, list):
        """ Convert profile to a list of {candidate: rank} per voter
            Then return a list of labeled candidates in order of rank """

        ordered_prof = []

        for ballot in self._preferences:
            ordered_ballot = {}
            for candidate, rank in zip(self._labels_candidates, ballot):
                ordered_ballot[candidate] = rank

            ordered_prof.append(ordered_ballot)

        sorted_dicts = [dict(sorted(profile.items(), key=lambda kv: kv[1])) for profile in ordered_prof]
        sorted_dict_keys = [list(d.keys()) for d in sorted_dicts]

        return sorted_dicts, sorted_dict_keys

    def to_tournament_matrix(self) -> array:
        return array([])

    def to_count_matrix(self) -> array:
        """ Create a matrix representation of the profile,
        where matrix[c][r] represent the frequency of candidate c in rank r """

        matrix_rank = [[0] * len(self._preferences[0]) for _ in self._preferences[0]]

        for ballot in self._preferences:
            for j, rank in enumerate(ballot):
                matrix_rank[j][rank] += 1

        return array(matrix_rank).T

    def to_count_dataframe(self) -> DataFrame:
        """ Creates a dataframe representation of the profile from the matrix representation """
        data_dict = dict()

        # Create dictionary for dataframe
        # data_dict["Candidates"] = self._labels_candidates
        for i in range(len(self._rank_matrix)):
            data_dict[f"Rank {i + 1}"] = self._rank_matrix[:, i]

        df = DataFrame(data_dict, index=self.candidates)

        return df.T

    def to_ballot_dataframe(self) -> DataFrame:
        """ Creates a dataframe where the columns represent the ballots and the top row represents
        the number of candidates that voted for a particular candiate """

        # {Ballot: # of people that have it}
        ballot_counts = dict()
        for ballot in self.preferences:
            ballot_str = str(ballot)
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

            # Turn back into a list so the candidates occupy different cells
            ballot_list = ballot.replace('[', '').replace(']', '').split()

            # Transform numbers into names
            rows.append([self.candidates[int(index)] for index in ballot_list])

        # We will transpose the rows to format ballots properly
        columns = array(rows).T
        df = DataFrame(columns, columns=header)

        return df

    def create_IM(self, count=5) -> dict:
        IM_dict = dict()
        for possible_winner in range(self.n_candidates):
            # print(f"Generating alternative IM profiles for candidate {self.candidates[possible_winner]}...")

            self._IM_pairs_set = set()
            IM_dict[possible_winner] = self.generate_IM_profiles(possible_winner, count)

        return IM_dict

    def generate_IM_profiles(self, winner_idx, count) -> list:
        IM_profiles = []
        wanted_candidates = [i for i in range(self.n_candidates) if i != winner_idx]

        for i in range(count):
            for alt_candidate in wanted_candidates:
                # TODO If we have exhausted possibilities, we should change our option - for now we will return
                if len(self._IM_pairs_set) >= (self.n_candidates-1)*(self.n_candidates) - 2:
                    return IM_profiles

                IM_profile = self.generate_IM_rank_matrix(alt_candidate, winner_idx)
                if not array_equal(IM_profile, self._rank_matrix):
                    IM_profiles.append(IM_profile)

        return IM_profiles

    def generate_IM_rank_matrix(self, wanted_idx, winner_idx, option=1) -> array:
        """ Given a winner candidate, and a wanted candidate != winner candidate for a particular voter,
            return an alternative rank_matrix reflecting a non-truthful ballot from that voter

            - wanted_candidate = int, the index of a non-winner preferred candidate
            - winner_candidate = int, the index of the winner candidate for a particular rank_matrix
            @param: option = int representing the type of modification to the profile we will make

            @returns: alt_rank_matrix = 2D np.array representing an alternative rank_matrix """

        # TODO: FOR NOW: 3 IM matrices per possible winner!!!!!!!!!!!!!

        # TODO: DEAL WITH ZEROS IN THE MATRIX!!!!!!!!!!!!!!!!!!!!!!!!!!!!! So we don't end up with negative ranks
        # TODO Maybe after a certain number of iterations, if we see that the length of set pair does not change, we break!


        # Deep copy the rank matrix to avoid modifying the original one
        new_rank_matrix = self.rank_matrix.copy()

        # Get a list of possible candidates whose score we can increase, that are not the winner or the wanted candidate
        possible_candidates = [i for i in range(self.n_candidates) if i != winner_idx and i != wanted_idx]

        # Get possible ranks to increase/decrease
        possible_ranks = [i for i in range(1, self.n_candidates)]

        # Choose random non-winner/non-wanted candidates to alter their rank
        candidate_down = random.choice(possible_candidates)
        candidate_up = random.choice(possible_candidates)

        # Get the random ranks to modify scores from
        candidate_down_rank = random.choice(possible_ranks)
        candidate_up_rank = random.choice(possible_ranks)

        if option == 1:
            # Possible alteration 1: For n_candidates >= 3.
            # Swap the position of a non-winner and non-wanted candidate with the winner candidate on a given ballot
            # Say we have an honest ballot Bert -> Chad -> Adam
            # We want to transform it into Bert -> Adam -> Chad

            # We are decreasing the winner's score
            candidate_down = winner_idx

        elif option == 2:
            # Possible alteration 2:
            # We could include the winner as part of the leftover candidates maybe?
            # We can also include another candidate that is not the winner to decrease its score

            # Add the winner to be included in the list of candidates whose score we will increase
            possible_candidates.append(winner_idx)

            # Choose a candidate whose score we will increase
            candidate_up = random.choice(possible_candidates)

        #     elif option == 3:
        # Possible alteration 3:
        # Any option from before BUT now, rank 1 is also in play

        i = 0
        # Make sure these choices aren't already chosen - if they are in our set of existing (rank, candidate) pairs, choose again
        while (candidate_down_rank, candidate_down) in self._IM_pairs_set:
            # print(f"Found existing down pair {(candidate_down_rank, candidate_down)} in the set. Choosing again")
            random.seed(random.choice(range(1000000)))
            candidate_down_rank = random.choice(possible_ranks)
            candidate_down = random.choice(possible_candidates)
            i += 1
            if i == 100:
                print("Couldn't find another rank_down/candidate_down combo!")
                return new_rank_matrix
            # print(f"New pair of choice {(candidate_down_rank, candidate_down)}")
            # print(f"Length of pair set: {len(self._IM_pairs_set)}")
            # print()

        self._IM_pairs_set.add((candidate_down_rank, candidate_down))
        # print("---------------")
        i = 0
        while (candidate_up_rank, candidate_up) in self._IM_pairs_set:
            # print(f"Found existing up pair {(candidate_up_rank, candidate_up)} in the set. Choosing again")
            random.seed(random.choice(range(1000000)))
            candidate_up_rank = random.choice(possible_ranks)
            candidate_up = random.choice(possible_candidates)
            # print(f"Found existing up pair {(candidate_up_rank, candidate_up)} in the set. Choosing again")
            # print(f"Length of pair set: {len(self._IM_pairs_set)}")
            # print()
            i += 1
            if i == 100:
                print("Couldn't find another rank_up/candidate_up combo!")
                return new_rank_matrix

        self._IM_pairs_set.add((candidate_up_rank, candidate_up))

        # Perform operations
        new_rank_matrix[candidate_up_rank, candidate_up] -= 1
        new_rank_matrix[candidate_down_rank, candidate_down] -= 1

        new_rank_matrix[candidate_up_rank, candidate_down] += 1
        new_rank_matrix[candidate_down_rank, candidate_up] += 1

        return new_rank_matrix

    def generate_IIA_profiles(self, count) -> array:
        return array


def generate_profile_dataset(num_profiles, n_voters, candidates):
    dataset = []
    print("Generating dataset...")
    for i in tqdm(range(num_profiles)):
        dataset.append(AVProfile(n_voters, origin="distribution", params="spheroid", candidates=candidates))
    return dataset

def test():
    pass
    # _ = AVProfile(5, origin="distribution",
    #               params="spheroid", candidates=["Adam", "Bert", "Chad"])

    # _ = AVProfile(20, origin="distribution",
    #               params="spheroid", candidates=["Adam", "Bert",
    #                                              "Chad", "Dean", "Elon"])

    # dataset = generate_profile_dataset(10, 500, ["Adam", "Bert", "Chad", "Dean", "Elon"])


if __name__ == "__main__":
    test()
