from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD
# from tensorflow.keras.utils import plot_model
# from tensorflow.keras.models import load_model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow import GradientTape
from tensorflow import convert_to_tensor
import tensorflow.keras.backend as kb

import numpy as np
import pandas as pd
import svvamp as sv
from whalrus.profile.Profile import Profile
from whalrus.rule.RuleCondorcet import RuleCondorcet

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
                                                  vmf_pole=np.random.random_integers(0, len(candidates),
                                                                                     len(candidates)))
            elif params == "VMFHypersphere":
                pop = sv.PopulationVMFHypersphere(V=n_voters, C=len(candidates),
                                                  vmf_concentration=[50, 50],
                                                  vmf_probability=None,
                                                  vmf_pole=[
                                                      np.random.random_integers(0, len(candidates), len(candidates)),
                                                      np.random.random_integers(0, len(candidates), len(candidates))],
                                                  stretching=1)
            else:
                pop = sv.PopulationSpheroid(V=n_voters, C=len(candidates))

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
        return self._rank_matrix.T

    @property
    def flat_rank_matrix(self):
        """
            Input matrix for neural network
            - Flatten dataset for network input layer purposes
            - Reshape to (n_features, 1) for tensor input, then transpose to make it n_features columns,
               where n_features = n_candidates**2 """

        return self.rank_matrix.flatten('F').reshape(self.n_candidates*self.n_candidates, 1).T

    @property
    def tournament_matrix(self):
        return self._tournament_matrix

    @property
    def rank_df(self):
        return self._df_rank.T

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

    # Profile special candidates
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
        first_candidates = self.rank_matrix[0]

        for candidate_idx, candidate_votes in enumerate(first_candidates):
            if candidate_votes >= 0.5 * self.n_voters:
                one_hot = np.zeros(shape=(self.n_candidates,))
                one_hot[candidate_idx] = 1
                return self.candidates[candidate_idx], one_hot

        return "No Majority Winner", np.zeros(shape=(self.n_candidates,))

    # TODO: Instead of returning one of the candidates with equal probability,
    #  return the one with the most second choice votes!!!!!! OR most votes recursively
    def get_plurality(self) -> (str, np.array):
        """ Returns the plurality winner of a given profile. If there are ties, it randomly returns one of the candidates
            with equal probability

            Returns:
                - The name of the winner
                - A numpy array of size n_candidates, where candidate i is 1 if they are a majority winner,
                and the rest are 0 """

        # Get the plurality winner from the first row of the rank matrix, break ties randomly
        winner_idx = np.random.choice(np.flatnonzero(self.rank_matrix[0] == self.rank_matrix[0].max()))
        winner = self.candidates[winner_idx]

        # Get the one-hot vector
        one_hot = np.zeros(shape=(self.n_candidates,))
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

    def to_tournament_matrix(self) -> np.array:
        return np.array([])

    def to_count_matrix(self) -> np.array:
        """ Create a matrix representation of the profile,
        where matrix[c][r] represent the frequency of candidate c in rank r """

        matrix_rank = [[0] * len(self._preferences[0]) for _ in self._preferences[0]]

        for ballot in self._preferences:
            for j, rank in enumerate(ballot):
                matrix_rank[j][rank] += 1

        return np.array(matrix_rank)

    def to_count_dataframe(self) -> pd.DataFrame:
        """ Creates a dataframe representation of the profile from the matrix representation """
        data_dict = dict()

        # Create dictionary for dataframe
        # data_dict["Candidates"] = self._labels_candidates
        for i in range(len(self._rank_matrix)):
            data_dict[f"Rank {i + 1}"] = self._rank_matrix[:, i]

        df = pd.DataFrame(data_dict, index=self._labels_candidates)

        return df

    def to_ballot_dataframe(self) -> pd.DataFrame:
        """ Creates a dataframe where the columns represent the ballots and the top row represents
        the number of candidates that voted for a particular candiate """

        # {Ballot: # of people that have it}
        ballot_counts = dict()
        for ballot in self._preferences:
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
        columns = np.array(rows).T
        df = pd.DataFrame(columns, columns=header)

        return df


def generate_profile_dataset(num_profiles, n_voters, candidates):
    dataset = []
    for i in range(num_profiles):
        dataset.append(AVProfile(n_voters, origin="distribution", params="spheroid", candidates=candidates))
    return dataset


class AVNet(Model):
    def __init__(self, n_features, n_candidates, inp_shape):
        '''
            Simple version: 1 input layer (relu), 1 hidden layer (relu), 1 output layer (softmax)
            Extras:
            - Dropout
            - BatchNormalization

        '''
        super(AVNet, self).__init__()

        # Define layers
        self.input_layer = Dense(n_features, activation='relu', input_shape=inp_shape)
        self.mid_layer = Dense(n_candidates * n_features, activation='relu')
        #         self.last_layer = Dense(n_candidates, activation='relu')
        self.scorer = Dense(n_candidates, activation='softmax')


    def call(self, inputs):
        """ Inputs is some tensor version of the ballots in an AVProfile
            For testing purposes we will use AVProfile.rank_matrix, which represents
            the count of each candidate per rank """

        # Get rank matrices
        x = self.input_layer(inputs)
        x = self.mid_layer(x)
        #         x = self.last_layer(x)

        return self.scorer(x)

# Define optimizer and loss function (MUST BE "GLOBAL!")
optimizer = Adam(learning_rate=0.01)
CCE = CategoricalCrossentropy()

def scores_to_winner(candidate_vector, candidates):
    return candidates[np.argmax(candidate_vector.numpy())]

def av_loss(model, profile, verbose=False):
    """ An av_loss function will take an instance of <profile, condorcet_winner, majority_winner, simulator_list>, where

        - profile = AVProfile.rank_matrix (shape: n_candidates x n_candidates)
        - condorcet_winner = AVProfile.condorcet_winner (one-hot vector of size n_candidates)
        - majority_winner = AVProfile.majority_winner (one-hot vector of size n_candidates)
        - (IIA or IM)_simulator = AVProfile.IIA_profiles, AVProfile.IM_profiles (list of matrices of shape: n_candidates x n_candidates)

        Idea: For 1 profile, the av_loss will be the sum of all the loss components according to the constraints above.
        -> When there is a one-hot vector, we will use cross entropy
        -> When there is a simulator list, we will add up the cross entropy losses after "running the election"

        Extras:
        -> Add a regularization term for the sum of cross entropy in the simulator list case

    """

    # Decompose profile object
    profile_matrix = profile.flat_rank_matrix
    condorcet_w = profile.condorcet_w
    majority_w = profile.condorcet_w
    plurality_w = profile.plurality_w
    condorcet_w_vec = profile.condorcet_w_vector
    majority_w_vec = profile.majority_w_vector
    plurality_w_vec = profile.plurality_w_vector

    # Make a forward pass
    predictions = model(profile_matrix)

    # Interpret the results
    if verbose:
        print("------------------------------------------------------------")
        print("Condorcet winner:", condorcet_w)
        print("Majority winner:", majority_w)
        print("Plurality winner:", plurality_w)
        print("-----------------")
        print("Predicted winner:", scores_to_winner(predictions, profile.candidates))
        print("------------------------------------------------------------")

    def loss(plurality_c, pred_c):
        # If there isn't a Condorcet or a Majority winner, just use the Plurality winner
        if np.count_nonzero(condorcet_w_vec) == 0 and np.count_nonzero(majority_w_vec):
            plurality_score = CategoricalCrossentropy(plurality_c, pred_c)
            return plurality_score
        else:
            condorcet_score = CCE(condorcet_w_vec, pred_c)
            majority_score = CCE(majority_w_vec, pred_c)
            score = condorcet_score + majority_score
            return score

    # Return loss function
    return loss(plurality_c=plurality_w_vec, pred_c=predictions)


def grad(model, profile, verbose=False):
    with GradientTape() as tape:
        loss_value = av_loss(model, profile, verbose)

    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def main():

    #### Step 1: Define basic parameters
    n_profiles = 20
    n_voters = 5
    n_candidates, candidates = 3, ["Adam", "Bert", "Chad"]
    n_features = n_candidates * n_candidates

    print("Number of voters:", n_voters)
    print("Number of candidates:", n_candidates)
    print("Number of features:", n_features)

    #### Step 2: Generate profiles/dataset

    profiles = generate_profile_dataset(n_profiles, n_voters, candidates)

    #### Step 3: Define model and start training loop
    av_model = AVNet(n_features, n_candidates, inp_shape=(1, n_features))

    epochs = 10

    for epoch in range(epochs):

        print(f"Epoch {epoch} ==============================================================")
        # Iterate through the list of profiles, not datasets
        for i in range(len(profiles)):

            # Perform forward pass + calculate gradients
            if i % 10 == 0:
                loss_value, grads = grad(av_model, profiles[i], verbose=True)
            else:
                loss_value, grads = grad(av_model, profiles[i])

            # Print information
            if i % 10 == 0:
                print(f"Step: {optimizer.iterations.numpy()}, Loss: {loss_value}")

            optimizer.apply_gradients(zip(grads, av_model.trainable_variables))

        print()

if __name__ == "__main__":
    main()

