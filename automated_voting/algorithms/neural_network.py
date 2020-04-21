"""
@author: Daniel Firebanks-Quevedo
"""

import sys
sys.path.append("../..")  # To call from the automated_voting/algorithms/ folder
sys.path.append("..")     # To call from the automated_voting/ folder


from automated_voting.voting.profiles import load_dataset, AVProfile, generate_profile_dataset
from automated_voting.voting.election import get_winner

from numpy import count_nonzero, mean
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam, SGD, Adagrad
# from tensorflow.keras.utils import plot_model
# from tensorflow.keras.models import load_model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow import GradientTape
# from tensorflow import convert_to_tensor
# import tensorflow.keras.backend as kb
import time
from tqdm import tqdm

class AVNet(Model):
    def __init__(self, n_features, n_candidates, n_voters, inp_shape, opt, l_rate, arch):
        '''
            Simple version: 1 input layer (relu), 1 hidden layer (relu), 1 output layer (softmax)
            Extras:
            - Dropout
            - BatchNormalization

        '''
        super(AVNet, self).__init__()

        # This architecture variable allows us to run concurrent experiments automatically
        self.arch = arch

        # Define layers
        self.input_layer = Dense(n_features, activation='relu', input_shape=inp_shape)

        if self.arch == 1:
            self.mid_layer = Dense(n_candidates * n_features, activation='relu')
            self.last_layer = Dense(n_candidates * n_voters, activation='relu')

        elif self.arch == 2:
            self.mid_layer = Dense(n_candidates * n_voters, activation='relu')
            self.last_layer = Dense(n_candidates * n_features, activation='relu')

        elif self.arch == 3:
            self.mid_layer = Dense(n_candidates * n_voters, activation='relu')
            self.last_layer = Dense(n_candidates * n_features, activation='linear')

        elif self.arch == 4:
            self.mid_layer = Dense(n_candidates * n_features, activation='relu')
            self.last_layer = Dense(n_candidates * n_voters, activation='linear')

        elif self.arch == 5:
            self.mid_layer = Dense(n_candidates * n_voters, activation='linear')
            self.last_layer = Dense(n_candidates * n_features, activation='linear')

        elif self.arch == 6:
            self.mid_layer = Dense(n_candidates * n_features, activation='linear')
            self.last_layer = Dense(n_candidates * n_voters, activation='linear')

        elif self.arch == 7:
            self.mid_layer = Dense(128, activation='relu')
            self.last_layer = Dense(256, activation='relu')

        elif self.arch == 8:
            self.mid_layer = Dense(128, activation='linear')
            self.last_layer = Dense(256, activation='linear')

        elif self.arch == 9:
            self.mid_layer = Dense(n_candidates * n_features, activation='relu')
            self.last_layer = Dense(n_candidates * n_features, activation='relu')

        elif self.arch == 10:
            self.mid_layer = Dense(n_candidates * n_features, activation='linear')
            self.last_layer = Dense(n_candidates * n_features, activation='linear')

        elif self.arch == 11:
            self.mid_layer = Dense(n_candidates * n_features, activation='relu')
            self.last_layer = Dense(n_candidates * n_features, activation='linear')

        # Voter level, candidate level
        elif self.arch == 12:
            self.mid_layer = Dense(n_candidates * n_features, activation='relu')
            self.extra_layer = Dense(n_candidates * n_voters, activation='relu')
            self.last_layer = Dense(n_candidates * n_features, activation='relu')

        elif self.arch == 13:
            self.mid_layer = Dense(n_candidates * n_features, activation='relu')
            self.extra_layer = Dense(n_candidates * n_voters, activation='linear')
            self.last_layer = Dense(n_candidates * n_features, activation='relu')

        elif self.arch == 14:
            self.mid_layer = Dense(n_candidates * n_features, activation='linear')
            self.extra_layer = Dense(n_candidates * n_voters, activation='linear')
            self.last_layer = Dense(n_candidates * n_features, activation='linear')

        else:
            print("Must enter a valid network architecture! [1-8]")
            sys.exit(1)

        self.leaky_relu = LeakyReLU(alpha=0.3)
        self.dropout = Dropout(0.2)
        self.batch_norm = BatchNormalization()
        self.scorer = Dense(n_candidates, activation='softmax')

        if opt == "Adam":
            self.optimizer = Adam(learning_rate=l_rate)
        elif opt == "Adagrad":
            self.optimizer = Adagrad(learning_rate=l_rate)
        else:
            self.optimizer = SGD(learning_rate=l_rate)

        self.CCE = CategoricalCrossentropy()
        self.av_losses = []

    def reset_scores(self):

        # Scoring functions
        self.total_condorcet = 0
        self.total_majority = 0
        self.total_plurality = 0
        self.total_IM = 0

        self.condorcet_score = 0
        self.majority_score = 0
        self.plurality_score = 0
        self.IM_score = 0

    def call(self, inputs, **kwargs):
        """ Inputs is some tensor version of the ballots in an AVProfile
            For testing purposes we will use AVProfile.rank_matrix, which represents
            the count of each candidate per rank """

        x = self.input_layer(inputs)
        if self.arch in [1, 2, 7, 8, 9]:
            x = self.mid_layer(x)
            x = self.dropout(x)
            x = self.last_layer(x)
            x = self.dropout(x)
        elif self.arch in [3, 4, 11]:
            x = self.mid_layer(x)
            x = self.dropout(x)
            x = self.last_layer(x)
            x = self.leaky_relu(x)
            x = self.dropout(x)
        elif self.arch in [5, 6, 10]:
            x = self.mid_layer(x)
            x = self.leaky_relu(x)
            x = self.dropout(x)
            x = self.last_layer(x)
            x = self.leaky_relu(x)
            x = self.dropout(x)
        elif self.arch == 12:
            x = self.mid_layer(x)
            x = self.dropout(x)
            x = self.extra_layer(x)
            x = self.dropout(x)
            x = self.last_layer(x)
            # x = self.dropout(x)
        elif self.arch == 13:
            x = self.mid_layer(x)
            x = self.dropout(x)
            x = self.extra_layer(x)
            x = self.leaky_relu(x)
            x = self.dropout(x)
            x = self.last_layer(x)
            # x = self.dropout(x)
        elif self.arch == 14:
            x = self.mid_layer(x)
            x = self.leaky_relu(x)
            x = self.dropout(x)
            x = self.extra_layer(x)
            x = self.leaky_relu(x)
            x = self.dropout(x)
            x = self.last_layer(x)
            x = self.leaky_relu(x)
            # x = self.dropout(x)

        return self.scorer(x)

    def av_loss(self, profile, verbose=False):
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

        # Original profile rank matrix
        profile_matrix = profile.flatten_rank_matrix()

        # Winners as strings
        condorcet_w = profile.condorcet_w
        majority_w = profile.majority_w
        plurality_w = profile.plurality_w

        # One-hot vectors
        condorcet_w_vec = profile.condorcet_w_vector
        majority_w_vec = profile.majority_w_vector
        plurality_w_vec = profile.plurality_w_vector

        # Make a forward pass to the profile
        predictions = self.call(profile_matrix)
        pred_w = get_winner(predictions, profile.candidates)

        # Simulated preferences according to the predicted winner
        alternative_profiles = profile.IM_rank_matrices[get_winner(predictions, profile.candidates, out_format="idx")]
        alternative_profiles = [profile.flatten_rank_matrix(alt_profile) for alt_profile in alternative_profiles]

        # Keep track of scores
        if condorcet_w != "No Condorcet Winner":
            self.total_condorcet += 1
            if pred_w == condorcet_w:
                self.condorcet_score += 1
        if majority_w != "No Majority Winner":
            self.total_majority += 1
            if pred_w == majority_w:
                self.majority_score += 1
        else:  # We will use Plurality if no Condorcet or Majority winner
            self.total_plurality += 1
            if pred_w == plurality_w:
                self.plurality_score += 1

        # Interpret the results
        if verbose:
            print("------------------------------------------------------------")
            # print("Profile rank matrix:", profile.rank_matrix)
            print("Condorcet winner:", condorcet_w)
            print("Majority winner:", majority_w)
            print("Plurality winner:", plurality_w)
            print("-----------------")
            print("Predicted winner:", get_winner(predictions, profile.candidates))
            print("Predicted winner quality scores:", predictions)
            print("------------------------------------------------------------")

        def loss(plurality_c, pred_c):
            alt_profiles_score = 0

            # Calculate IM score first - sum of crossentropy for all alternative winners given an IM profile
            IM_score = 0

            for IM_alt_profile in alternative_profiles:
                alt_winner = self.call(IM_alt_profile)
                pred_c_vec = get_winner(pred_c, profile.candidates, out_format="one-hot")

                # Keep track of when winners match between altered profiles and original profile
                if get_winner(alt_winner, profile.candidates) == get_winner(pred_c, profile.candidates):
                    alt_profiles_score += 1

                IM_score += self.CCE(pred_c_vec, alt_winner)

            # Store scores
            self.total_IM += len(alternative_profiles)
            self.IM_score += alt_profiles_score

            if verbose:
                print(f"IM score: {alt_profiles_score}/{len(alternative_profiles)}")

            # Regularize IM score by the number of simulated profiles
            IM_score /= len(alternative_profiles)

            # If there isn't a Condorcet or a Majority winner, just use the Plurality winner
            if count_nonzero(condorcet_w_vec) == 0 and count_nonzero(majority_w_vec) == 0:
                plurality_score = self.CCE(plurality_c, pred_c)
                return plurality_score + IM_score
            else:
                condorcet_score = self.CCE(condorcet_w_vec, pred_c)
                majority_score = self.CCE(majority_w_vec, pred_c)

                if count_nonzero(condorcet_w_vec) != 0 and count_nonzero(majority_w_vec) == 0:
                    return condorcet_score + IM_score

                if count_nonzero(condorcet_w_vec) == 0 and count_nonzero(majority_w_vec) != 0:
                    return majority_score + IM_score

                else:
                    return majority_score + condorcet_score + IM_score

        # Return loss function
        return loss(plurality_c=plurality_w_vec, pred_c=predictions)

    def calculate_grad(self, profile, verbose=False):
        with GradientTape() as tape:
            loss_value = self.av_loss(profile, verbose)

        return loss_value, tape.gradient(loss_value, self.trainable_variables)

    def train(self, profiles, epochs):
        for _ in tqdm(range(epochs)):
            self.reset_scores()

            # print(f"Epoch {epoch} ==============================================================")
            # Iterate through the list of profiles, not datasets
            for i, profile in enumerate(profiles):

                # Perform forward pass + calculate gradients
                # if i % 33 == 0:
                #     loss_value, grads = self.calculate_grad(profile, verbose=True)
                #     print(f"Step: {self.optimizer.iterations.numpy()}, Loss: {loss_value}")
                # else:
                loss_value, grads = self.calculate_grad(profile)

                self.av_losses.append(loss_value)

                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

            # print("\n\n")

    def get_results(self):
        print("===================================================")
        print("AVNet Results")
        print("---------------------------------------------------")

        results = dict()

        if self.total_condorcet != 0:
            print(
                f"Condorcet Score: {self.condorcet_score}/{self.total_condorcet} = {self.condorcet_score / self.total_condorcet}")
            results["Condorcet Score"] = (f"{self.condorcet_score}/{self.total_condorcet}", self.condorcet_score / self.total_condorcet)
        else:
            print("No Condorcet Winners")
            results["Condorcet Score"] = (0, 0)

        if self.total_majority != 0:
            print(
                f"Majority Score: {self.majority_score}/{self.total_majority} = {self.majority_score / self.total_majority}")
            results["Majority Score"] = (f"{self.majority_score}/{self.total_majority}", self.majority_score / self.total_majority)
        else:
            print("No Majority Winners")
            results["Majority Score"] = (0, 0)
        if self.total_plurality != 0:
            print(
                f"Plurality Score: {self.plurality_score}/{self.total_plurality} = {self.plurality_score / self.total_plurality}")
            results["Plurality Score"] = (f"{self.plurality_score}/{self.total_plurality}", self.plurality_score / self.total_plurality)
        else:
            print("No Plurality Winners were used")
            results["Plurality Score"] = (0, 0)

        print(f"IM Score: {self.IM_score}/{self.total_IM} = {self.IM_score / self.total_IM}")
        results["IM Score"] = (f"{self.IM_score}/{self.total_IM}", self.IM_score / self.total_IM)

        print(f"IM CCE Mean:", mean(self.av_losses))
        results["IM CCE Score"] = round(mean(self.av_losses), 3)

        return results

def main():

    # Step 1: Define basic parameters
    n_candidates = 5
    n_features = n_candidates * n_candidates

    # Step 2: Load a dataset
    profiles = load_dataset('../../data/spheroid_nC5_nV500_nP100.profiles')

    # Step 3: Define model and start training loop
    start_n = time.time()
    av_model = AVNet(n_features, n_candidates, inp_shape=(1, n_features))
    av_model.train(profiles, epochs=10)

    _ = av_model.get_results()

    elapsed_n = time.time() - start_n
    print("Train time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_n)))


if __name__ == "__main__":
    main()
