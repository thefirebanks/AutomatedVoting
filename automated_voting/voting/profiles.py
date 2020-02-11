import numpy as np
import pandas as pd
import svvamp as sv
from whalrus.profile.Profile import Profile


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

        else:  # origin == "data", params = filename
            preferences = np.genfromtxt(params, delimiter=',', dtype=str)
            self._labels_candidates = preferences[0, :]
            self._preferences = preferences[1:, :].astype(int)

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

    @property
    def rank_matrix(self):
        return self._rank_matrix.T

    @property
    def tournament_matrix(self):
        return self._tournament_matrix

    @property
    def rank_df(self):
        return self._df_rank.T

    @property
    def candidates(self):
        return self._labels_candidates

    @property
    def preferences(self):
        return self._preferences

    @property
    def labeled_ballots(self):
        return self._labeled_ranks

    @property
    def ranked_ballots(self):
        return self._profile_dicts

    @property
    def ballot_df(self):
        return self._ballot_df

    def label_profile(self) -> (list, list):
        """ Convert profile to a list of {candidate: rank} per voter
            Then return a list of labeled candidates in order of rank """

        ordered_prof = []

        for ballot in self.preferences:
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

        matrix_rank = [[0] * len(self.preferences[0]) for _ in self.preferences[0]]

        for ballot in self.preferences:
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
    return np.array(dataset)


def test():
    _ = AVProfile(5, origin="distribution",
                  params="spheroid", candidates=["Adam", "Bert", "Chad"])

    # _ = AVProfile(500, origin="distribution",
    #               params="spheroid", candidates=["Adam", "Bert",
    #                                              "Chad", "Dean", "Elon"])

    # dataset = generate_profile_dataset(10, 500, ["Adam", "Bert", "Chad", "Dean", "Elon"])


if __name__ == "__main__":
    test()
