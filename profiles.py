import numpy as np
import pandas as pd
import svvamp as sv
from whalrus.profile.Profile import Profile


def label_profile(profile, candidates):
    """ Convert profile to a list of {candidate: rank} per voter
        Then return a list of labeled candidates in order of rank """

    ordered_prof = []

    for ballot in profile:
        ordered_ballot = {}
        for candidate, rank in zip(candidates, ballot):
            ordered_ballot[candidate] = rank

        ordered_prof.append(ordered_ballot)

    sorted_dicts = [dict(sorted(profile.items(), key=lambda kv: kv[1])) for profile in ordered_prof]
    sorted_dict_keys = [list(d.keys()) for d in sorted_dicts]

    return sorted_dicts, sorted_dict_keys


def profile_as_matrix(profile):
    """ Create a matrix representation of the profile,
    where matrix[c][r] represent the frequency of candidate c in rank r """

    matrix_rank = [[0] * len(profile[0]) for _ in profile[0]]

    for ballot in profile:
        for j, rank in enumerate(ballot):
            matrix_rank[j][rank] += 1

    return np.array(matrix_rank)


def profile_as_dataframe(matrix_rank, candidates):
    """ Creates a dataframe representation of the profile from the matrix representation """
    data_dict = dict()

    # Create dictionary for dataframe
    data_dict["Candidates"] = candidates
    for i in range(len(matrix_rank)):
        data_dict[f"{i}th rank"] = matrix_rank[:, i]

    df = pd.DataFrame(data_dict)

    return df


def create_profile_from_distribution(n_voters, candidates, pop_type="spheroid"):
    """ Return a Profile object given a number of voters, a list of candidates and a population type """

    # TODO: INSERT IF STATEMENTS FOR MULTIPLE POPULATION TYPES if necessary
    # Create a population object
    pop = sv.PopulationSpheroid(V=n_voters, C=len(candidates))
    pop.labels_candidates = candidates

    # Create a dataframe representation of the profile so we can print it out
    matrix_rank = profile_as_matrix(pop.preferences_rk)
    df_rank = profile_as_dataframe(matrix_rank, pop.labels_candidates)

    # Create a labeled version of the ranking from the population
    profile_dicts, labeled_ranks = label_profile(pop.preferences_rk, pop.labels_candidates)

    # Create a profile object - automatically creates ordered ballots
    return Profile(labeled_ranks), df_rank, matrix_rank


def create_profile_from_data(fname):
    pass