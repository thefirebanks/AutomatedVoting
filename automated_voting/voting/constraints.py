''' Fairness and Non-manipulability constraints '''

from whalrus.rule.RuleCondorcet import RuleCondorcet


def check_individual_manipulation():
    pass


def check_coalition_manipulation():
    pass


def check_trivial_manipulation():
    pass


def check_condorcet(profile, results):
    """ Check if a profile contains a condorcet winner and if the voting rule gave it as a winner
        Returns:
            1  if the Condorcet winner was elected
            -1 if there is a Condorcet winner but was not elected
            0  if there isn't a Condorcet winner """

    condorcet = RuleCondorcet(profile)

    if len(condorcet.cowinners_) == 1 and len(results) == 1:
        if next(iter(results)) == next(iter(condorcet.cowinners_)):
            return 1
        else:
            return -1
    return 0


def check_majority(profile, results):
    """ Check if the voting rule outputs a candidate that is preferred (ranked first) by the majority of the population
        We will look into the first column of the profile matrix where matrix[i][0] represents how many people
        voted for candidate i as first in their ranking, and compare it with the winner

        Returns:
            1  if majority criterion was satisfied
            -1 if it wasn't
            0  if there isn't a clear majority winner """


    profile_df = profile.rank_df

    pop_number = sum(profile_df.iloc[0, 1:].astype(int))

    first_candidates = profile_df.iloc[:, 1]

    for i, candidate in enumerate(first_candidates):
        if candidate >= 0.5 * pop_number:
            if profile_df.iloc[i, 0] == next(iter(results)):
                return 1
            else:
                return -1
    return 0


def check_monotonicity():
    pass


def check_IIA():
    pass

