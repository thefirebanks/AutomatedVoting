"""
@author: Daniel Firebanks-Quevedo
"""
import sys

sys.path.append("../..")  # To call from the automated_voting/algorithms/ folder
# sys.path.append("..")     # To call from the automated_voting/ folder
import os

from numpy.random import choice
from numpy import zeros, count_nonzero, argmax, mean
import pandas as pd

from whalrus.rule.RuleScorePositional import RuleScorePositional
from automated_voting.voting.constraints import check_condorcet, check_majority
from automated_voting.voting.profiles import load_dataset, AVProfile

# Basic Rules
from whalrus.rule.RuleCopeland import RuleCopeland
from whalrus.rule.RuleCondorcet import RuleCondorcet
# from whalrus.rule.RuleApproval import RuleApproval
from whalrus.rule.RuleBorda import RuleBorda
from whalrus.rule.RulePlurality import RulePlurality
from whalrus.rule.RuleVeto import RuleVeto

# Rounds
from whalrus.rule.RuleIRV import RuleIRV
from whalrus.rule.RuleNanson import RuleNanson
from whalrus.rule.RuleCoombs import RuleCoombs

# Funky Rules
from whalrus.rule.RuleMaximin import RuleMaximin
from whalrus.rule.RuleBucklinInstant import RuleBucklinInstant
from whalrus.rule.RuleSchulze import RuleSchulze

# Cross entropy to evaluate losses
from tensorflow.keras.losses import CategoricalCrossentropy

CANDIDATES = ["Austin", "Brock", "Chad", "Derek", "Ethan", "Gabe", "Jack", "Liam", "Mike", "Tyler"]
CANDIDATE_MAP = dict(zip(CANDIDATES, range(len(CANDIDATES))))

RULE_NAMES = ["RuleBorda", "RuleMaximin", "RuleCopeland", "RuleCondorcet", "RulePlurality", "RuleSchulze",
                 "RuleBucklinInstant", "RuleVeto"]

RULE_METHODS = [RuleBorda, RuleMaximin, RuleCopeland, RuleCondorcet, RulePlurality, RuleSchulze, RuleBucklinInstant,
             RuleVeto]

CCE = CategoricalCrossentropy()


def get_winner(candidate_vector, candidates, out_format="string"):
    if out_format == "idx":
        return argmax(candidate_vector.numpy())

    elif out_format == "one-hot":
        one_hot = zeros(shape=(len(candidates),))
        one_hot[argmax(candidate_vector.numpy())] = 1
        return one_hot

    elif out_format == "one-hot-baseline":
        one_hot = zeros(shape=(len(candidates),))
        one_hot[CANDIDATE_MAP[candidate_vector]] = 1

        return one_hot
    else:
        return candidates[argmax(candidate_vector.numpy())]


def evaluate_baselines(profiles, rules=None, rule_names=None):
    """ Runs the given profiles through a handful of voting rules and returns a dictionary of dictionaries,
            indicating the name of the voting rule and the scores per each evaluation.

            Parameters:
                - profiles: list of AVProfile objects
                - rules: list of whalrus.Rule rules to apply on those profiles
                - *args: Any necessary extra arguments per rule

            Sample return: {"Borda": {"Condorcet_score": s_1, "Majority_score": s_2, "IM_score":s_3, etc.}}

            Rules used as baseline:
                - Borda Count
                - Condorcet winner
                - Approval
                - Copeland
                - Maximin
                - Plurality
                - IRV? """

    if not rules and not rule_names:
        rules = RULE_METHODS
        rule_names = RULE_NAMES

    rules_dict = dict()

    # Helpers for aggreate versions
    top = dict()
    bottom = dict()

    constraints = dict()

    # Initialize results dict
    for i in range(len(rules)):
        rules_dict[rule_names[i]] = dict()
        rules_dict[rule_names[i]]["Winner"] = []
        rules_dict[rule_names[i]]["Winner_onehot"] = []
        rules_dict[rule_names[i]]["Condorcet"] = 0
        rules_dict[rule_names[i]]["Majority"] = 0
        rules_dict[rule_names[i]]["IM_score_individual"] = []
        rules_dict[rule_names[i]]["IM_score_aggregate"] = ""
        rules_dict[rule_names[i]]["IM_ties"] = 0
        rules_dict[rule_names[i]]["IM_CCE"] = []
        top[rule_names[i]] = 0
        bottom[rule_names[i]] = 0

    for profile in profiles:
        # print("Profile:", profile.rank_matrix)

        # Get the existing winners of a particular profile as one-hot vectors
        constraints["Condorcet"] = profile.condorcet_w_vector
        constraints["Majority"] = profile.majority_w_vector
        constraints["Plurality"] = profile.plurality_w_vector

        for i, rule in enumerate(rules):
            # print("Rule:", rule)
            # rules_dict[rule_names[i]] = dict()
            r = rule(profile)

            winners = r.cowinners_
            pred_winner = next(iter(winners)) if len(r.cowinners_) == 1 else choice(list(winners))

            rules_dict[rule_names[i]]["Winner"].append(pred_winner)

            rules_dict[rule_names[i]]["Condorcet"] += check_condorcet(profile, winners)
            rules_dict[rule_names[i]]["Majority"] += check_majority(profile, winners)

            im_CCE, im, im_ties = AVLoss(pred_winner, constraints,
                                         profile.IM_ballots[CANDIDATE_MAP[pred_winner]], rule, profile.candidates)

            rules_dict[rule_names[i]]["IM_ties"] += im_ties
            rules_dict[rule_names[i]]["IM_CCE"].append(im_CCE)

            top[rule_names[i]] += im
            bottom[rule_names[i]] += len(profile.IM_ballots[CANDIDATE_MAP[next(iter(r.cowinners_))]])

            rules_dict[rule_names[i]]["IM_score_individual"].append(
                f"{im}/{len(profile.IM_ballots[CANDIDATE_MAP[pred_winner]])}")

    # Include aggregate measures + mean
    for i in range(len(rule_names)):
        rules_dict[rule_names[i]]["IM_score_fraction"] = f"{top[rule_names[i]]}/{bottom[rule_names[i]]}"
        rules_dict[rule_names[i]]["IM_score_aggregate"] = round(top[rule_names[i]] / bottom[rule_names[i]], 3)
        rules_dict[rule_names[i]]["IM_CCE_mean"] = round(mean(rules_dict[rule_names[i]]["IM_CCE"]), 3)
        rules_dict[rule_names[i]]["IM_mean"] = round(top[rule_names[i]]/bottom[rule_names[i]], 3)

        print("Average IM Cross entropy:", rules_dict[rule_names[i]]["IM_CCE_mean"])

    im_totals = sum(top.values()) / sum(bottom.values())
    im_score = 1 - im_totals

    print("Survival to IM:", im_totals)
    print("IM rate:", im_score)

    return rules_dict


def AVLoss(pred_winner, constraints, alt_profiles, rule, candidates):
    """
    Given an election outcome, calculate its AVLoss.
    AVLoss is defined as the sum of individual losses per constraint of the designer

    For example, if there are 3 constraints (Condorcet, Majority, Individual Manipulation) we would:
        1. Calculate the cross entropy loss for Condorcet and Majority, add them up
        2. Add up the cross entropy loss for all the simulated profiles that we tested for IM,
            dividing by a regularization term
        3. Return the sum of results from 1 and 2
    """

    total_loss = 0
    pred_winner_vector = get_winner(pred_winner,
                                    candidates, out_format="one-hot-baseline")

    # 1. Get cross entropy score for IM scenarios - for now we will use an average
    IM_score = 0
    im = 0
    im_ties = 0

    for j, alt_profile in enumerate(alt_profiles):
        alt_r = rule(alt_profile)
        if len(alt_r.cowinners_) != 1:
            im_ties += 1

        alt_winner = choice(list(alt_r.cowinners_))
        alt_winner_vector = get_winner(alt_winner, candidates, "one-hot-baseline")

        IM_score += CCE(pred_winner_vector, alt_winner_vector).numpy()

        # print(alt_winner, pred_winner, type(alt_winner), type(pred_winner))
        if alt_winner == pred_winner:
            im += 1

    IM_score /= len(alt_profiles)

    # 2. Calculate cross entropy scores for single winners
    condorcet_w_vec = constraints["Condorcet"]
    majority_w_vec = constraints["Majority"]
    plurality_w_vec = constraints["Plurality"]

    # We gotta check if we use plurality or condorcet and majority
    if count_nonzero(condorcet_w_vec) == 0 and count_nonzero(majority_w_vec):
        plurality_score = CCE(plurality_w_vec, pred_winner_vector).numpy()
        total_loss += plurality_score + IM_score

    else:
        condorcet_score = CCE(condorcet_w_vec, pred_winner_vector).numpy()
        majority_score = CCE(majority_w_vec, pred_winner_vector).numpy()

        if count_nonzero(condorcet_w_vec) != 0 and count_nonzero(majority_w_vec) == 0:
            total_loss += condorcet_score + IM_score

        elif count_nonzero(condorcet_w_vec) == 0 and count_nonzero(majority_w_vec) != 0:
            total_loss += majority_score + IM_score

        else:
            total_loss += majority_score + condorcet_score + IM_score

    return round(total_loss, 3), im, im_ties


def extract_features(fname):
    """
    Get the name of distribution, number of candidates, number of voters, and number of profiles from filename
    """
    # "gaussian_nC3_nV10_nP10.profiles" -> [distr_name, n_candidates, n_voters, n_profiles]
    return fname.replace(".profiles", "").replace("nC", "").replace("nV", "").replace("nP", "").split("_")


def output_results(results):
    """
    Create a table containing the individual scores of each voting rule
    """

    cols = ["Distribution", "Candidates", "Voters", "Profiles"]
    for rule in RULE_NAMES:
        cols.append(f"{rule}_IM_score")

    rows = []
    for distr_features, distr_results in results:
        row = distr_features
        for rule, res in distr_results.items():
            row.append(res["IM_CCE_mean"])

        rows.append(row)

    df = pd.DataFrame(rows, columns=cols)
    df.to_csv("BaselinesResults.csv")

def main():
    path = "../../data/"
    fdir = os.listdir(path)

    fdir = [f for f in fdir if ".DS_Store" not in f]

    # Store all results to output to table later
    all_results = []

    for fname in fdir:
        print("=======================================")
        print("FILE NAME:", fname)
        profiles = load_dataset(path + fname)
        features = extract_features(fname)

        # rrm = sum([p._repeated_rank_matrices for p in profiles])
        # print(f"Repeated rank matrices for {fname}: {rrm}")
        # print("-------")

        baseline_results = evaluate_baselines(profiles, RULE_METHODS, RULE_NAMES)
        all_results.append([features, baseline_results])

        print("Baselines results")
        for rule, res in baseline_results.items():
            print("Rule:", rule)
            print(res)
            print("-------")
        print("=======================================")


    output_results(all_results)

    # try:
    #     n_voters = int(sys.argv[1])
    # except Exception as e:
    #     raise Exception("Insert a number of voters!")
    #
    # # 1. Create a profile
    # profile = AVProfile(n_voters, origin="distribution", params="spheroid", candidates=["Adam", "Bert", "Chad"])
    # print(profile.rank_df)
    # print(profile.rank_matrix)
    #
    # # 2. Set weights in order of position
    # weights = [2, 1, 0]
    #
    # # 3. Run election
    # results = election(profile, weights)
    #
    # print("Is Condorcet compliant?:", check_condorcet(profile, results))
    # print("Satisfies majority criterion?", check_majority(profile.rank_df.T, results))
    #
    # for result, value in results.items():
    #     print(f"{result}: {value}")


if __name__ == "__main__":
    main()

# def election(profile, weights=None):
#     if weights is None:
#         raise Exception("Must insert weights!")
#
#     rule = RuleScorePositional(profile, points_scheme=weights)
#
#     results = dict()
#     results["Gross scores"] = rule.gross_scores_
#     results["Average scores"] = rule.scores_
#     results["Average scores as floats"] = rule.scores_as_floats_
#     results["Winner"] = rule.cowinners_
#
#     return results
