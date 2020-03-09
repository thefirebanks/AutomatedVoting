"""
@author: Daniel Firebanks-Quevedo
"""
import sys
sys.path.append("../..")  # To call from the automated_voting/algorithms/ folder
# sys.path.append("..")     # To call from the automated_voting/ folder
import os


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

CANDIDATES = ["Austin", "Brock", "Chad", "Derek", "Ethan", "Gabe", "Jack", "Liam", "Mike", "Tyler"]
CANDIDATE_MAP = dict(zip(CANDIDATES, range(len(CANDIDATES))))

def run_baselines(profiles):
    all_rules = [RuleBorda, RuleMaximin, RuleCopeland, RuleCondorcet,  RulePlurality, RuleSchulze, RuleBucklinInstant, RuleVeto]
                 #RuleIRV, RuleNanson, RuleCoombs,

    all_names = ["RuleBorda", "RuleMaximin", "RuleCopeland", "RuleCondorcet", "RulePlurality", "RuleSchulze", "RuleBucklinInstant", "RuleVeto"]
                 #"RuleIRV", "RuleNanson", "RuleCoombs", "RuleSchulze", "RuleBucklinInstant", "RuleVeto"]

    baseline_results = evaluate_baselines(profiles, all_rules, all_names)

    return baseline_results

def evaluate_baselines(profiles, rules, rule_names, *args):
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

    rules_dict = dict()

    # Helpers for aggreate versions
    top = []
    bottom = []

    # Initialize results dict
    for i in range(len(rules)):
        rules_dict[rule_names[i]] = dict()
        rules_dict[rule_names[i]]["Winner(s)"] = []
        rules_dict[rule_names[i]]["Condorcet"] = 0
        rules_dict[rule_names[i]]["Majority"] = 0
        rules_dict[rule_names[i]]["IM_score_individual"] = []
        rules_dict[rule_names[i]]["IM_score_aggregate"] = ""
        # rules_dict[rule_names[i]]["IM"] = 0
        # rules_dict[rule_names[i]]["IM_ties"] = 0

    for profile in profiles:
        # print("Profile:", profile.rank_matrix)
        for i, rule in enumerate(rules):
            # print("Rule:", rule)
            # rules_dict[rule_names[i]] = dict()
            r = rule(profile)

            winners = r.cowinners_
            rules_dict[rule_names[i]]["Winner(s)"].append(next(iter(winners)) if len(r.cowinners_) == 1 else list(winners))
            rules_dict[rule_names[i]]["Condorcet"] += check_condorcet(profile, winners)
            rules_dict[rule_names[i]]["Majority"] += check_majority(profile, winners)

            im = 0
            im_ties = 0

            # print("Total IM profiles:", len(profile.IM_ballots[CANDIDATE_MAP[next(iter(r.cowinners_))]]))

            for j, alt_profile in enumerate(profile.IM_ballots[CANDIDATE_MAP[next(iter(r.cowinners_))]]):
                # print(profile.IM_rank_matrices[CANDIDATE_MAP[next(iter(r.cowinners_))]][j])
                alt_r = rule(alt_profile, *args)
                if len(alt_r.cowinners_) == 1:
                    alt_winners = next(iter(alt_r.cowinners_))
                    if alt_winners == next(iter(winners)):
                        im += 1
                else:
                    im_ties += 1

            top.append(im)
            bottom.append(len(profile.IM_ballots[CANDIDATE_MAP[next(iter(r.cowinners_))]]))

            # Is this thing correct???????? Why dhave the same score???oes every rule
            # print("BRO", top, bottom)

            rules_dict[rule_names[i]]["IM_score_individual"].append(
                f"{im}/{len(profile.IM_ballots[CANDIDATE_MAP[next(iter(r.cowinners_))]])}")


    # Include aggregate meaures
    for i in range(len(rules)):
        rules_dict[rule_names[i]]["IM_score_aggregate"] = f"{sum(top)}/{sum(bottom)}, {sum(top)/sum(bottom)}"

    return rules_dict


def election(profile, weights=None):
    if weights is None:
        raise Exception("Must insert weights!")

    rule = RuleScorePositional(profile, points_scheme=weights)

    results = dict()
    results["Gross scores"] = rule.gross_scores_
    results["Average scores"] = rule.scores_
    results["Average scores as floats"] = rule.scores_as_floats_
    results["Winner(s)"] = rule.cowinners_

    return results


def main():

    path = "../../data/"
    fdir = os.listdir(path)

    fdir = [f for f in fdir if ".DS_Store" not in f]

    for fname in fdir:
        print("=======================================")
        print("FILE NAME:", fname)
        profiles = load_dataset(path + fname)



        rrm = sum([p._repeated_rank_matrices for p in profiles])
        print(f"Repeated rank matrices for {fname}: {rrm}")
        print("-------")

        baseline_results = run_baselines(profiles)

        print("Baselines results")
        for rule, res in baseline_results.items():
            print("Rule:", rule)
            print(res)
            print("-------")
        print("=======================================")

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
