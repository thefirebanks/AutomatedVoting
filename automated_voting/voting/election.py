from whalrus.rule.RuleScorePositional import RuleScorePositional
# from autoprofiles import AVProfile
from automated_voting.voting.constraints import check_condorcet, check_majority

# Basic Rules
from whalrus.rule.RuleCopeland import RuleCopeland
from whalrus.rule.RuleCondorcet import RuleCondorcet
from whalrus.rule.RuleApproval import RuleApproval
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
import sys

def run_baselines(profiles):
    all_rules = [RuleBorda, RuleMaximin, RuleCopeland, RuleCondorcet,  RulePlurality]
    # RuleApproval, RuleVeto, RuleIRV, RulePlurality, RuleVeto,
    # RuleNanson, RuleCoombs, RuleBucklinInstant, RuleSchulze]

    all_names = ["RuleBorda", "RuleApproval", "RuleMaximin", "RuleCopeland", "RuleCondorcet"]
    # "RuleApproval", "RuleVeto", "RuleIRV", "RulePlurality", "RuleVeto", "RuleNanson",
    # "RuleCoombs", "RuleBucklinInstant", "RuleSchulze"]

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

    candidate_map = {"Austin": 0, "Brad": 1, "Chad": 2}

    for profile in profiles:
        for i, rule in enumerate(rules):
            rules_dict[rule_names[i]] = dict()
            r = rule(profile)

            winners = r.cowinners_
            rules_dict[rule_names[i]]["Winner(s)"] = winners if len(r.cowinners_) == 1 else list(winners)
            rules_dict[rule_names[i]]["Condorcet"] = check_condorcet(profile, winners)
            rules_dict[rule_names[i]]["Majority"] = check_majority(profile, winners)
            # rules_dict[rule_names[i]]["Plurality"] = 2

            rules_dict[rule_names[i]]["IM"] = 0
            rules_dict[rule_names[i]]["IM_ties"] = 0

            for alt_profile in profile.IM_ballots[candidate_map[next(iter(r.cowinners_))]]:

                alt_r = rule(alt_profile, *args)
                if len(alt_r.cowinners_) == 1:
                    alt_winners = next(iter(alt_r.cowinners_))
                    if alt_winners == next(iter(winners)):
                        rules_dict[rule_names[i]]["IM"] += 1
                else:
                    rules_dict[rule_names[i]]["IM_ties"] += 1

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
    pass
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
