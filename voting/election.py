from whalrus.rule.RuleScorePositional import RuleScorePositional
from profiles import create_profile
from constraints import check_condorcet, check_majority

''' NOTES: 
    - To pick a subset of population candidates use svvamp PopulationSubsetCandidates if needed
    - Find a way of taking in types of population (distribution) and constraints?
    - Should this be a class?'''


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

    # 1. Create a profile
    profile, profile_df, profile_matrix = create_profile(500, origin="distribution",
                                                         params="spheroid", candidates=["Adam", "Bert", "Chad"])
    print(profile_df)

    # 2. Set weights in order of position
    weights = [2, 1, 0]

    # 3. Run election
    results = election(profile, weights)

    print("Is Condorcet compliant?:", check_condorcet(profile, results))
    print("Satisfies majority criterion?", check_majority(profile_df, results))

    for result, value in results.items():
        print(f"{result}: {value}")


if __name__ == "__main__":
    main()
