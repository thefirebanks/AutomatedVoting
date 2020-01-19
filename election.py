from whalrus.rule.RuleScorePositional import RuleScorePositional
from profiles import create_profile_from_distribution, create_profile_from_data


def election(profile, weights=None):
    if not weights:
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
    profile, profile_df, profile_matrix = create_profile_from_distribution(500, ["Adam", "Bert", "Chad"])
    # profile, profile_df, profile_matrix = create_profile_from_data(file_name)
    print(profile_df)

    # 2. Set weights in order of position
    weights = [25, 23, 22]

    # 3. Run election
    results = election(profile, weights)

    for result, value in results.items():
        print(f"{result}: {value}")


if __name__ == "__main__":
    main()
