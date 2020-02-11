from whalrus.rule.RuleScorePositional import RuleScorePositional
from profiles import AVProfile
from constraints import check_condorcet, check_majority
import sys

''' NOTES: 
    - To pick a subset of population candidates use svvamp PopulationSubsetCandidates if needed
    - Find a way of taking in types of population (distribution) and constraints?
    - Should this be a class?'''

'''    #%% Independence of Irrelevant Alternatives (IIA)

    @property
    def log_IIA(self):
        """String. Parameters used to compute :meth:`~svvamp.Election.not_IIA`
        and related methods.
        """
        return "IIA_subset_maximum_size = " + format(
            self.IIA_subset_maximum_size)

    def not_IIA(self):
        """Independence of Irrelevant Alternatives, incomplete mode.

        :return: (``is_not_IIA``, ``log_IIA``).

        Cf. :meth:`~svvamp.Election.not_IIA_full` for more details.
        """
        if self._is_IIA is None:
            self._compute_IIA()
        if np.isnan(self._is_IIA):
            return np.nan, self.log_IIA
        else:
            return (not self._is_IIA), self.log_IIA

    def not_IIA_full(self):
        """Independence of Irrelevant Alternatives, complete mode.

        :return: (``is_not_IIA``, ``log_IIA``, ``example_subset_IIA``,
                 ``example_winner_IIA``).

        ``is_not_IIA``: Boolean. ``True`` if there exists a subset of
        candidates including the sincere winner
        :attr:`~svvamp.ElectionResult.w`, such that if the election is held
        with this subset of candidates, then
        :attr:`~svvamp.ElectionResult.w` is not the winner anymore.
        If the algorithm cannot decide, then the result is ``numpy.nan``.

        ``log_IIA``: String. Parameters used to compute IIA.

        ``example_subset_IIA``: 1d array of booleans. If the election is
        not IIA, ``example_subset_IIA`` provides a subset of candidates
        breaking IIA. ``example_subset_IIA[c]`` is ``True`` iff candidate
        ``c`` belongs to the subset. If the election is IIA (or if the
        algorithm cannot decide), then ``example_subset_IIA = numpy.nan``.

        ``example_winner_IIA``: Integer (candidate). If the election is
        not IIA, ``example_winner_IIA`` is the winner corresponding to the
        counter-example ``example_subset_IIA``. If the election is IIA (or
        if the algorithm cannot decide), then
        ``example_winner_IIA = numpy.nan``.

        .. seealso::

            :meth:`~svvamp.Election.not_IIA`.
        """
        if self._is_IIA is None:
            self._compute_IIA()
        if np.isnan(self._is_IIA):
            return np.nan, self.log_IIA, \
                   self._example_subset_IIA, self._example_winner_IIA
        else:
            return (not self._is_IIA), self.log_IIA, \
                   self._example_subset_IIA, self._example_winner_IIA

    def _IIA_impossible(self, message):
        """Actions when IIA is impossible.
        Displays a message and sets the relevant variables.

        Arguments:
        message -- String. A log message.
        """
        self._mylog(message, 1)
        self._is_IIA = True
        self._example_subset_IIA = np.nan
        self._example_winner_IIA = np.nan

    def _compute_IIA(self):
        """Compute IIA: _is_IIA, _example_subset_IIA and _example_winner_IIA.
        """
        self._mylog("Compute IIA", 1)
        if self.meets_IIA:
            self._IIA_impossible("IIA is guaranteed for this voting system.")
            return
        if self.meets_Condorcet_c_ut_abs and self.w_is_condorcet_winner_ut_abs:
            self._IIA_impossible("IIA guaranteed: w is a Condorcet winner.")
            return
        if self.meets_Condorcet_c_ut_abs_ctb and self.w_is_condorcet_winner_ut_abs_ctb:
            self._IIA_impossible("IIA guaranteed: w is a Condorcet winner "
                                 "with candidate tie-breaking.")
            return
        if self.meets_Condorcet_c_ut_rel and self.w_is_condorcet_winner_ut_rel:
            self._IIA_impossible("IIA guaranteed: w is a relative Condorcet "
                                 "winner.")
            return
        if (self.meets_Condorcet_c_ut_rel_ctb and
                self.w_is_condorcet_winner_ut_rel_ctb):
            self._IIA_impossible("IIA guaranteed: w is a relative Condorcet "
                                 "winner with candidate tie-breaking.")
            return
        if self.meets_Condorcet_c_rk and self.w_is_condorcet_winner_rk:
            self._IIA_impossible("IIA guaranteed: w is a Condorcet winner "
                                 "with voter tie-breaking.")
            return
        if (self.meets_Condorcet_c_rk_ctb and
                self.w_is_condorcet_winner_rk_ctb):
            self._IIA_impossible("IIA guaranteed: w is a Condorcet winner "
                                 "with voter and candidate tie-breaking.")
            return
        if (self.meets_majority_favorite_c_ut and
                self.pop.plurality_scores_ut[self.w] > self.pop.V / 2):
            self._IIA_impossible("IIA guaranteed: w is a majority favorite.")
            return
        if (self.meets_majority_favorite_c_rk and
                self.pop.plurality_scores_rk[self.w] > self.pop.V / 2):
            self._IIA_impossible("IIA guaranteed: w is a majority favorite "
                                 "with voter tie-breaking.")
            return
        if (self.meets_majority_favorite_c_ut_ctb and
                self.w == 0 and
                self.pop.plurality_scores_ut[self.w] >= self.pop.V / 2):
            self._IIA_impossible("IIA guaranteed: w is a majority favorite "
                                 "with candidate tie-breaking (w = 0).")
            return
        if (self.meets_majority_favorite_c_rk_ctb and
                self.w == 0 and
                self.pop.plurality_scores_rk[self.w] >= self.pop.V / 2):
            self._IIA_impossible("IIA guaranteed: w is a majority favorite "
                                 "with voter and candidate tie-breaking "
                                 "(w = 0).")
            return
        if self._with_two_candidates_reduces_to_plurality:
            if self.w_is_not_condorcet_winner_rk_ctb:
                # For subsets of 2 candidates, we use the matrix of victories
                # to gain time.
                self._mylog("IIA failure found by Condorcet failure "
                            "(rk, ctb)", 2)
                self._is_IIA = False
                self._example_winner_IIA = np.nonzero(
                    self.pop.matrix_victories_rk_ctb[:, self.w])[0][0]
                self._example_subset_IIA = np.zeros(self.pop.C, dtype=bool)
                self._example_subset_IIA[self.w] = True
                self._example_subset_IIA[self._example_winner_IIA] = True
            else:
                self._mylog("IIA: subsets of size 2 are ok because w is a "
                            "Condorcet winner (rk, ctb)", 2)
                self._compute_IIA_aux(subset_minimum_size=3)
        else:
            self._compute_IIA_aux(subset_minimum_size=2)

    def _compute_IIA_aux(self, subset_minimum_size):
        """Compute IIA: is_IIA, example_subset_IIA and example_winner_IIA.

        Arguments:
        subset_minimum_size -- Integer.

        Tests all subsets from size 'subset_minimum_size' to
        'self.IIA_subset_maximum_size'. If self.IIA_subset_maximum_size < C-1,
        then the algorithm may not be able to decide whether election is IIA
        or not: in this case, we may have is_IIA = NaN.
        """
        self._mylogv("IIA: Use _compute_IIA_aux with subset_minimum_size =",
                     subset_minimum_size, 1)
        subset_maximum_size = int(min(
            self.pop.C - 1, self.IIA_subset_maximum_size))
        for C_r in range(subset_minimum_size, subset_maximum_size + 1):
            if self.w <= C_r - 1:
                candidates_r = np.array(range(C_r))
            else:
                candidates_r = np.concatenate((range(C_r - 1), [self.w]))
            while candidates_r is not None:
                w_r = self._compute_winner_of_subset(candidates_r)
                if w_r != self.w:
                    self._mylog("IIA failure found", 2)
                    self._is_IIA = False
                    self._example_winner_IIA = w_r
                    self._example_subset_IIA = np.zeros(self.pop.C, dtype=bool)
                    for c in candidates_r:
                        self._example_subset_IIA[c] = True
                    return
                candidates_r = compute_next_subset_with_w(
                    candidates_r, self.pop.C, C_r, self.w)
        # We have not found a counter-example...
        self._example_winner_IIA = np.nan
        self._example_subset_IIA = np.nan
        if self.IIA_subset_maximum_size < self.pop.C - 1:
            self._mylog("IIA: I have found no counter-example, but I have " +
                        "not explored all possibilities", 2)
            self._is_IIA = np.nan
        else:
            self._mylog("IIA is guaranteed.", 2)
            self._is_IIA = True

    def _compute_winner_of_subset(self, candidates_r):
        """Compute the winner for a subset of candidates.

        This function is internally used to compute Independence of Irrelevant
        Alternatives (IIA).

        Arguments:
        candidates_r -- 1d array of integers. candidates_r(k) is the k-th
            candidate of the subset. This vector must be sorted in ascending
            order.

        Returns:
        w_r -- Integer. Candidate who wins the sub-election defined by
            candidates_r.
        """
        self._mylogv("IIA: Compute winner of subset ", candidates_r, 3)
        pop_test = PopulationSubsetCandidates(self.pop, candidates_r)
        result_test = self._create_result(pop_test)
        w_r = candidates_r[result_test.w]
        return w_r 
        '''


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
    try:
        n_voters = int(sys.argv[1])
    except Exception as e:
        raise Exception("Insert a number of voters!")

    # 1. Create a profile
    profile = AVProfile(n_voters, origin="distribution", params="spheroid", candidates=["Adam", "Bert", "Chad"])
    print(profile.rank_df)
    print(profile.rank_matrix)

    # 2. Set weights in order of position
    weights = [2, 1, 0]

    # 3. Run election
    results = election(profile, weights)

    print("Is Condorcet compliant?:", check_condorcet(profile, results))
    print("Satisfies majority criterion?", check_majority(profile.rank_df.T, results))

    for result, value in results.items():
        print(f"{result}: {value}")


if __name__ == "__main__":
    main()
