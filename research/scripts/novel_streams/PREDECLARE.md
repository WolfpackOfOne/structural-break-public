# PRE-DECLARATION — descriptive diagnostics only (no experiment ID, no promotion)

Written before D4 produced any number.

D4 asks the descriptive questions in the driving brief sections 26/41/42:
  * how long do excursions outside the per-series historical null persist,
    in never-break series versus in mature-break series?
  * does an explicit EXCURSION-DURATION / EXCURSION-MASS state variable,
    computed causally from history-only nulls, separate mature-break positives
    from never-break negatives inside the dominant cell (t>=200, age>=100)?
  * is that separation already inside RT-600 (within-t rank correlation and
    conditional lift), or is it new?

What is measured: single-column cell AUC, within-t rank correlation with the
RT-600 cross-fitted blend, and the cell AUC of RT-600 rank + candidate rank
(equal weight, uncalibrated) versus RT-600 alone. Fold 0 only.

What this is NOT: a promotion candidate, a headline OOF number, or a licence to
tune. No grid search. Window/threshold grid fixed here before the run:
  windows w in (32, 64, 128); null thresholds at historical q90 and q99 of the
  SAME rolling statistic at the SAME w; channels = z (location), z^2 (scale),
  AR(2)-residual^2 (innovation scale).
Kill reading: if every duration/mass column is inside +/-0.01 cell AUC of 0.5
after RT-600 conditioning, the "persistence-as-duration" family is dead as a
new channel and the top-10 must not spend a slot on it.
