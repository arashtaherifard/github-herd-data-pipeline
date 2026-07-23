LEGACY INVALID CLASSIFICATION EXPERIMENT

These artifacts belong to the original repository-level high-growth
classification experiment.

The experiment is retained only as an audit trail and must not be reported
as a valid predictive result.

Reason for invalidation:
- 144 of 145 repositories were capped at 1,500 collected stargazer events.
- early_4week_stars + later_stars equaled 1,500 for those repositories.
- The target became_high_growth was therefore perfectly recoverable from
  early-period features.
- The perfect cross-validation scores reflected a collection-cap artifact,
  not genuine future-growth prediction.

The corrected classification design predicts future_growth_surge from
weekly historical features using purged temporal splits.
