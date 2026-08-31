# Branch/Tag Pruning Audit -- 2026-08-31

Branch: `release/2026-research-consolidation`
Audited consolidation head before this audit-only documentation commit:
`855317d6d1fa1bcee2a9a83c26dfc88793750977`

This is the final reachability audit required before treating branch pruning as
a post-merge cleanup task. It does not delete branches, move tags, force-push, or
rewrite history.

## Method

Commands were run after `git fetch --all --tags --prune`.

- Remote branches: `git for-each-ref refs/remotes/origin --format='%(refname:short) %(objectname)'`, excluding the `origin`/`origin/HEAD` alias
- Tags: `git for-each-ref refs/tags --format='%(refname:short)'` plus `git rev-list -n 1 <tag>`
- Reachability: `git merge-base --is-ancestor <branch-head> HEAD`
- Open PRs: `gh pr list --state open`
- Local worktrees: `git worktree list --porcelain`

## Summary

Remote branch heads audited: 34, excluding the `origin`/`origin/HEAD` alias.

- 15 remote branch heads are already reachable from the consolidation head.
- 9 remote branch heads are protected by exact existing milestone tags.
- 1 remote branch head is contained in an existing later milestone tag.
- `origin/main` is the canonical base and is not a pruning candidate.
- `origin/release/2026-research-consolidation` is the active PR branch.
- 7 remote branch heads are neither reachable from the consolidation head nor
  protected by an existing tag. They are explicitly **not prunable** without a
  separate tag/import decision.

Open PRs at audit time:

- #14: `release/2026-research-consolidation` -> `main`
- #12: `chore/repo-cleanup-2026` -> `main`

Because no branch deletion is performed in PR #14, the untagged/unreachable
branches below are not blockers for moving `main`. They are blockers only for a
future destructive cleanup pass that proposes deleting those specific branches.

## Reachable From Consolidation Head

These remote branch heads are already ancestors of
`release/2026-research-consolidation` and therefore will be reachable from
`main` after PR #14 is merged.

| Branch | Head |
| --- | --- |
| `origin/engineering/rt1257-deployment-2026` | `5ee281c456216ffed8d1c209aea1329675ea5bed` |
| `origin/engineering/rt600-final-reliability-2026` | `d47937718ba91c2327aafa4528bdba39ee06661f` |
| `origin/research/catboost-specialist-2026` | `0ca415daa38214f834044a71475e1cbd97218f40` |
| `origin/research/causal-representation-frontier-2026` | `81658764c95a7d91f3ca1f80ff25ad8c2b9b33e2` |
| `origin/research/current` | `aca2c4f9b68ad6315956f7c499ebefb2ae311f7b` |
| `origin/research/deep-ensemble-frontier-2026` | `ec0e390027267d6752cccd0fdc7b10e513ec683a` |
| `origin/research/deep-ensemble-frontier-crunch-2026` | `3f94d5521dfc65957a2528ecb6707b64e0982147` |
| `origin/research/deep-ensemble-frontier-local-2026` | `be30e75a52b071e5636f6cffd617d140ffc93924` |
| `origin/research/gpu-tabular-2026` | `0c1a7df5ce0cae23b87aedf5c798cebd69bc9add` |
| `origin/research/leaderboard-alpha-2026` | `d92860d73d3dd6c3bf27654176a1380c32b415cb` |
| `origin/research/learner-diversity-2026` | `7e5ee4c6e4390c21c466e5f9e84fa75bb1ab0e8c` |
| `origin/research/new-avenues-pilots-2026` | `b47b22ad7e0b85fe977cb65453ae8531227d414d` |
| `origin/research/wave6-alpha` | `a0849568f4dfaae6bfbbb032a00d8823a4b4725b` |
| `origin/research/wave7-t2-promotion` | `d5d78ed1203f763b22ee9daf74e999577664d7c2` |
| `origin/research/wave7-teacher-distillation` | `5093e0a8b781e80f5c395d714ea0bfd842dcd0ae` |

## Existing-Tag Protected

These remote branch heads are not ancestors of the consolidation head, but their
head commits are already preserved by existing milestone tags.

| Branch | Head | Existing tag |
| --- | --- | --- |
| `origin/claude/project-setup-github-20g6pt` | `0463c5b03b2fcd376892a5252763ebe5cd0ac61e` | `wave7-early-attempt-superseded` |
| `origin/claude/rt600-baseline-submission` | `9aaa9b046609eb94d9fdd09f7ae07d9249fe2f65` | `rt600-production-0.6268` |
| `origin/codex/oracle-information-frontier-2026` | `5a3b8a0377eb9d42e4188e2af38ffd0fdff6e527` | `oracle-information-frontier-2026-study` |
| `origin/codex/reproduce-2025-public-solution` | `422e4b2db989051c39d517f03a947ae475f36a5c` | `reproduce-2025-public-solution-audit` |
| `origin/codex/wave3-engineering` | `24675a63887b82512e1ffa60404ab72d738f12bc` | `wave3-engineering-rt150` |
| `origin/production/rt600` | `6e591305b3e9877f3c2fa3f6a3f86d2fa4721760` | `rt600-production-reliability-2026` |
| `origin/research/multi-agent-2026` | `23cb4da306eaa63c2109a4ca92a37319b4785e6d` | `multi-agent-2026-final` |
| `origin/research/wave5-alpha` | `74f75d91641d6234822f2a09c4b0ea19814898b2` | `wave5-alpha-c1c2c3-superseded` |
| `origin/research/wave8-future-aware-distillation` | `589e1db93b778dd9573b197233bec104c4a1644e` | `wave8-future-aware-final` |

`origin/release/rt600-reliability-2026` at
`0a42f7758a40a0fd291190c04864f79fbdbf9935` is contained in the later existing
tag `rt600-production-reliability-2026`, whose target commit descends from that
release branch head.

## Not Prunable Without Separate Preservation

These remote branch heads are not reachable from the consolidation head and are
not protected by an existing tag. They must not be deleted in a pruning pass
unless a separate owner-approved tag/import decision preserves their unique
history first.

| Branch | Head | Note |
| --- | --- | --- |
| `origin/chore/repo-cleanup-2026` | `36a0c6d2b0ff13fc2a0c97b2a27995b11679e81f` | open PR #12 |
| `origin/claude/local-model-results-review-lk0f7i` | `33fa041921f2e29d3b7bfbffcfac565997f9789a` | no existing tag |
| `origin/claude/structural-break-competition-entry-yjoj8r` | `7892dcbbc44305c27c60c9e5c3ada52ca277f729` | no existing tag |
| `origin/grok-response-issues-20260830` | `c9f08c6473aa7a1345c5ffc0ad98b6907c8109e9` | post-snapshot response-audit branch |
| `origin/research/data-forensics-2026` | `6786096898eca2e97646e295b8fa3ab43c10d170` | no existing tag |
| `origin/research/multi-agent-frontier-20260829` | `00f7ebd02a99a7ddbb10f5ed7e4eec61abaf916d` | no existing tag |
| `origin/research/raw-data-atlas-2026` | `40215ebf2b010d1488d24b1447510b9cc5484fae` | no existing tag |

## Local-Only Worktree Branches

The following local heads have no upstream, are checked out in local worktrees,
and are not remote-pruning candidates. They are also not safe to delete locally
without a separate decision.

| Branch | Head |
| --- | --- |
| `deepseek-response-issues-20260830` | `cd4bdb9e0f19c4d668fc3a9a03fe46a531f8b36a` |
| `kimi-response-issues-20260830` | `de3678a3b84ef1426ac603f1f0c450262d287d93` |
| `research/investigation-20260829-unresolved-questions` | `75fd8411ed94c9bf8b8c2f49abca357d0d4396d1` |

## Pruning Decision

No branch deletion is required before PR #14 can move `main`. The audit
establishes which branches are safe-by-reachability, safe-by-existing-tag, and
not safe without additional preservation. A future cleanup may delete only the
safe sets, and only after confirming:

1. PR #14 has merged and the reachable set is actually reachable from `main`.
2. The branch is not the head of an open PR.
3. The branch is not checked out by an active local worktree.
4. Any branch in the "not prunable" table has received its own tag/import
   decision first.
