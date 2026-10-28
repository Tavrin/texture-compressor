# Changelog

## [Unreleased]

- Initial block compression implementation. BC7 remains experimental.
- BC7 `Effort::Fast` uses a new search: principal-axis endpoints with
  least-squares refinement, selector choice by projection, and extra modes or
  partitions only for blocks that need them. Its output differs from earlier
  builds; `Balanced` and `Quality` output is unchanged.
