# Shared releases

All five distributions use the same Git-derived version, starting at **7.0.0**:
`pyannote-core`, `pyannote-database`, `pyannote-metrics`, `pyannote-pipeline`, and
`pyannote-audio`. Their Python namespaces and CLI entry points are unchanged.
Internal dependencies accept the compatible 7.x series; users can still install
individual packages without installing audio or PyTorch.

Hatch-VCS searches the parent Git repository. Pre-monorepo tags below 7.0.0 are
excluded from version discovery. Untagged builds are development versions in the
7.0.0 series. Source distributions preserve their version in package metadata and
can be built independently, outside this repository.

## Before merging this migration

- Review the source snapshots in `MIGRATION.md`. Include any later commits from the
  original repositories before the final cutover.
- Merge this PR with a **merge commit** to retain imported ancestry. Squashing or
  rebasing the migration would discard the imported histories from main's ancestry.
- Configure a PyPI trusted publisher for **each** distribution:
  owner `pyannote`, repository `pyannote-audio`, workflow `ci.yml`, environment `pypi`.
  Audio already uses this repository; the other four publishers need to change.
- Coordinate the cutover: retire the old repositories' publishing workflows and
  point their READMEs/contributor instructions to this repository. Archiving those
  repositories is optional and is not performed by this PR.
- Keep the existing audio required checks `ci-complete` and `build`. A completion
  job retains the `build` check after documentation succeeds; `ci-complete` gates all packages.
- Documentation deploys to `https://pyannote.github.io/pyannote-audio/`, with `/core/`,
  `/database/`, `/metrics/`, `/pipeline/`, and `/audio/` subdirectories. Arrange redirects
  or notices at the old documentation sites separately.

## Publish a release

1. Finalize the root changelog through a PR and merge it after all CI checks pass.
2. Create a GitHub release on `main` with a shared tag such as `7.0.0` (or `v7.0.0`).
   Publishing the release triggers CI; pushing a tag or saving a draft does not.
3. CI builds five wheels and five source archives once, verifies their common version
   and internal dependency compatibility, runs the combined test matrix and docs,
   and checks existing PyPI artifacts before publishing.
4. The same artifacts are attached to the GitHub release and published in dependency
   order: core, database, metrics, pipeline, audio.
5. Confirm that all five packages appear on PyPI at the shared version.

PyPI has no transaction spanning multiple projects. If uploading fails partway,
rerun the failed jobs using the existing build artifacts. The guard rejects conflicting
files; `uv publish --check-url` accepts files already uploaded with identical hashes.
Do not move a published tag or overwrite published artifacts. Use a new version if
contents need to change. A partially published release is not complete until all five
packages are available.

Pre-releases such as `7.0.0rc1` use the same process and must also include all packages.
