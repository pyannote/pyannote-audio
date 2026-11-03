# Monorepo source snapshots

This draft starts from the main branches below. It preserves package source, tests,
assets, licenses, and historical changelogs. No unmerged feature branches or local
untracked files are imported.

| Package | Original repository | Imported main commit |
| --- | --- | --- |
| audio | https://github.com/pyannote/pyannote-audio | cc679dc7d661f130369f99e5771dc79883d4fd79 |
| core | https://github.com/pyannote/pyannote-core | f8f0465a6d9f08843b8b498e2296601d1cc2fe28 |
| database | https://github.com/pyannote/pyannote-database | ac9003ea73624e4e66bf76683927be38f1437345 |
| metrics | https://github.com/pyannote/pyannote-metrics | 2b02d04586bad4dedb7ad05ea01a987c7021bf12 |
| pipeline | https://github.com/pyannote/pyannote-pipeline | 867226624e811a430e212072d911d2000939d276 |

Audio's history remains the repository's original history. The other four histories
are connected by subtree merge commits, without squashing. Their old tags and releases
remain in their original repositories; new shared tags belong to pyannote.audio.

Changes in this PR are limited to layout, packaging, workspace development, documentation
links, and CI/release orchestration. Package implementation files are not intentionally
changed. Review future source synchronization separately from migration changes.

The GitHub repository remains pyannote/pyannote-audio, retaining its stars, issues, and
pull requests. This draft does not archive repositories, change PyPI publishers, create
a tag, or publish packages. See RELEASING.md for the cutover checklist.
