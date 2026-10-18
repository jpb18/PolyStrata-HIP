# Publication privacy cleanup — 18 October 2026

This cleanup supersedes the earlier warning in [RELEASE_AUDIT.md](RELEASE_AUDIT.md) about identifying paths in outgoing history. The user requested that the information be removed before publication and integration into the principal branch, `main`.

## What was cleaned

- Local Windows and Unix profile directory components in documentation, benchmark provenance, archived scripts and a server fixture were replaced with generic `user` paths.
- The repository owner's identifying name/email in their commits were mapped to the public GitHub identity `jpb18 <jpb18@users.noreply.github.com>`. Other contributors' author and committer identities, copyright notices and license text were preserved.
- The repository's local Git identity now uses that public handle and noreply email for future commits. Global Git settings were not changed.
- Ignore rules cover private environment files, common credential containers, root private keys and full repository bundles. Intentionally public `.env.example` files remain eligible for version control.

The original 1,781 commits retain their topology, messages apart from identifying text, author/committer dates and contributions. Their object IDs changed because the earliest affected file versions were part of the inherited history. No commit signatures were present that needed removal. There are no remote tags to migrate.

An original complete Git bundle, the exact redaction policy and the old/new commit map are retained only in ignored `tmp/privacy-history/`. Those files contain private originals and must not be published. This preserves recovery without keeping the old history in the branches intended for publication.

## Verification

The reachable candidate history was scanned as Git objects, rather than only scanning the checked-out files:

| Check | Result |
| --- | --- |
| Original historical file versions requiring redaction | 377 unique blobs |
| Current files changed by identifying-text redaction | 99 |
| Current changed JSON numeric/token structures | 63 compared, unchanged |
| Existing commit count/topology | 1,781 commits preserved |
| Reachable cleaned history | Zero remaining matches for the redaction policy in blobs or commits |
| Recognizable GitHub/AWS/OpenAI credential patterns | No matches found in either scanned history |
| Binary objects requiring redaction | None |
| `src/`, `include/`, `sycl/`, `setup.py`, `serve/server.py` | Unchanged from the prepared release tree |
| Saved Qwen prompt arrays | Eight decoded with the installed tokenizer; no owner identity matches |
| Git object integrity | `git fsck` passed |
| Anonymized server fixture regression | 22 tests passed with normal localhost access |
| Launcher/configuration preservation | Eight regression tests passed |

Only identifying text was replaced; this is not an inference, numerical-policy or benchmark-threshold change. The one changed current server fixture preserves the input format while anonymizing profile paths. Recorded hashes of historical publication files may differ after redaction; the private originals retain their exact bytes.

## Publishing the cleaned branches

The previous remote `main` and `hip-poly` share the old history. Replacing that history requires explicit branch updates protected by `--force-with-lease` against the exact remote tips inspected before publication. An ordinary merge with the old remote branches would make the identifying history reachable again. Do not publish local backup refs or use `git push --all`/`--mirror` for this cleanup.

After replacement, existing clones need to move to the cleaned branches without merging their old history back. A new clone is the simplest route. Rewriting branch history does not erase copies in other clones, GitHub caches or previously fetched commit URLs. The backup bundle and private object/ref archives remain local by design.
