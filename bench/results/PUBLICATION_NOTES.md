# Publication redactions

During the HIP release preparation, a local Windows user-directory component was replaced with `user` in 65 already tracked benchmark files. These are publication copies. Original bytes are retained in the author's ignored `tmp/release-preparation/private-benchmark-originals/` archive. [The manifest](PUBLICATION_REDACTIONS.json) records original and publication SHA-256 values and replacement counts.

Only the identifying text component was changed. JSON fields, numeric values, token-ID arrays, timings, thresholds, metrics and file names were preserved. JSON numeric structures were compared before and after. Historical content hashes recorded inside reports may refer to the original files; the manifest distinguishes the publication representations. Do not claim byte identity between these redacted files and the originals.

Recorded local paths are provenance, not paths a new developer should copy. Supply the model, executable and output paths appropriate to your installation. Use saved token IDs for an exact prompt comparison. Fresh prompt generation reads the current documentation and source corpus, so documentation changes can change newly generated prompts; always use the same frozen prompt file across compared arms and record its hash.

New raw campaign directories remain local by default. Before publishing another campaign, review its paths, process arguments, model inputs and outputs as well as ordinary credentials. Preserve the original campaign and review a separate publication copy.
