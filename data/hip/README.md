# Frozen Q8 expert profile

`qwen36-q3-q8-expert-profile.txt` is the usage-count profile from the Qwen3.6 UD-Q3_K_M Q8 validation campaign. It is copied without content changes from `bench/results/2026-10-15-hip-kv-q8-phase4b/provenance/expert-profile-campaign.txt`.

SHA-256: `1bc70f7edf706578a24aca5a6e8beee4f308e192142ae88339968f5da1924a9d`.

The launcher verifies this hash and copies the file into each Q8 run directory before inference, because the engine updates expert-use counts. Do not use this frozen file directly as the writable profile of an unrelated run. The matching frozen executable is a separate local artifact; copying this profile does not approve another executable or model. See [the HIP guide](../../docs/HIP_GUIDE.md#q8-kv-and-streaming).
