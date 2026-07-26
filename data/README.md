# Data guide

The repository ships the processed artifacts required at runtime. A clean clone
can answer structured-measurement questions and load the retrieval corpus
without downloading the large raw measurement archives. The preparation scripts
are included so the artifacts can be reproduced when the original sources are
available.

## Directory map

```text
data/
|-- corpus_raw/             # nine downloaded WHO/EU/EEA PDF or HTML sources
|-- corpus_processed/
|   |-- baseline_chunks.jsonl
|   |-- parents.jsonl
|   |-- children.jsonl
|   |-- child_to_parent.json
|   |-- manifest.csv
|   |-- corpus_stats.json
|   `-- documents/          # normalized Markdown extractions
|-- retrieval_index/
|   |-- child_embeddings.npy
|   `-- index_metadata.json
|-- evaluation/
|   |-- operational_questions.jsonl
|   `-- results/
|       |-- evaluation_summary.json
|       |-- evaluation_details.jsonl
|       |-- operational_summary.json
|       |-- operational_details.jsonl
|       |-- RESULTS.md
|       |-- OPERATIONAL_RESULTS.md
|       |-- SECURITY_RESULTS.md
|       |-- MCP_INSPECTOR_RESULTS.md
|       `-- archive/2026-07-23_full_ragas/
`-- measurements/
    |-- raw_*_2024.zip      # local-only EEA downloads; ignored by Git
    `-- processed/
        |-- eea_sampling_point_annual_2024.parquet
        |-- eea_sampling_point_annual_2024.csv
        |-- eea_country_summary_2024.csv
        `-- eea_excluded_low_coverage_2024.csv
```

The project's 14-question retrieval golden set is currently stored at the
repository root as `golden_dataset.jsonl`. It contains a question, reference
answer, relevant document IDs, relevant parent IDs, category, and difficulty.
The consolidated evaluator writes its audited inputs and scores under
`data/evaluation/results/`. Check the summary's `mode`: `retrieval_only`
records the offline smoke test, while `full_ragas` confirms that all four judge
metrics ran.

The saved summary has `mode: full_ragas`, `questions: 14`, `top_k: 4`,
and HyDE disabled for both controlled branches:

| Metric | Baseline | Final |
| --- | ---: | ---: |
| Context recall | 0.7143 | 1.0000 |
| Context precision | 0.5952 | 0.7480 |
| Faithfulness | 0.9107 | 0.9286 |
| Answer relevancy | 0.7110 | 0.9663 |

The 14 detail rows and their aggregates were cross-checked, then copied
byte-for-byte to `results/archive/2026-07-23_full_ragas/`. See `RESULTS.md` for
the hashes, supplementary retrieval metrics, the 102.4x retrieval-latency
trade-off, and the visible `gold_05`, `gold_10`, and `gold_11` limitations.
These documentary results do not measure agent-selected tools, security
rejections, end-to-end latency, or cost.

`operational_questions.jsonl` is the separate ten-case agent benchmark input.
The agent-version `0.6.0` run is stored in
`operational_details.jsonl` and `operational_summary.json`, with an immutable
copy under `results/archive/2026-07-23_operational_agent_v060/`. Every case used
the real planner and live MCP definitions; expected tools were post-run labels,
not forced routes.

| Operational measure | Accepted result |
| --- | ---: |
| Completed questions | 10/10 |
| Mean end-to-end latency | 19.8555 s |
| TokenBudget-estimated mean USD cost | $0.000000 |
| Mean total tokens | 24,203.4 |
| Strict expected-tool-set match | 8/10 (80%) |
| Present, allowlisted final citations | 10/10 |

The 14 actual MCP calls comprised five documentary searches, two country
summaries, five country comparisons, and two station-extremes calls. Version
`0.6.0` requires requested distribution/coverage content and labels future
thresholds as benchmarks until they apply. The zero-dollar value is specific to
the configured zero-price free-plan estimate. See `OPERATIONAL_RESULTS.md` for
the full token/call accounting, resumable-run caveat, tool mismatches, and
artifact hashes.

`SECURITY_RESULTS.md` records the 16/16 passing security suite, its five-case
before/after table, and the deliberate TokenBudget trigger.
`MCP_INSPECTOR_RESULTS.md` records the successful four-tool Streamable HTTP
Inspector run and its controlled invalid-argument case. Inspector CLI `0.17.2`
completed despite an npm engine warning because Node `20.19.2` was below the
release's declared `>=22.7.5`; the application runtime itself does not require
Node.

## Documentary corpus

The corpus contains nine authoritative sources:

- World Health Organization 2021 global air-quality guideline summary
- Directive 2008/50/EC
- Directive (EU) 2024/2881
- EEA 2026 air-quality status overview, PM2.5 page, NO2 page, and methodology
- EEA exceedance methodology
- ETC HE report on 2024 validated air quality

`corpus_processed/manifest.csv` is the provenance record. It stores the title,
publisher, source URL, evidence status, source filename, SHA-256 digest,
extraction counts, and generated chunk counts for every document. Consult that
file instead of inferring provenance from a local filename.

The `corpus_stats.json` input/output directories and the manifest's
processed-document paths retain information from the Windows machine that built
the corpus. These fields are audit metadata and are not consumed at runtime.

The current processed corpus has:

- 9 documents
- 154 baseline chunks (500 words, 50-word overlap)
- 99 parent chunks (800 words, 100-word overlap)
- 480 child chunks (200 words, 30-word overlap)

Children are searched; the final context contains unique parent passages. The
dense index metadata binds `child_embeddings.npy` to the child-file checksum and
the configured embedding model. A mismatch causes the retriever to rebuild the
index rather than silently use stale vectors.

### Rebuild the corpus and dense index

The expected PDFs and HTML pages are already present in `corpus_raw/`. From the
repository root:

```powershell
.\.venv311\Scripts\python.exe scripts\prepare_corpus.py
.\.venv311\Scripts\python.exe src\retrieval.py --no-hyde --rebuild-index "WHO annual PM2.5 guideline"
```

Linux/macOS:

```bash
./.venv/bin/python scripts/prepare_corpus.py
./.venv/bin/python src/retrieval.py --no-hyde --rebuild-index "WHO annual PM2.5 guideline"
```

The preparation script stops if an expected source is absent or if extraction
produces too little text. It does not silently create a partial corpus.

## Structured measurements

The measurement pipeline uses hourly 2024 EEA sampling-point records for
France, Germany, and Italy and pollutants PM2.5 and NO2. Raw data comes from the
[European Air Quality Portal download service](https://aqportal.discomap.eea.europa.eu/download-data/).

The preparation rules are encoded in `scripts/prepare_measurements.py`:

- accept hourly aggregations only;
- accept EEA validity codes 1, 2, and 3;
- require verification code 1;
- deduplicate the same sampling point, pollutant, and timestamp by the latest
  result time;
- calculate annual unweighted sampling-point means;
- retain series with at least 75% of the 8,784 hours in leap year 2024;
- keep rejected low-coverage series in a separate audit CSV;
- calculate flags and distances for WHO 2021, EU 2030, and current EU annual
  thresholds.

The processed runtime data is intentionally committed because the raw archives
are large. `data/measurements/*.zip` is ignored by Git.

### Rebuild the measurement files

Download the required hourly 2024 archives from the EEA portal, place them in
`data/measurements/`, and name them with the pattern `raw_*_2024.zip`. The local
source set used for the current build is:

```text
raw_de_it_pm25_2024.zip
raw_fr_de_it_no2_2024.zip
raw_sample_fr_pm25_2024.zip
```

Then run:

```powershell
.\.venv311\Scripts\python.exe scripts\prepare_measurements.py
```

Linux/macOS:

```bash
./.venv/bin/python scripts/prepare_measurements.py
```

The script refuses to write final annual outputs if a required
country/pollutant pair is absent, units are unexpected, or duplicate series
remain across source files.

### Measurement limitations

- Results describe monitoring/sampling points, not population exposure.
- Countries have different numbers and types of monitoring stations; rankings
  are not population-weighted.
- The processed source set's latest recorded interval end is
  `2024-12-31 00:00:00`, so the final 24 hours of the leap year are not present.
  The coverage denominator remains 8,784 hours and this is a known source
  limitation, not silently imputed data.
- Location metadata is not included, so sampling-point identifiers cannot be
  presented as city names.

## Runtime path overrides

The default measurement files are under `data/measurements/processed`. Advanced
deployments can point at equivalent files through:

```text
AIR_QUALITY_ANNUAL_DATA
AIR_QUALITY_EXCLUDED_DATA
AIR_QUALITY_COUNTRY_SUMMARY_DATA
```

All replacement files must preserve the schemas validated by
`src/measurements.py`.
