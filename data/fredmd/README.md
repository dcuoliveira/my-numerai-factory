# FRED-MD (vendored)

`fred-md.csv` — McCracken–Ng FRED-MD monthly database, used by
`src/prediction/farm_paper.py` to replicate Lee & Lee (2025) Table 1 as a
validation of the E27 FARM estimators.

- **Span:** 1959:01 – 2019:08 (728 monthly rows + 1 transform-code header row).
- **Columns:** 129 = `sasdate` + 128 series. Row 1 is the McCracken–Ng
  `Transform:` tcode row.
- **Source:** https://raw.githubusercontent.com/bashtage/python-introduction/main/course/introduction/data/fred-md.csv
  (Kevin Sheppard's *Introduction to Python for Econometrics, Statistics and Data
  Analysis* course — a mirror of the ~2019-09 FRED-MD vintage). Fetched because
  the canonical host `files.stlouisfed.org` returns S3 403 to this box's egress IP.
- **SHA256:** `037b7b8d779272613ca8b7c48a00a9702b07f22baeac6f8815ea3b960b7523ba`
- **Bytes:** 608821

The paper uses 1960:01–2019:12; this vintage ends 2019:08, a 4-month end shortfall
that does not affect the pattern+anchor validation bar. The paper keeps 119 of
~127 predictors (dropping series with short history); `farm_paper.py` handles
missing columns the same way.
