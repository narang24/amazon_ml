# Business Entity Resolution — ML Pipeline

[![Pipeline CI/CD](https://github.com/YOUR_ORG/amazon_ml/actions/workflows/pipeline.yml/badge.svg)](https://github.com/YOUR_ORG/amazon_ml/actions/workflows/pipeline.yml)

A shared, fully-automated **Business Entity Resolution** ML pipeline for a team of 3, running on AWS (Free Tier / credits) with GitHub Actions CI/CD.

---

## Architecture Overview

```
GitHub (code) ──push/merge──► GitHub Actions
                                    │
                          ┌─────────▼──────────┐
                          │  Job 1: pytest     │ (runs on all PRs)
                          └─────────┬──────────┘
                                    │ passes
                          ┌─────────▼──────────┐
                          │  Job 2: Deploy     │ (main branch only)
                          │  & Run on EC2      │
                          └─────────┬──────────┘
                                    │
                  ┌─────────────────▼──────────────────────┐
                  │              AWS EC2                    │
                  │  ┌──────────┐   ┌────────────────────┐│
                  │  │  S3      │◄──│  pipeline.py        ││
                  │  │  ~1GB    │   │  ├─ preprocess.py   ││
                  │  │  dataset │   │  ├─ blocking.py     ││
                  │  │  outputs │   │  ├─ features.py     ││
                  │  │  mlruns  │   │  ├─ model.py        ││
                  │  └──────────┘   │  └─ evaluate.py     ││
                  │                 └────────────────────┘│
                  └────────────────────────────────────────┘
                                    │
                          ┌─────────▼──────────┐
                          │  Metrics logged to │
                          │  GitHub Summary    │
                          │  + MLflow + S3     │
                          └────────────────────┘
```

---

## Pipeline Stages

| Step | Module | Description |
|------|--------|-------------|
| 1 | `preprocess.py` | Load TSVs, normalise names/addresses/countries |
| 2 | `blocking.py` | Generate candidate pairs (exact + LSH) |
| 3 | `features.py` | Extract 40+ similarity features per pair |
| 4 | `model.py` | Train LightGBM, sweep threshold for max F0.5 |
| 5 | `pipeline.py` | Predict on **test** set → `matching_results.tsv` **and** `candidate_pairs.tsv` (test-set blocking output, same one-row-per-S1-entity format) |
| 5b | `validate_submission.py` | Validate both output files against the official format rules before they're trusted |
| 6 | `evaluate.py` | Compute precision, recall, F0.5, candidate recall (validation split) |

`output/candidate_pairs.tsv` is always written from the **test-set** candidates that
were actually fed to the model at inference time (not the train-set candidates used
internally to measure `candidate_recall`), so every ID in `matching_results.tsv` is
guaranteed to appear in it, per the official submission rules.

---

## Metrics Tracked

| Metric | Description |
|--------|-------------|
| **F0.5** | Official competition metric (macro-averaged, precision-weighted) |
| Precision | Macro-averaged precision |
| Recall | Macro-averaged recall |
| Candidate Recall | Fraction of GT pairs covered by blocking |
| Block Time (s) | Time spent in blocking step |
| Total Runtime (s) | End-to-end pipeline runtime |

Metrics are:
- Printed in the **GitHub Actions job summary** after every run
- Logged to **MLflow** (stored in S3)
- Saved as `output/metrics.json`

---

## One-Time Setup

### Step 1 — AWS Setup

1. **Create an S3 bucket** (once):
   ```bash
   python scripts/upload_data.py \
     --bucket your-er-bucket \
     --data-dir ./data \
     --create-bucket
   ```

2. **Launch an EC2 instance** (Ubuntu 22.04, t3.medium or better):
   - Attach an **IAM Role** with `AmazonS3FullAccess`
   - Note the **Instance ID** (e.g. `i-0abc123def456789`)
   - Save the SSH private key (`.pem` file)

3. **Set up EC2** (run once via SSH):
   ```bash
   scp -i your-key.pem scripts/setup_ec2.sh ubuntu@<EC2_IP>:~
   ssh -i your-key.pem ubuntu@<EC2_IP>
   bash setup_ec2.sh
   ```

### Step 2 — GitHub Secrets

In your GitHub repo → **Settings → Secrets and variables → Actions**, add:

| Secret | Value |
|--------|-------|
| `AWS_ACCESS_KEY_ID` | IAM user access key (for GitHub Actions runner) |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `EC2_INSTANCE_ID` | e.g. `i-0abc123def456789` |
| `EC2_SSH_KEY` | Contents of your `.pem` private key |
| `EC2_USER` | `ubuntu` (for Ubuntu AMIs) |
| `S3_BUCKET` | Your S3 bucket name |

### Step 3 — IAM Permissions

The IAM user used by GitHub Actions needs **only**:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow", "Action": ["s3:*"], "Resource": "arn:aws:s3:::your-er-bucket/*"},
    {"Effect": "Allow", "Action": ["ec2:StartInstances","ec2:StopInstances",
      "ec2:DescribeInstances"], "Resource": "*"}
  ]
}
```

---

## Development Workflow

### Run locally (no AWS needed)
```bash
# Install dependencies
pip install -r requirements.txt

# Place your data in data/train/ and data/test/
# Run pipeline (no S3 sync)
python src/pipeline.py --config config/pipeline.yaml

# Run evaluation on existing results
# NOTE: --gt-path only has ground truth for the TRAIN S1 ids. If output/matching_results.tsv
# holds TEST-set predictions (the normal end state of a pipeline.py run), this comparison
# is meaningless (disjoint entity ids) -- it's meant for a run whose predictions cover the
# train/validation split. The validation-set F0.5/precision/recall that actually matter are
# already in output/metrics.json (logged during Step 4's threshold sweep).
python src/evaluate.py --output-dir output --gt-path data/train/train_ground_truth.tsv

# Validate the submission files' format before uploading / zipping (pipeline.py already
# runs this automatically as Step 5b and records the result in output/metrics.json)
python src/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir data/test

# Run tests
pytest tests/ -v
```

### Team collaboration workflow
```
1. Create feature branch: git checkout -b feat/better-blocking
2. Make changes, run tests locally: pytest tests/ -v
3. Push & open PR → GitHub Actions runs tests automatically
4. PR review → Merge to main
5. GitHub Actions: tests pass → deploys to EC2 → runs pipeline → logs metrics
6. Review metrics in GitHub Actions summary
7. Repeat
```

---

## Project Structure

```
amazon_ml/
├── .github/
│   └── workflows/
│       └── pipeline.yml       # CI/CD workflow
├── config/
│   └── pipeline.yaml          # All tunable parameters
├── data/
│   ├── train/                 # train_source1/2/3.tsv + ground_truth.tsv
│   └── test/                  # test_source1/2/3.tsv
├── output/                    # Generated: matching_results.tsv, candidate_pairs.tsv (both test-set), metrics.json
├── processed/                 # Cached preprocessed data
├── mlruns/                    # MLflow tracking (synced to S3)
├── scripts/
│   ├── setup_ec2.sh           # One-time EC2 setup
│   └── upload_data.py         # Upload dataset to S3
├── src/
│   ├── preprocess.py          # Step 1: Text normalisation
│   ├── blocking.py            # Step 2: Candidate generation
│   ├── features.py            # Step 3: Pairwise feature extraction
│   ├── model.py               # Step 4: LightGBM training + evaluation
│   ├── pipeline.py            # Orchestrator (steps 1–7, incl. submission validation)
│   ├── validate_submission.py # Vendored official format validator (stdlib only)
│   └── evaluate.py            # Standalone metrics reporter
├── tests/
│   ├── test_preprocess.py
│   ├── test_blocking.py
│   └── test_features.py
└── requirements.txt
```

---

## Stopping / Starting EC2

GitHub Actions **automatically stops EC2** after every pipeline run.  
To start it manually for debugging:
```bash
aws ec2 start-instances --instance-ids i-0abc123def456789
```
Always stop when done:
```bash
aws ec2 stop-instances --instance-ids i-0abc123def456789
```
