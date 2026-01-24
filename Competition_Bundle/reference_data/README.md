# Reference Data

Reference Data consists of **Test Labels** (ground truth). These are separated from Input Data to ensure only the scoring program can access them. The ingestion program should **NOT** have access to these labels.

## Structure

```
reference_data/
├── test_labels.json    # Test labels as JSON dictionary {filename: label}
├── test_labels.txt     # Test labels as CSV (filename, label)
├── test_labels.npy     # Test labels as NumPy array (ordered)
└── train_labels.json   # Training labels (for reference only)
```

## File Formats

### test_labels.json
JSON dictionary mapping filenames to labels:
```json
{
  "grain100_x28y21-var7_8000_us_2x_2020-12-02T135904_corr.npz": 7,
  "grain101_x28y21-var7_8000_us_2x_2020-12-02T135904_corr.npz": 5,
  ...
}
```

### test_labels.txt
CSV format:
```
filename,label
grain100_x28y21-var7_8000_us_2x_2020-12-02T135904_corr.npz,7
grain101_x28y21-var7_8000_us_2x_2020-12-02T135904_corr.npz,5
...
```

### test_labels.npy
NumPy array with labels in the same order as sorted test filenames.

## Usage

The scoring program will:
1. Load test labels from `reference_data/test_labels.json` (or other format)
2. Load predictions from the ingestion program output
3. Compare predictions with ground truth labels
4. Compute accuracy and other metrics

**Security Note**: These files should NOT be accessible to the ingestion program. Only the scoring program should read them to evaluate submissions.