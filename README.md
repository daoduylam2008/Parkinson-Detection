# Parkinson Detection

Machine-learning pipeline for detecting Parkinson's disease from tabular clinical/biomedical features. The repo contains data-preprocessing utilities, the source datasets, and a notebook for experimentation.

> **Disclaimer:** This is a research/educational project. It is not a medical device and must not be used for diagnosis.

## Repository structure

```
Parkinson-Detection/
├── main.py           # Entry point (placeholder, currently empty)
├── process.py        # Data loading, feature/target extraction, train/test split
├── test.ipynb        # Experiments and model evaluation
├── parkinsons.xlsx   # Main dataset
├── test.xlsx         # Test dataset
└── .gitattributes
```

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/daoduylam2008/Parkinson-Detection.git
cd Parkinson-Detection
pip install pandas numpy scikit-learn openpyxl jupyter
```

`openpyxl` is needed for `pd.read_excel` to read the `.xlsx` files.

## Data format

Each dataset is a spreadsheet where:

- every column except the last is a feature
- the **last column is the target label**

`process.py` relies on this layout. Only numeric (`int`/`float`) feature values are kept.

## Usage

### Preprocessing (`process.py`)

| Function | Purpose |
|---|---|
| `preprocess_data(raw)` | Splits a DataFrame into feature names, target name, feature DataFrame, and target array |
| `create_features_targets_dict(raw_features, targets, features_names, target_names)` | Builds a scikit-learn-style dict with `data`, `target`, `feature_names`, `target_names` |
| `process_data(data)` | 80/20 train/test split (`random_state=42`) |

Example:

```python
import pandas as pd
from process import preprocess_data, create_features_targets_dict, process_data

raw = pd.read_excel("parkinsons.xlsx")

feature_names, target_name, raw_features, targets = preprocess_data(raw)
data = create_features_targets_dict(raw_features, targets, feature_names, target_name)
X_train, X_test, y_train, y_test = process_data(data)
```

### Experiments

Open the notebook to train and evaluate models:

```bash
jupyter notebook test.ipynb
```

## Roadmap

- [ ] Implement the training/inference entry point in `main.py`
- [ ] Add `requirements.txt`
- [ ] Document model choice and evaluation results
- [ ] Remove `__pycache__/` and `.DS_Store` from version control (add a `.gitignore`)

## License

No license specified yet. Add one (e.g. MIT) before sharing or accepting contributions.
