# 🚀 How to Run

Use the following command to run an experiment for Multiclass classification using only FRENCH-ITALIAN Translated DATA

```bash
python frit_main.py \
  --model_name "FacebookAI/roberta-base" \
  --strategy "multiclass_scratch" \
  --directory "../data/" \
  --data_point "llm_both" \
  --fold 1 \
  --sweep_count 1 \
  --up_sample 'yes'


| Argument        | Type   | Description                                                       |
| --------------- | ------ | ----------------------------------------------------------------- |
| `--model_name`  | string | Name of the transformer model (e.g., `"FacebookAI/roberta-base"`) |
| `--strategy`    | string | Strategy to use: `"multiclass_scratch"` or `"pair_wise"`          |
| `--directory`   | string | Path to the data directory                                        |
| `--data_point`  | string | **Must always be** `"llm_both"`                                   |
| `--fold`        | int    | Fold number for cross-validation                                  |
| `--sweep_count` | int    | Number of hyperparameter sweep iterations                         |
| `--up_sample`   | string | Whether to upsample minority classes: `'yes'` or `'no'`           |


## Example
```bash
python frit_main.py \
  --model_name "bert-base-uncased" \
  --strategy "pair_wise" \
  --directory "../data/" \
  --data_point "llm_both" \
  --fold 2 \
  --sweep_count 50 \
  --up_sample 'no'
