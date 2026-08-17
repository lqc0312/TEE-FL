# TEE-FL
Source code and experimental results for TEE-FL: Federated Learning with SGX Enclave Page Cache Component Isolation.

## Repository Structure
- `code/`: Three core experiment scripts
  - `exp1_sgx_epc.py`: SGX EPC memory telemetry experiment
  - `exp2_component_auc.py`: Component-isolation detection AUC experiment
  - `exp3_gini_lorenz.py`: Gini / Lorenz fairness experiment
- `results/`: Experimental outputs (JSON, CSV, LaTeX tables, PNG figures)
- `report.md`: Validation report comparing old vs new results

## Requirements
- Python 3.8+
- numpy
- matplotlib

## Usage
Run each experiment from the repository root:

    python code/exp1_sgx_epc.py --models "3D-ResNet-18" "3D-ResNet-50" "MLP-MIMIC" "2D-ResNet-18" --sim --out results/sgx
    python code/exp2_component_auc.py --reps 5 --out results/component_auc
    python code/exp3_gini_lorenz.py --n_clients 64 --n_rounds 10 --out results/fairness

## Notes
- The code runs in simulation mode by default unless real SGX hardware is detected.
- A bug fix was applied to LaTeX line-break strings in exp2_component_auc.py and exp3_gini_lorenz.py.
