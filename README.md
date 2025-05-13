# CIFAR-10-Non-IID
CIFAR-10 Non-IID classification project using PyTorch and CNNs with a time-constrained precision target.
# CIFAR-10 Classification with Non-IID Data using CNN and PyTorch

This project implements a deep learning classification pipeline using a custom Convolutional Neural Network (CNN) on the CIFAR-10 dataset. The dataset is artificially split in a **non-IID** manner, simulating real-world data imbalances across classes. The goal is to reach a validation **precision ≥ 0.7** within **1 minute** on a T4 GPU, demonstrating model efficiency under time constraints.

## 📌 Key Features

- ✅ Custom non-IID data distribution:
  - Classes 0–4: 2,500 images each
  - Classes 5–9: 5,000 images each
- ✅ Data loading with PyTorch's `Dataset` and `DataLoader`
- ✅ CNN with:
  - Batch Normalization
  - Dropout
  - 3 Convolutional blocks
- ✅ Real-time precision tracking and time-limited training (≤ 60s)
- ✅ Performance visualizations (loss, accuracy, precision)
- ✅ Evaluation using metrics: Accuracy, Precision, Recall, F1-Score

## 🧠 Model Architecture

- `Conv2D(3→64)` → ReLU → MaxPool
- `Conv2D(64→128)` → ReLU → MaxPool
- `Conv2D(128→256)` → ReLU → MaxPool
- Flatten → `Linear(4096→512)` → Dropout → `Linear(512→10)`

## 📊 Training Strategy

- Time-limited training enforced via `signal.alarm()` (Unix systems)
- Macro-average precision computed per epoch
- Best-performing model (based on validation precision) saved as `best_simplecnn.pth`

## 📈 Example Output

- 📉 Training Loss and Accuracy per Epoch
- 📊 Validation Precision Plot with target line at 0.7
- 🧪 Full classification report on the validation set

## 🚀 Tech Stack

- Python 3.10+
- PyTorch
- torchvision
- NumPy, Matplotlib, PIL
- scikit-learn (for precision, recall, F1 evaluation)

## 🗂 Files Included

- `Group_24_project_2_2025_non_iid.ipynb`: Main notebook with all implementation
- `best_simplecnn.pth`: Saved model file (if included)
- `README.md`: Project overview and instructions

## 🧪 Results Snapshot

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.8080 |
| Precision  | 0.8000 |
| Recall     | 0.7634  |
| F1 Score   | 0.7749  |


## 📌 Instructions to Run

1. Clone this repository
2. Launch the notebook in Google Colab or your local environment with GPU
3. Ensure GPU is set to **T4** (in Colab: Runtime → Change runtime type → T4 GPU)
4. Run all cells. The model will train within a 60s time limit.
5. Final performance metrics will be printed and plotted

## ⚠️ Note

- Time-limited training uses `signal` which is supported only on Unix-based OS (like Linux or macOS).
- The model is trained under strict time and resource constraints; results may vary slightly run-to-run.

## 👩‍💻 Author

Vedalasya Sonti  
*University of Sydney | ELEC5304 Project (2025)*

