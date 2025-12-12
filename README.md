# Battery-SOH-NN - Use a neural network to predict battery soh

This final project was completed as a part of EEP596: Practical Introduction to Deep Learning Applications and Theory
Taught at the University of Washington, Autumn Quarter 2025

## Project Overview
This project attempts to predicts the State of Health (SOH) of Li-ion batteries using short snippets (10s-30s) of discharge data. It compares multiple approaches:
- **Baseline**: Linear Regression & Random Forest
- **Deep Learning**: Multi-Layer Perceptron (MLP) on statistical features (Best Performance, R² ~ 0.75)
- **Time-Series**: GRU on raw sequential data

The goal is to accurately estimate battery health without needing full discharge cycles, enabling faster diagnostics.

The motivation for this project arose from prior work experience within Data Centers and the critical UPS systems that operate within.

34 Batteries were split into 80/20 train/validate, and test
Random sampling of snippets to cover at 5% of discharge duration
Feature engineered to sample Voltage, Current, Temperature, and Time (12 features in all)
Outliers were removed from the data (segments where SOH > 1 or SOH < 0)

Final results show that the MLP was the best predictor of battery SOH, however, it is not accurate to the degree to make business decisions based on the results.
Further work could include any of the following:
1) Predict binary for flag Good/Bad Batteries
2) Deeper networks to capture temporal dependencies
3) Additional battery datasets and/or different battery chemistry

Of note, the model performance did not significantly change when varying snippet size from 5s to 30s. This leads me to believe that the data presented with a 5s snippet is adequate for prediction. I would be interested to further investigate if shorter durations, down to 1s snippets, could yield the same results.

There is significant literature on this exact topic with multiple different NN architectures to capture battery degredation
https://rdcu.be/eUrlm

In hindsight, these previous learnings should have been integrated before starting this project.

## Folder Structure
```
├── README.md           # Project documentation
├── requirements.txt    # Python dependencies
├── src/
│   ├── main.py         # Entry point for training models
│   ├── utils.py        # Data loading, preprocessing, feature extraction
│   └── model.py        # PyTorch model definitions (SOHNet, SOHGRU)
├── data/               # Raw battery data (.mat files)
├── checkpoints/        # Saved models and scalers
├── demo/               # Demo script
├── results/            # Generated plots and visualizations
└── unused/             # Archived scripts from development
```

## Setup Instructions

1. **Prerequisites**
   - Python 3.8+
   - Recommended: Virtual environment (venv or conda)

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Data**
   - The project expects NASA PROGNOSTIA `.mat` files in `data/RawData/`.
   - If starting fresh, ensure your `.mat` files are placed there.

## How to Run

### 1. Training (Optional)
To retrain the best model (MLP) from scratch:
```bash
cd src
python main.py --epochs 200
```
This will save the trained model to `../checkpoints/soh_model_nn.pth`.

### 2. Demo
To run the demo script, which uses the pre-trained model to predict SOH for a random battery sample:
```bash
cd demo
python demo.py
```

### Expected Output
The demo script will output something like:
```
=== Battery SOH Prediction Demo ===
Loading Model...
Loading Sample Data...
Using battery file: B0005.mat
Selected Cycle ID: 100
Actual SOH: 0.8543

Predictions on 5 random snippets: [0.8421, 0.8510, 0.8495, 0.8552, 0.8601]
Average Predicted SOH: 0.8516
Error: 0.0027

Demo Completed Successfully.
```

## Pre-trained Model
The trained model (`soh_model_nn.pth`) and scaler (`scaler.pkl`) are included in the `checkpoints/` directory.

## Model Architectures

### 1. Multi-Layer Perceptron (MLP) - Best Performer
A feed-forward neural network designed to process statistical features extracted from short discharge snippets.
- **Input**: 12 engineered features (Mean/Std/Min/Max of Voltage, Current, Temp; Voltage Slope & Intercept).
- **Structure**:
    - `Linear` (12 $\rightarrow$ 128) $\rightarrow$ `ReLU` $\rightarrow$ `Dropout` (0.2)
    - `Linear` (128 $\rightarrow$ 64) $\rightarrow$ `ReLU` $\rightarrow$ `Dropout` (0.2)
    - `Linear` (64 $\rightarrow$ 32) $\rightarrow$ `ReLU`
    - `Linear` (32 $\rightarrow$ 1) (Output)
- **Why it works**: The statistical summaries (especially voltage slope and mean temperature) capture the degradation trend effectively even without explicit temporal modeling.

### 2. Gated Recurrent Unit (GRU)
A recurrent neural network designed to capture temporal dynamics from raw sequence data.
- **Input**: Raw sequences of `(Voltage, Current, Temperature)` over time steps (10s or 30s).
- **Structure**:
    - `GRU` (Input=3, Hidden=96)
    - `LayerNorm` $\rightarrow$ `Dropout` (0.1)
    - `Linear` (96 $\rightarrow$ 64) $\rightarrow$ `ReLU`
    - `Linear` (64 $\rightarrow$ 1) (Output)
- **Performance**: Improved significantly when sequence length increased from 10s to 30s, but still slightly underperformed the MLP.



## Acknowledgments
- **Dataset**: NASA Prognostics Data Repository (Battery Data Set). https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- **B. Saha and K. Goebel (2007). “Battery Data Set”, NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA**


>>>>>>> ae85c4b (Initial commit)
