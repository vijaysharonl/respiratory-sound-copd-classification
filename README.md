Respiratory Sound Classification

This project classifies respiratory sounds into **COPD** and **Non-COPD** using Mel spectrograms and a Convolutional Neural Network (CNN) with Efficient Channel Attention (ECA).

Dataset
-  Source: [ICBHI Respiratory Sound Database][https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database]
- **Input:** `.wav` audio files and corresponding `.txt` annotations
- **Labels:** Extracted from `patient_diagnosis.csv`

Pipeline
1. **Segment Audio:** Extract labeled segments from `.wav` files using annotation files.
2. **Convert to Mel Spectrograms:** Generate visual representations of audio segments.
3. **Organize Dataset:** Separate into COPD and Non-COPD based on patient diagnosis.
4. **Train/Test Split:** Create train/test sets for model training.
5. **Model Training:** Train CNN with dual-path convolutions and ECA attention.
6. **Evaluation:** Accuracy, loss, confusion matrix, and classification report.

Requirements
- Python 3.8+
- PyTorch
- Librosa
- scikit-learn
- Matplotlib
- Seaborn

📊 Results
Visualizations include training curves, confusion matrix, and classification report.
Early stopping and dynamic sample weighting included for better generalization.


Training
```bash
python train.py

