# Arabic Dialect Classification Using Deep Learning

This project applies deep learning to classify spoken Arabic dialects from audio data, tackling the linguistic diversity of Arabic. We leverage pre-trained models, spectrograms, and Mel-Frequency Cepstral Coefficients (MFCCs) to capture unique dialectal features. This work aims to improve Arabic-specific language processing systems.

## Dataset

Collected audio data from YouTube podcasts represents five dialects: Moroccan, Egyptian, Gulf, Levantine, and Modern Standard Arabic (MSA). Data was split into training, validation, and testing sets, with augmentation techniques applied to improve model robustness.

## Methodology

1. **Data Collection & Preprocessing**: Audio data was cleaned, segmented, and augmented (e.g., time-stretching, pitch-shifting).
2. **Feature Extraction**: Used spectrograms and MFCCs to capture temporal and frequency-based dialectal cues.
3. **Model Selection**: Tested DenseNet, MobileNet, VGGish, and YAMNet. Transformer models are planned for future improvements.

## Results

Initial models yielded ~80% training accuracy but showed a generalization gap in testing (~50% accuracy). Regularization, data augmentation, and hyperparameter tuning were applied to address overfitting.

## Future Work

Exploring transformer-based models (e.g., Wav2Vec) to improve accuracy by capturing long-range temporal patterns and enhancing noise robustness.

For more details, see the [full paper](link-to-paper.pdf).
