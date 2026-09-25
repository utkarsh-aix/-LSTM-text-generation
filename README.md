# Shakespeare Word-Level LSTM Text Generation

## Project Overview
This project implements a word-level Long Short-Term Memory (LSTM) language model using TensorFlow/Keras to generate Shakespeare-style text from a given seed sequence.

## Assessment Objective
The objective of this assessment is to train a recurrent neural network that learns next-word prediction from William Shakespeare's works and generates text iteratively from a seed input.

## Dataset
- **Dataset**: Project Gutenberg, *The Complete Works of William Shakespeare*
- **eBook**: #100
- **Dataset Page**: https://www.gutenberg.org/ebooks/100
- **Direct Text Source**: https://www.gutenberg.org/files/100/100-0.txt
- **Automatic Download**: The project automatically downloads the plain-text dataset during preprocessing if it is not already present in the `data/` directory.
- **No Credentials Required**: No Kaggle account or API key is required.
- **Version Control**: The raw dataset (`data/*.txt`) is excluded via `.gitignore` and is not intended to be committed to GitHub.

## Execution Environment
The complete project was executed and verified locally in VS Code:
- **Operating System**: Windows
- **Editor**: VS Code
- **Terminal**: Windows CMD terminal
- **Virtual Environment**: Python virtual environment (`.venv`)
- **Framework**: TensorFlow / Keras
- **Local Execution**: The entire implementation, training, testing, and final results were run directly on a local Windows PC inside VS Code.
- **Google Colab**: Google Colab was **NOT** used for the actual implementation, training, testing, or final results.

## Preprocessing Pipeline
The preprocessing pipeline prepares the raw Shakespeare text for word-level next-token prediction:
```
Raw Shakespeare text
  -> lowercase
  -> remove punctuation
  -> normalize whitespace
  -> word tokenization
  -> vocabulary creation
  -> integer encoding
  -> fixed-length sequence creation
  -> next-word target
```
1. **Cleaning**: Lowercases all text, strips non-alphabetic punctuation characters (preserving words and spacing), and normalizes whitespace.
2. **Tokenization**: Splits cleaned text into individual word tokens.
3. **Vocabulary Creation**: Builds a frequency-ranked word-to-integer dictionary with index 0 reserved for `<UNK>`.
4. **Sequence Creation**: Generates fixed-length sliding windows of tokens (`sequence_length = 20`) with the subsequent token as the target label (`y`).
5. **Train/Validation Split**: Splits generated sequences into 90% training and 10% validation sets.
6. **Vocabulary Persistence**: Saves vocabulary mappings to `models/vocab.json`.

## Training Data & Baseline Configuration
The following baseline configuration and metrics reflect the completed local training run:
- **Sequence Length**: 20
- **Sequences Created**: 150,000
- **Training Sequences**: 135,000
- **Validation Sequences**: 15,000
- **Total Words**: 988,682
- **Vocabulary Size**: 24,461
- **Batch Size**: 256
- **Epochs Requested**: 5
- **Epochs Completed**: 5
- **Training Time**: 988.64 seconds (~16.48 minutes)
- **Best Validation Loss**: 6.53932

## Model Architecture
The network is built using the Keras Functional API:
```
Input Sequence (length = 20)
  -> Embedding Layer (input_dim = 24,461, output_dim = 128)
  -> LSTM Layer (128 units)
  -> Dropout Layer (rate = 0.2)
  -> Dense Layer (units = 24,461, activation = softmax)
```

- **Embedding Layer**: Projects integer-encoded tokens into a dense 128-dimensional embedding space.
- **LSTM Layer**: 128 recurrent units processing the sequence history.
- **Dropout Layer**: Rate 0.2 applied after the recurrent layer for regularization.
- **Dense Output Layer**: Softmax layer over the full 24,461-word vocabulary.
- **Optimizer**: Adam optimizer.
- **Loss Function**: Sparse Categorical Crossentropy.
- **EarlyStopping**: Halts training if validation loss does not improve, restoring best weights.
- **ModelCheckpoint**: Saves the best-performing model weights to `models/best_model.keras`.

## Training Strategy
The model is trained using a supervised next-word prediction objective:
- **Optimization**: Adam optimizer with a default learning rate of 0.001.
- **Loss Calculation**: Sparse Categorical Crossentropy applied directly to vocabulary integer targets without one-hot encoding overhead.
- **Regularization & Validation**: 10% held-out validation split (15,000 sequences) monitored across 5 epochs with batch size 256.
- **Callbacks**:
  - `EarlyStopping`: Monitors validation loss (`val_loss`) with `patience=2` to prevent overfitting and restore the best weights.
  - `ModelCheckpoint`: Automatically saves the best-performing model weights to `models/best_model.keras`.

## Text Generation Strategy
Text generation operates iteratively via word-level next-token prediction:
1. **Seed Input**: Accepts a seed text prompt (e.g., `"to be or not to be"`).
2. **Integer Encoding**: Converts input words to integer IDs using the saved vocabulary (padding shorter sequences with 0 to match the 20-word input length).
3. **Probability Distribution**: Passes the sequence into the trained LSTM model to output next-word probabilities across the entire 24,461-word vocabulary.
4. **Temperature Sampling**: Scales logits by a configurable temperature parameter before softmax sampling:
   - **Higher Temperature (e.g., > 1.0)**: Produces more diverse, creative, and varied token selections.
   - **Lower Temperature (e.g., < 0.7)**: Produces more conservative, confident, and repetitive outputs.
5. **Autoregressive Appending**: Appends the sampled word to the running sequence and slides the 20-token window forward, repeating the process for the requested number of words.

The generation module supports configurable command-line arguments:
- `--seed`: Starting prompt text.
- `--words`: Number of words to generate.
- `--temperature`: Sampling temperature.

## Generated Examples
Multiple seed prompts were tested during generation, including:
- `"to be or not to be"`
- `"the king"`
- `"love is"`

The output text from generation runs is saved to `outputs/generated_text.txt`. The generated output shows Shakespeare-like vocabulary and local word patterns, but longer generated sequences can lose semantic and grammatical consistency.

## Current Model Observations
- **Loss Convergence**: Training loss steadily decreased over the course of the 5 epochs, and validation loss reached a best score of 6.53932.
- **Vocabulary & Style**: The model successfully acquired recognizable Shakespearean vocabulary and characteristic local phrase structures.
- **Coherence**: As expected for a word-level language model trained for 5 epochs on 150,000 sequences, generated text is not guaranteed to remain grammatically or semantically consistent across longer outputs.
- **Evaluation Perspective**: Exact next-word accuracy is not the sole indicator of language-model quality for this assessment; cross-entropy loss reduction and token sampling diversity reflect expected learning dynamics across a 24,461-word vocabulary.
- The model produces stylistically representative text suited for an assessment baseline without claiming human-level or publication-ready literary coherence.

## Training History
- Training and validation loss curves are plotted and saved to `outputs/training_history.png`.
- The plot visually confirms convergence across epochs and validates stability between training and validation loss.

## Saved Model & Artifacts
The project produces the following artifacts:
- `models/best_model.keras`: Saved Keras model checkpoint containing the optimal weights.
- `models/vocab.json`: Vocabulary mappings (`word_to_id` and `id_to_word`).
- `outputs/training_history.png`: Plot of training vs. validation loss across epochs.
- `outputs/generated_text.txt`: Sample text generated from seed prompts.
- `outputs/run_metadata.json`: JSON file recording training parameters, dataset statistics, and execution time.

*Note: Large model files, generated assets, and raw dataset files are excluded from git via `.gitignore`.*

## How to Run Locally in VS Code

Run the following commands in the Windows CMD terminal from the project root directory:

### 1. Check Python Version
```cmd
python --version
```

### 2. Create and Activate Virtual Environment
```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 3. Upgrade Pip and Install Dependencies
```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run Preprocessing
```cmd
python -m src.preprocess
```

### 5. Train the Model
```cmd
python -m src.train
```

### 6. Run Default Text Generation
```cmd
python -m src.generate
```

### 7. Run Custom Text Generation
```cmd
python -m src.generate --seed "to be or not to be" --words 50 --temperature 0.8
```

You can also use the Windows batch scripts or the root runner:
```cmd
run_train.bat
run_generate.bat
```
or:
```cmd
python main.py train
python main.py generate
```

## Optional Experiment
The project includes an optional experiment module (`src/experiment.py` / `python main.py experiment`) configured to compare different sequence and model architectures:
- **Baseline**: Sequence length 20, 1 LSTM layer (128 units).
- **Deeper / Longer**: Sequence length 40, 2 LSTM layers (128 units each).

*Note: This experiment is provided as an optional extension. It is not marked as completed unless verified by existing project output records (`outputs/experiment_results.json`).*

## Project Structure
```text
data/
models/
outputs/
notebooks/
src/
main.py
requirements.txt
README.md
.gitignore
run_train.bat
run_generate.bat
```

- `data/`: Destination for downloaded raw Shakespeare text (gitignored).
- `models/`: Saved model weights (`best_model.keras`) and vocabulary (`vocab.json`).
- `outputs/`: Plots, generated text, and run metadata (gitignored).
- `notebooks/`: Jupyter notebook (`lstm_text_generation.ipynb`).
- `src/`: Source code modules (`preprocess.py`, `model.py`, `train.py`, `generate.py`, `experiment.py`).
- `main.py`: Entry point CLI to run training, generation, or experiments.
- `requirements.txt`: Python package requirements.
- `run_train.bat` & `run_generate.bat`: Windows CMD convenience scripts.
- `.gitignore`: Git exclusions for models, datasets, environments, and outputs.

## GitHub Instructions
The repository is structured to track only source code, configuration, scripts, and documentation:
- **Tracked in Git**: Python source files, batch scripts, configuration files, notebook, requirements, and documentation.
- **Excluded via `.gitignore`**: Raw text datasets (`data/*.txt`), Python virtual environment (`.venv/`), compiled model checkpoints (`models/*.keras`), and generated output files (`outputs/*`).

To push the repository to GitHub:
```cmd
git init
git add .
git commit -m "Complete Shakespeare LSTM text generation assessment"
git branch -M main
git remote add origin https://github.com/utkarsh-aix/-LSTM-text-generation.git
git push -u origin main
```

## Limitations
- **Large Vocabulary**: A vocabulary size of 24,461 words creates a large output layer and sparse prediction targets.
- **Exact Word Prediction**: Natural language generation from literary texts possesses high inherent entropy, making next-word prediction difficult.
- **Long-Sequence Coherence**: Without attention mechanisms or transformer-based architectures, LSTMs struggle to retain long-range narrative continuity.
- **Hardware Dependencies**: Training duration and resource consumption are constrained by local CPU/GPU execution speed.
- **Assessment Scope**: This model is an interview assessment demonstration of recurrent neural networks and NLP preprocessing, not a production-grade generative model.

