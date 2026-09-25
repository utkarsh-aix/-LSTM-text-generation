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
- **No Cloud/API Dependency**: No external LLM APIs, cloud services, or API keys were used.

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
5. **Train/Validation Split**: Splits generated sequences into 90% training (270,000 sequences) and 10% validation (30,000 sequences) sets.
6. **Vocabulary Persistence**: Saves vocabulary mappings to `models/vocab.json`.

## Final Training Results & Configuration
The following baseline configuration and metrics reflect the completed final local training run:
- **Dataset**: Project Gutenberg Complete Works of Shakespeare
- **Total Words**: 988,682
- **Vocabulary Size**: 24,461
- **Sequence Length**: 20
- **Sequences Created**: 300,000
- **Training Sequences**: 270,000
- **Validation Sequences**: 30,000
- **Batch Size**: 256
- **Epochs Requested**: 5
- **Epochs Completed**: 5
- **Training Time**: approximately 2088.89 seconds (~34.8 minutes)
- **Best Validation Loss**: 6.78526

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
- **ModelCheckpoint**: Automatically saves the best-performing model weights to `models/best_model.keras`.

## Training Strategy & Loss Interpretation
The model is trained using a supervised next-word prediction objective:
- **Optimization**: Adam optimizer with a default learning rate of 0.001.
- **Loss Calculation**: Sparse Categorical Crossentropy applied directly to vocabulary integer targets without one-hot encoding overhead.
- **Regularization & Validation**: 10% held-out validation split (30,000 sequences) monitored across 5 epochs with batch size 256.
- **Callbacks**:
  - `EarlyStopping`: Monitors validation loss (`val_loss`) with `patience=2` to prevent overfitting and restore the best weights.
  - `ModelCheckpoint`: Automatically saves the best-performing model weights to `models/best_model.keras`.
- **Training Behavior**:
  - Training loss decreased consistently across all five epochs.
  - Validation loss decreased from approximately 7.01 down to approximately 6.78, and then slightly plateaued/increased in the final epoch.
  - The checkpointing callback retained the optimal model weights corresponding to the best validation loss (6.78526).

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

## Example Generated Outputs
The model was tested with multiple seed prompts, producing the following outputs saved in `outputs/generated_text.txt`:

### Seed: `"to be or not to be"`
> "to be or not to be polonius and my uncle but hold his trenches blood that s wonder and the ring of fire o they was rich but crew the titles of conquest with his mind which will his white which he ll not move virtue"

### Seed: `"the king"`
> "the king osric you shall prove a friends to tell you what therefore i have make some little weary mine but i have along there will you there eat you leave that with an trembling prince poins and now to retire he"

### Seed: `"love is"`
> "love is helen with many person one and worms is a voices that dispatch st thee back martius to see me menenius i shall give thee young cleopatra where s a son i have sworn him i so not bold the forest"

## Current Model Observations & Limitations
- **Vocabulary & Stylistic Patterns**: The model successfully learned Shakespeare-style vocabulary, character references (e.g., *Polonius*, *Osric*, *Menenius*, *Cleopatra*), and local phrasing.
- **Semantic & Grammatical Coherence**: Generated sequences can still contain grammatical and semantic inconsistencies over longer spans. This is an expected characteristic of a single-layer word-level LSTM trained on a fixed 20-word context window without attention mechanisms or transformer architectures.
- **No Claim of Human-Level Generation**: The model does not claim to generate perfect or human-level coherent literature; it demonstrates recurrent sequence modeling and probability sampling for an interview assessment.
- **High Entropy / Large Vocabulary**: A vocabulary size of 24,461 words creates a large output space and sparse prediction targets, making exact next-word prediction difficult.
- **Hardware Dependencies**: Local CPU/GPU execution constraints govern training throughput (~34.8 minutes for 300,000 sequences on the local Windows PC).

## Training History
- Training and validation loss curves are plotted and saved to `outputs/training_history.png`.
- The plot visually confirms consistent reduction in training loss across epochs alongside initial validation loss reduction to 6.78526 before plateauing.

## Saved Model & Artifacts
The project produces the following artifacts:
- `models/best_model.keras`: Saved Keras model checkpoint containing the optimal weights.
- `models/vocab.json`: Vocabulary mappings (`word_to_id` and `id_to_word`).
- `outputs/training_history.png`: Plot of training vs. validation loss across epochs.
- `outputs/generated_text.txt`: Sample text generated from test seed prompts.
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

## Reproducibility Notes
- **Fixed Random Seed**: The training script sets a fixed random seed (`SEED = 42`) for TensorFlow and NumPy to ensure reproducibility across runs.
- **Model Checkpointing**: `ModelCheckpoint` ensures that only the weights achieving the best validation loss (6.78526) are saved to `models/best_model.keras`, regardless of late-epoch plateauing.
- **Deterministic Preprocessing**: Data cleaning, vocabulary generation, and integer encoding follow a strictly deterministic pipeline.
- **Self-Contained Execution**: All dependencies are locked in `requirements.txt`, and the full pipeline runs locally on Windows without cloud dependencies.

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

## GitHub Usage Instructions
The repository is structured to track only source code, configuration, scripts, and documentation:
- **Tracked in Git**: Python source files, batch scripts, configuration files, notebook, requirements, and documentation.
- **Excluded via `.gitignore`**: Raw text datasets (`data/*.txt`), Python virtual environment (`.venv/`), compiled model checkpoints (`models/*.keras`), and generated output files (`outputs/*`).

To commit and push updates:
```cmd
git add .
git commit -m "Finalize Shakespeare LSTM text generation project"
git branch -M main
git push -u origin main
```
