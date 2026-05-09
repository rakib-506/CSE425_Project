# Music Generation Project

This project explores multiple deep learning approaches for symbolic music generation using the **MAESTRO v3.0.0** piano MIDI dataset. The workflow starts with preprocessing raw MIDI files and then trains four different models:

1. LSTM Autoencoder
2. Variational Autoencoder (VAE)
3. Decoder-only Transformer
4. Reinforcement Learning with Human Feedback (RLHF) fine-tuning on the Transformer

The repository is organized as a sequence of Jupyter notebooks, where each later task builds on outputs produced earlier in the pipeline.

## Project Objectives

- preprocess raw MIDI data into model-ready formats
- train multiple generative models for piano music
- compare classical sequence modeling and latent-variable approaches
- generate new MIDI compositions
- improve Transformer outputs with reward-based fine-tuning

## Dataset

This project uses the **MAESTRO v3.0.0** dataset.

Expected dataset location:

```text
../maestro-v3.0.0
```

The preprocessing notebook expects the metadata CSV at:

```text
../maestro-v3.0.0/maestro-v3.0.0.csv
```

## Repository Files

### `00_preprocessing.ipynb`

This notebook prepares the dataset for all later tasks.

What it does:
- explores the MAESTRO dataset and prints summary statistics
- parses MIDI files and extracts note-event information
- converts MIDI files into piano-roll representations
- creates fixed-length sequence windows for autoencoder and VAE training
- creates tokenized event sequences for Transformer-based modeling
- saves processed arrays, vocabulary files, and plots under `outputs/`



### `01_task1_lstm_autoencoder.ipynb`

This notebook implements an **LSTM Autoencoder** for music generation from piano-roll sequences.

What it does:
- loads piano-roll sequence data created in `00_preprocessing.ipynb`
- trains an encoder-decoder LSTM model
- reconstructs input sequences and evaluates reconstruction quality
- generates new MIDI samples from the trained model
- saves checkpoints, plots, generated MIDI files, and evaluation metrics

Generated Midi_files_Link : https://drive.google.com/drive/folders/1_rRBDunOnOBrmAa5vC2YNGJmuvz4IY0W?usp=sharing



### `02_task2_vae.ipynb`

This notebook implements a **Variational Autoencoder (VAE)** for music generation.

What it does:
- loads the same piano-roll sequence data from preprocessing
- learns a structured latent space using reconstruction loss and KL divergence
- visualizes latent-space behavior
- performs interpolation and sample generation
- compares Task 2 results against Task 1

Generated Midi_files_Link : https://drive.google.com/drive/folders/11B-TJKlrdeD0YaGQkrIGTRYG0Z20QtQ8?usp=sharing

### `03_task3_transformer.ipynb`

This notebook implements a **decoder-only Transformer** for autoregressive music generation from token sequences.

What it does:
- loads tokenized training data and the event vocabulary from preprocessing
- trains a Transformer language model on symbolic music tokens
- evaluates the model using loss and perplexity
- visualizes attention patterns
- generates MIDI outputs from token predictions
- compares the Transformer against earlier baselines

Generated Midi_files_Link : https://drive.google.com/drive/folders/19XUHO8kZKtwb6oumTRijuL-bDsvHbTgU?usp=sharing

### `04_task4_rlhf.ipynb`

This notebook applies **Reinforcement Learning with Human Feedback (RLHF)** style fine-tuning to the pretrained Transformer from Task 3.

What it does:
- reloads the Transformer architecture and pretrained weights from Task 3
- defines an automated reward function as a proxy for human preference
- runs REINFORCE-style policy-gradient fine-tuning
- compares model behavior before and after RLHF
- generates final MIDI outputs and analysis plots

Generated Midi_files_Link : https://drive.google.com/drive/folders/1GTP3lV3qN9J3YcOhcm-znAlQPt1KK5JG?usp=sharing

## Execution Order

Run the notebooks in the following order:

1. `00_preprocessing.ipynb`
2. `01_task1_lstm_autoencoder.ipynb`
3. `02_task2_vae.ipynb`
4. `03_task3_transformer.ipynb`
5. `04_task4_rlhf.ipynb`

Why this order matters:
- Task 1 and Task 2 depend on sequence data created in preprocessing
- Task 3 depends on tokenized data and vocabulary from preprocessing
- Task 4 depends on the best Transformer checkpoint from Task 3

## Generated Folder Structure

After running the notebooks, the project creates an `outputs/` directory like this:

```text
outputs/
|-- processed/
|-- plots/
|-- task1/
|   `-- generated_midis/
|-- task2/
|   `-- generated_midis/
|-- task3/
|   `-- generated_midis/
`-- task4/
    `-- generated_midis/
```

## Requirements

The notebooks use the following Python libraries:

- `jupyter`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `torch`
- `pretty_midi`
- `music21`
- `tqdm`
- `scikit-learn`

Depending on your environment, you may also want a CUDA-enabled PyTorch installation for GPU training.

Example installation:

```bash
pip install jupyter numpy pandas matplotlib seaborn torch pretty_midi music21 tqdm scikit-learn
```

## How To Run

1. Place the MAESTRO dataset in the expected parent directory.
2. Open the notebooks in Jupyter Notebook or JupyterLab.
3. Run `00_preprocessing.ipynb` first.
4. Run the remaining notebooks in order.
5. Check the `outputs/` folder for saved models, plots, generated MIDI files, and evaluation results.

## Notes

- Several notebooks include GPU bootstrap cells for installing CUDA-enabled PyTorch if needed.
- On Windows, some notebooks disable cuDNN by default for LSTM stability on certain NVIDIA setups.
- Training can take significant time depending on dataset size, hardware, and model configuration.

## Summary

This repository presents a full music-generation pipeline:

- preprocessing symbolic music data from MAESTRO
- training and evaluating LSTM Autoencoder, VAE, and Transformer models
- generating MIDI outputs from each model
- fine-tuning the Transformer with RLHF-inspired reward optimization

It is suitable for demonstrating a progression from basic sequence autoencoding to modern token-based generation and reward-guided refinement.

Presentation video link: https://drive.google.com/file/d/1CkEv4i0oDF7B7SUY9lHTEERUC-OpYS0_/view?usp=drive_link

