# AI-Based Targeted Password Guessing

This project implements AI-based targeted password guessing using transformer models (T5, BART, and Llama) to predict passwords based on user information and context. The system can be trained on custom datasets and used to generate password guesses for security research and penetration testing purposes.

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- Other dependencies listed in requirements (see ```requirements.txt`` file)

## Installation

1. Clone the repository:

2. Install required packages:
```bash
pip install requirements.txt
```

3. Set up your Hugging Face token in `tunners/constants.py` if using gated models like Llama.

## 📁 Project Structure

```
├── config.json              # Configuration for training and testing
├── generate_guesses.py       # Main script for password generation
├── tunne_models.py          # Model training script
├── generators/
│   ├── generator.py         # Base generator class
│   └── llama_generator.py   # Llama-specific generator
├── tunners/
│   ├── bart_tunner.py       # BART fine-tuning
│   ├── llama_tunner.py      # Llama fine-tuning
│   ├── t5_tunner.py         # T5 fine-tuning
│   ├── tunner.py            # Base tuner class
│   └── constants.py         # Training constants and hyperparameters
└── dev/                     # Development files
```

## Configuration

The main configuration is defined in `config.json` with separate sections for training and testing:

### Training Configuration
```json
{
  "id": 0,
  "name": "T5",
  "description": "T5 training",
  "model_name": "google-t5/t5-base",
  "model_path": "t5/model",
  "training_set_path": "datasets/train.jsonl"
}
```

### Testing Configuration
```json
{
  "id": 0,
  "name": "T5",
  "model_path": "t5/model",
  "test_set_path": "datasets/test.jsonl",
  "guesses": 1000,
  "temperature": 1.0,
  "top_k": 50,
  "top_p": 0.9
}
```

## Dataset Format

### Training Dataset
Each line should be a JSON object with input-output pairs:
```json
{"input": "example@email.com", "output": "password123"}
```

### Test Dataset
Test data includes additional metadata:
```json
{"input": "example@email.com", "output": "password123", "email": "example@email.com"}
```

### Advanced Input Formatting
You can include rich user information (up to 128 characters by default):
```json
{"input": "Username: johnsmith84@yahoo.com\nName: John Smith\nNationality: Unknown\nGender: Male\nYear: 1984", "output": "password123"}
```

## Usage

### Training Models

Train a model using the configuration file:
```bash
python tunne_models.py config.json <model_id>
```

Where `<model_id>` corresponds to the model configuration in your JSON file:
- `0` for T5
- `1` for BART
- `2` for Llama

### Generating Password Guesses

Use the trained models to generate password guesses:
```bash
python generate_guesses.py config.json <model_id>
```

## ⚙️ Customization

### Hyperparameters
Modify training parameters in `tunners/constants.py`:
- `MAX_MODEL_INPUTS_LENGHT`: Maximum input length (default: 128)
- `LEARNING_RATE`: Learning rate for training (default: 3e-5)
- `NUM_TRAIN_EPOCH`: Number of training epochs (default: 2)
- `PER_DEVICE_TRAIN_BATCH_SIZE`: Training batch size (default: 16)

### Generation Parameters
Configure generation behavior in the test configuration:
- `guesses`: Number of password guesses to generate
- `temperature`: Controls randomness (higher = more random)
- `top_k`: Limits vocabulary to top-k tokens
- `top_p`: Nucleus sampling threshold

## Evaluation

The system provides built-in evaluation metrics including:
- Password match accuracy
- Levenshtein distance analysis
- Generation time statistics
- Success rate at different guess counts

## Ethical Considerations

This tool is designed for:
- Security research
- Penetration testing with proper authorization
- Educational purposes

**Please use responsibly and only on systems you own or have explicit permission to test.**

## License

This project is for educational and research purposes. Please ensure compliance with your local laws and regulations when using this tool.

