# PII Redaction with Piiranha - A Transformer-based Solution for Sensitive Data Masking

## Overview
In the evolving landscape of data privacy, protecting personally identifiable information (PII) in textual data is paramount. This repository introduces a robust solution for the detection and redaction of PII from text using the **Piiranha-v1** model. Leveraging state-of-the-art natural language processing (NLP) techniques, this transformer-based model specializes in identifying and securely obfuscating sensitive personal data such as names, addresses, phone numbers, and other confidential information.

## Abstract
The core mechanism of this solution hinges upon the **Piiranha-v1** model, a fine-tuned transformer-based architecture trained for token classification at the granularity of text tokens. The model employs tokenizers to preprocess input text and uses prediction logits to classify token-level sensitive entities, culminating in their redaction. The redaction process supports two modes:

- **Aggregated Redaction**: All PII entities are masked uniformly with a generic `[redacted]` placeholder.
- **Detailed Redaction**: Each specific PII type is individually anonymized (e.g., `[PERSON_NAME]`, `[PHONE_NUMBER]`).

This bifurcation ensures adaptability in diverse operational contexts, particularly for compliance with regulatory standards such as GDPR, HIPAA, and others.

## Technology Stack
- **Transformers**: Utilizes the Hugging Face Transformers library for pre-trained NLP models.
- **PyTorch**: The underlying framework for efficient deep learning and model inference.
- **Token Classification**: Segments text into tokens and classifies each based on its relevance to sensitive data.
- **CUDA Acceleration**: Supports GPU-accelerated inference for improved performance on compatible hardware.
- **Model**: `Piiranha-v1-detect-personal-information`, a specialized transformer model for detecting a wide array of PII entities.

## Installation Instructions

### Prerequisites
To run the solution locally, ensure the following are installed:
- Python 3.8 or higher
- PyTorch (with CUDA support for GPU acceleration, if applicable)
- Hugging Face Transformers library

### Installing Dependencies
Install the required libraries using pip:
```bash
pip install transformers
pip install torch
```

### Model Setup
Download and load the **Piiranha-v1** model using the Hugging Face Transformers library:
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_name = "iiiorg/piiranha-v1-detect-personal-information"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
```
Ensure the model and tokenizer are correctly loaded before proceeding with redaction tasks.

## Core Functionality

### Redaction Process
The model follows a two-phase methodology for PII redaction:
1. **Tokenization**: Input text is tokenized into sub-word units, which are processed by the transformer model for classification.
2. **Prediction and Masking**: The model generates predictions for each token, classifying it as a PII category or non-sensitive. PII tokens are replaced with a redaction placeholder (`[redacted]`) or their specific type.

### Key Parameters
- **Aggregate Redaction** (`aggregate_redaction=True`): Masks all detected PII with a generic `[redacted]` label. Ideal for applications where the nature of the PII is irrelevant or needs uniform obfuscation.
- **Detailed Redaction** (`aggregate_redaction=False`): Replaces each PII type with its specific category (e.g., `[PERSON_NAME]`, `[PHONE_NUMBER]`). Provides granularity for compliance purposes.

### Redaction Functionality
The following Python function implements the PII redaction process:
```python
def mask_pii(text, aggregate_redaction=True):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get the model predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted labels
    predictions = torch.argmax(outputs.logits, dim=-1)

    # Convert token predictions to word predictions
    encoded_inputs = tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=True)
    offset_mapping = encoded_inputs['offset_mapping']

    masked_text = list(text)
    is_redacting = False
    redaction_start = 0
    current_pii_type = ''

    for i, (start, end) in enumerate(offset_mapping):
        if start == end:  # Special token
            continue

        label = predictions[0][i].item()
        if label != model.config.label2id['O']:  # Non-O label
            pii_type = model.config.id2label[label]
            if not is_redacting:
                is_redacting = True
                redaction_start = start
                current_pii_type = pii_type
            elif not aggregate_redaction and pii_type != current_pii_type:
                # End current redaction and start a new one
                apply_redaction(masked_text, redaction_start, start, current_pii_type, aggregate_redaction)
                redaction_start = start
                current_pii_type = pii_type
        else:
            if is_redacting:
                apply_redaction(masked_text, redaction_start, end, current_pii_type, aggregate_redaction)
                is_redacting = False

    # Handle case where PII is at the end of the text
    if is_redacting:
        apply_redaction(masked_text, redaction_start, len(masked_text), current_pii_type, aggregate_redaction)

    return ''.join(masked_text)
```

**Note**: The `apply_redaction` function is assumed to be defined elsewhere in the codebase to handle the actual replacement of PII tokens.

## Example Usage
```python
example_text = "John Doe lives at 1234 Elm St, New York. His phone number is 555-123-4567."

# Aggregated redaction
masked_example_aggregated = mask_pii(example_text, aggregate_redaction=True)
print(masked_example_aggregated)

# Detailed redaction
masked_example_detailed = mask_pii(example_text, aggregate_redaction=False)
print(masked_example_detailed)
```

### Outputs
- **Aggregated Redaction**: All PII entities are replaced with `[redacted]`.
- **Detailed Redaction**: Each PII type is replaced with its corresponding identifier (e.g., `[PERSON_NAME]`, `[PHONE_NUMBER]`).

## Device Support
The model supports both CPU and GPU deployment. For accelerated inference, enable CUDA if available:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

## Contributing
Contributions are welcome! Please submit issues or pull requests to enhance this repository. Report bugs or suggest new features to improve the project.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- **Hugging Face Transformers**: For providing cutting-edge transformer models.
- **PyTorch**: For enabling efficient deep learning operations.
- **Piiranha Model**: For pioneering state-of-the-art PII detection.
