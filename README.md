# Hospitality Intent Classifier

This project is part of my journey into learning and understanding how machine learning and NLP models can be applied in real-world domains like hospitality. I built this **Intent Classification System** to automatically detect a guest's query intent (like `BANQUET_BOOKING`, `ROOMS`, `PAYMENT`, etc.) from natural language messages.

---

## Dataset Used

Dataset: **[hospitality_intents_en](https://huggingface.co/datasets/WellaBanda/hospitality_intents_en)**  
Source: Hugging Face  
License: MIT  
Size: 82.4k labeled guest queries

This dataset includes labeled guest queries in English mapped to hospitality-related intents such as **AMENITIES**, **BOOKING**, **FOOD**, **ROOMS**, and more.

---

## Model

Used **TinyBERT** from Hugging Face for transfer learning. The model was chosen for its small size and efficiency, making it ideal for quick iteration and light-weight deployment.

Model link: [prajjwal1/bert-tiny](https://huggingface.co/prajjwal1/bert-tiny)

---

##  What I Learned

- How to fine-tune transformer models like BERT on custom datasets  
- Tokenization, preprocessing, and label encoding for multi-class intent classification  
- Writing a reusable inference class for predictions (`IntentPredictor`)  
- Building a pipeline with PyTorch and Hugging Face  
- Model evaluation using softmax probabilities and accuracy  
- Creating a Streamlit interface for live testing (optional for later deployment)

---

## Key Components

- `train_intent_model.py`: Fine-tuning script using Hugging Face Transformers  
- `model_output/`: Contains the tokenizer, model weights, config, label encoder  
- `predictor.py`: Custom class that loads and predicts user input intent  
- `text_length_distribution.png`: EDA visualization

---

## Why I Built This

I wanted to see how I could take raw guest queries (like those received by hotel chatbots or customer service) and classify them into **actionable categories** using machine learning. This project helped me solidify my understanding of end-to-end NLP pipeline.
---

## Future Plans

- Deploying this model using **Streamlit** or **Flask API**  
- Collecting more real-world messages to **fine-tune and improve accuracy**  
- Integrating it into a **Hotel AI Agent** that handles guest bookings and queries.