import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from models import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


epoch_to_load = 3

gen_dir = f"./checkpoints/gan/epoch_{epoch_to_load}/generator_bert"

tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

generator_protbert = AutoModelForMaskedLM.from_pretrained(gen_dir).to(device)
generator = Generator(protbert_model=generator_protbert).to(device)

def generate_sequences(input_texts, temperature=1.0):
    """
    Generate sequences using the generator model.
    Args:
        input_texts (list of str): Input sequences for generation.
        temperature (float): Sampling temperature for generation.
    Returns:
        list of str: Generated sequences.
    """
    # Tokenize input texts
    inputs = tokenizer(input_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Generate sequences
    with torch.no_grad():
        generated_ids = generator.generate(input_ids, attention_mask=attention_mask, temperature=temperature)

    # Convert token IDs to text
    generated_sequences = [
        tokenizer.decode(generated_ids[i], skip_special_tokens=True) for i in range(len(generated_ids))
    ]
    return generated_sequences

if __name__ == "__main__":
    # Example input
    input_texts = [
        "MASKED sequence example 1",
        "MASKED sequence example 2"
    ]
    
    # Generate sequences
    generated_sequences = generate_sequences(input_texts, temperature=1.0)

    # Print results
    for i, seq in enumerate(generated_sequences):
        print(f"Input {i + 1}: {input_texts[i]}")
        print(f"Generated {i + 1}: {seq}")
        print("-" * 40)