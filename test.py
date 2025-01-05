from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

print(tokenizer.vocab)
print(tokenizer.special_tokens_map)