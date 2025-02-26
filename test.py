""" from transformers import AutoTokenizer, EsmForProteinFolding, AutoModelForMaskedLM
from models import Generator
import torch
from val_metrics import calculate_plddt_scores_and_save_pdb, sample_sequence_length
from dataset import load_and_tokenize_dataset, get_dataloaders

torch.backends.cuda.matmul.allow_tf32 = True
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

print(tokenizer.vocab)

model_checkpoint_path = f"../checkpoints/dynamic/saved-260"

tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator_protbert = AutoModelForMaskedLM.from_pretrained(model_checkpoint_path).to(device)

generator = Generator(protbert_model=generator_protbert, mask_token_id = tokenizer.mask_token_id).to(device)
                                                                                                     
esmfold_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True).to("cuda")
esmfold_tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
esmfold_model.esm = esmfold_model.esm.half()
esmfold_model.eval()


avg_plddt_score = calculate_plddt_scores_and_save_pdb(generator, esmfold_tokenizer, esmfold_model, tokenizer,batch_size=4, num_sequences=10)
print(f"Average pLDDT score: {avg_plddt_score:.2f}") """


""" tokenized_datasets = load_and_tokenize_dataset(
    tokenizer,
    gen_file = "data/dnmt_gen.txt",
    critic_file = "data/dnmt_critic.txt",
    max_length = 512,
    fully_masked = True,
    full_dataset="data/dnmt_full.txt"
)
print(tokenized_datasets)
batch_size = 4
critic_dataloader = get_dataloaders(tokenized_datasets, batch_size) """
print("im here")