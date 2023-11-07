Make sure to add the mimic-iii file inside data and download https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view?usp=sharing and place it inside data as well


Make sure to change the directory paths found in:
```
vim processing.py

# line 310-312
# med_file = '/data/mimic-iii/PRESCRIPTIONS.csv'
# diag_file = '/data/mimic-iii/DIAGNOSES_ICD.csv'
# procedure_file = '/data/mimic-iii/PROCEDURES_ICD.csv'
```

To create a conda environment from the yaml: 
conda env create -f mimic_env.yaml


```
def CopyDrug_batch(drugs, diags, procs, train_flag=True):
  """
  Generates a batch of copy drugs.

  Args:
    drugs: A list of drug names.
    diags: A list of diagnosis codes.
    procs: A list of procedure codes.
    train_flag: A boolean flag indicating whether to generate training data or test data.

  Returns:
    A list of copy drugs.
  """

  copy_drugs = []
  for drug in drugs:
    copy_drug = drug
    if train_flag:
      # Randomly choose a diagnosis code and a procedure code.
      diag = random.choice(diags)
      proc = random.choice(procs)

      # Add the diagnosis code and procedure code to the copy drug.
      copy_drug += " " + diag + " " + proc

    copy_drugs.append(copy_drug)

  return copy_drugs
def CopyDrug_tranformer(drugs, diags, procs, train_flag=True):
  """
  Generates a batch of copy drugs using a transformer model.

  Args:
    drugs: A list of drug names.
    diags: A list of diagnosis codes.
    procs: A list of procedure codes.
    train_flag: A boolean flag indicating whether to generate training data or test data.

  Returns:
    A list of copy drugs.
  """

  # Load the transformer model.
  transformer_model = load_transformer_model()

  # Generate copy drugs using the transformer model.
  copy_drugs = transformer_model.generate_copy_drugs(drugs, diags, procs)

  # Return the copy drugs.
  return copy_drugs

def CopyDrug_generate_prob(drugs, diags, procs, train_flag=True):
  """
  Generates a probability distribution over copy drugs for a given drug, diagnosis, and procedure.

  Args:
    drugs: A list of drug names.
    diags: A list of diagnosis codes.
    procs: A list of procedure codes.
    train_flag: A boolean flag indicating whether to generate training data or test data.

  Returns:
    A probability distribution over copy drugs.
  """

  # Load the transformer model.
  transformer_model = torch.load("transformer_model.pt")

  # Encode the drug, diagnosis, and procedure.
  drug_encoding = transformer_model.encode_drug(drugs)
  diag_encoding = transformer_model.encode_diagnosis(diags)
  proc_encoding = transformer_model.encode_procedure(procs)

  # Generate a probability distribution over copy drugs.
  copy_drug_probs = transformer_model.generate_copy_drug_probs(drug_encoding, diag_encoding, proc_encoding)

  # Return the probability distribution over copy drugs.
  return copy_drug_probs
import torch

def CopyDrug_diag_proc_encode(diags, procs, train_flag=True):
  """
  Encodes a list of diagnosis codes and a list of procedure codes into a single tensor.

  Args:
    diags: A list of diagnosis codes.
    procs: A list of procedure codes.
    train_flag: A boolean flag indicating whether to generate training data or test data.

  Returns:
    A tensor encoding the diagnosis codes and procedure codes.
  """

  # Encode the diagnosis codes and procedure codes.
  diag_encoding = torch.tensor(diags, dtype=torch.long)
  proc_encoding = torch.tensor(procs, dtype=torch.long)

  # Concatenate the diagnosis code and procedure code encodings.
  diag_proc_encoding = torch.cat((diag_encoding, proc_encoding), dim=1)

  # Return the diagnosis code and procedure code encodings.
  return diag_proc_encoding
```

I also had to change the mimic_env.yaml file:
```
- pytorch=1.10.0 # Had to use pytorch 1.10.0,
```
Also had to upgrade dill
