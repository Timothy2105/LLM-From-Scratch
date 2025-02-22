from datasets import load_dataset

dataset = load_dataset("./openwebtext.py", trust_remote_code=True)

print("Dataset loaded!")
print(dataset)