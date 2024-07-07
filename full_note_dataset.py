from datasets import load_dataset

ds = load_dataset("aisc-team-a1/augmented-clinical-notes")


notes_L = ds['train']['full_note'][:20]