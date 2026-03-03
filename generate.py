import torch
from model import GPT
from data import encode, decode
from parameters import device

model = GPT().to(device)
model.load_state_dict(torch.load("checkpoints/model.pt", map_location=device))
model.eval()

while True:
    prompt = input("Enter prompt: ")
    context = torch.tensor([encode(prompt)], dtype=torch.long).to(device)

    generated = model.generate(context, max_new_tokens=200)[0].tolist()
    print(decode(generated))
