import torch
import sys
model = torch.load(sys.argv[1], map_location=torch.device('cpu'))
model.pop('optimizer')
torch.save(model, sys.argv[1])

