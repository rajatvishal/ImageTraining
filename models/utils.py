import torch

# Helper functions for model utilities
def load_checkpoint(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)