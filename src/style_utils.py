import torch

class StyleModel:
    def __init__(self, model, device="cpu"):
        self.model = model

    def get_cond(self, input):
        return self.model(input.last_hidden_state)

class ReduxImageEncoder(torch.nn.Module):
    def __init__(
        self,
        redux_dim: int = 1152,
        txt_in_features: int = 4096,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.redux_dim = redux_dim
        self.device = device
        self.dtype = dtype

        # Directly use torch.nn.Linear instead of the custom ops.Linear
        self.redux_up = torch.nn.Linear(redux_dim, txt_in_features * 3, dtype=dtype, device=device)
        self.redux_down = torch.nn.Linear(txt_in_features * 3, txt_in_features, dtype=dtype, device=device)

    def forward(self, sigclip_embeds) -> torch.Tensor:
        projected_x = self.redux_down(torch.nn.functional.silu(self.redux_up(sigclip_embeds)))
        return projected_x