import torch
import torch.nn as nn
import yaml
from pathlib import Path

from PrithviWxC.model import PrithviWxC


class PrithviBackbone(nn.Module):
    """
    Thin wrapper around the PrithviWxC model.

    Goal:
    - Load the pretrained PrithviWxC model (2.3B parameters).
    - Freeze all its parameters.
    - Provide a forward interface that we can later connect to the HR-Extreme encoder.

    For now, this class focuses on:
    - Instantiation from config + weights.
    - Freezing logic.
    - A placeholder forward that you can refine once we decide how exactly z_in
      should be injected into the Prithvi transformer stack.
    """

    def __init__(
        self,
        config_path: str,
        weights_path: str | None = None,
        in_mu=None,
        in_sig=None,
        static_mu=None,
        static_sig=None,
        output_sig=None,
        device: torch.device = torch.device("cuda"),
        load_weights: bool = True,
    ):
        """
        Args:
            config_path: Path to Prithvi config.yaml.
            weights_path: Path to pretrained weights .pt file.
            in_mu, in_sig: Input scalers (dynamic), as used in the original examples.
            static_mu, static_sig: Static input scalers.
            output_sig: Output anomaly variances (we use sqrt for std).
            device: Torch device to place the model on.
        """
        super().__init__()

        self.device = device

        # Load the config dictionary.
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        params = config["params"]

        # ---- default scalers if None ----
        if in_mu is None or in_sig is None:
            in_mu = torch.zeros(params["in_channels"])
            in_sig = torch.ones(params["in_channels"])
        if static_mu is None or static_sig is None:
            static_mu = torch.zeros(params["in_channels_static"])
            static_sig = torch.ones(params["in_channels_static"])
        if output_sig is None:
            # Prithvi expects a variance here; we take sqrt inside
            output_sig = torch.ones(params["in_channels"])

        # Instantiate the full PrithviWxC model as in the official examples.
        self.prithvi = PrithviWxC(
            in_channels=params["in_channels"],
            input_size_time=params["input_size_time"],
            in_channels_static=params["in_channels_static"],
            input_scalers_mu=in_mu,
            input_scalers_sigma=in_sig,
            input_scalers_epsilon=params["input_scalers_epsilon"],
            static_input_scalers_mu=static_mu,
            static_input_scalers_sigma=static_sig,
            static_input_scalers_epsilon=params["static_input_scalers_epsilon"],
            output_scalers=output_sig ** 0.5,
            n_lats_px=params["n_lats_px"],
            n_lons_px=params["n_lons_px"],
            patch_size_px=params["patch_size_px"],
            mask_unit_size_px=params["mask_unit_size_px"],
            mask_ratio_inputs=0.0,          # no masking for backbone use
            mask_ratio_targets=0.0,
            embed_dim=params["embed_dim"],
            n_blocks_encoder=params["n_blocks_encoder"],
            n_blocks_decoder=params["n_blocks_decoder"],
            mlp_multiplier=params["mlp_multiplier"],
            n_heads=params["n_heads"],
            dropout=params["dropout"],
            drop_path=params["drop_path"],
            parameter_dropout=params["parameter_dropout"],
            residual="none",                # we won't use climatology residual here
            masking_mode="global",
            encoder_shifting=True,
            decoder_shifting=True,
            positional_encoding="fourier",
            checkpoint_encoder=[],
            checkpoint_decoder=[],
        )

        # Only load checkpoint if requested and provided
        if load_weights and weights_path is not None:
            # weights_path = str(Path(weights_path).expanduser())
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
            if "model_state" in state_dict:
                state_dict = state_dict["model_state"]
            self.prithvi.load_state_dict(state_dict, strict=True)

        # Freeze backbone parameters
        for p in self.prithvi.parameters():
            p.requires_grad = False

    def forward_from_merra_batch(self, batch: dict) -> torch.Tensor:
        """
        Original Prithvi forward using a MERRA-2-style batch dict.

        This is useful for:
        - Sanity-checking that the backbone still works as expected.
        - Serving as a reference when we later define a custom forward for HR-Extreme.

        Args:
            batch: Dict with the same keys/structure as in the original
                   Prithvi inference scripts (after preproc).

        Returns:
            out: Tensor of shape (B, C, H, W) with Prithvi outputs.
        """
        with torch.no_grad():
            self.prithvi.eval()
            out = self.prithvi(batch)
        return out

    def forward_from_features(self, z_in: torch.Tensor) -> torch.Tensor:
        """
        Placeholder forward for HR-Extreme features.

        Intent (later):
        - Replace / augment Prithvi's internal embedding so that z_in is used to
          construct the token sequence passed into the transformer.

        For now:
        - Simply returns z_in to allow end-to-end wiring and testing.
        - This keeps the code functional while we design the exact injection point.

        Args:
            z_in: Tensor of shape (B, C_back_in, H_p, W_p), e.g. output of HREncoder.

        Returns:
            z_out: Currently just z_in; will later be the Prithvi-transformed features.
        """
        # TODO: implement integration with Prithvi tokenization / transformer stack.
        return z_in