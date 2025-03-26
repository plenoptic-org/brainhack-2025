import numpy as np
import plenoptic as po

import matplotlib.pyplot as plt
from typing import Optional, Callable, Any, Tuple, Union, Literal, List

import torch
from torchvision.models.feature_extraction import create_feature_extractor

class TorchInterface(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, return_node: str, transform: Optional[Callable] = None):
        super().__init__()
        self.extractor = create_feature_extractor(model, return_nodes=[return_node])
        self.model = model
        self.transform = transform
        self.return_node = return_node

    def forward(self, x):
        if self.transform is not None:
            x = self.transform(x)
        return self.extractor(x)[self.return_node]

    def plot_representation(
        self,
        data: torch.Tensor,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[float, float] = (15, 15),
        ylim: Optional[Union[Tuple[float, float], Literal[False]]] = None,
        batch_idx: int = 0,
        title: Optional[str] = None,
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        # Select the batch index
        data = data[batch_idx]

        # Compute across channels spatal error
        spatial_error = torch.abs(data).mean(dim=0).detach().cpu().numpy()

        # Compute per-channel error
        error = torch.abs(data).mean(dim=(1, 2))  # Shape: (C,)
        sorted_idx = torch.argsort(error, descending=True)
        sorted_error = error[sorted_idx].detach().cpu().numpy()

        # Determine figure layout
        if ax is None:
            fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [1, 1]})
        else:
            ax = po.tools.clean_up_axes(ax, False, ["top", "right", "bottom", "left"], ["x", "y"])
            gs = ax.get_subplotspec().subgridspec(2, 1, height_ratios=[3, 1])
            fig = ax.figure
            axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]

        # Plot average error across channels
        po.imshow(
            ax=axes[0], image=spatial_error[None, None, ...], title="Average Error Across Channels", vrange="auto0"
        )
        # axes[0].set_title()

        # Plot channel error distribution
        x_pos = np.arange(20)
        axes[1].bar(x_pos, sorted_error[:20], color="C1", alpha=0.7)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(sorted_idx[:20].tolist(), rotation=45)
        axes[1].set_xlabel("Channel")
        axes[1].set_ylabel("Absolute error")
        axes[1].set_title("Top 20 Channels Contributions to Error")

        if title is not None:
            fig.suptitle(title)

        return fig, axes

def showcaseInterface(iterations, mdl, img, image_shape=None):
    mdl.plot_representation(mdl(img))
    po.tools.remove_grad(mdl)
    mdl.eval()
    if image_shape:
        po.tools.validate.validate_model(mdl, device=0, image_shape=image_shape)
    else:
        po.tools.validate.validate_model(mdl, device=0)
    
    met = po.synth.Metamer(img, mdl)
    optim = torch.optim.Adam([met.metamer], lr=1e-3)
    # met.synthesize(iterations)
    met.synthesize(
        iterations,
        store_progress=10,
        optimizer=optim,
    )
    po.synth.metamer.plot_synthesis_status(met, ylim=False)