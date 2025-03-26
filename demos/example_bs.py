import plenoptic as po

from demos.example_torchinterface import TorchInterface, showcaseInterface
from torchvision.models.feature_extraction import get_graph_node_names

import brainscore_vision

def bs_demo(iterations=100):
    model = brainscore_vision.load_model("alexnet_training_seed_01")
    
    img = po.data.einstein(as_gray=False).to(0)
    # model.layers
    # "This is the closest to what we want, but doesn’t work because it’s an array. I opened an issue asking about this:" https://github.com/brain-score/vision/issues/1848

    # test_model = TorchInterface(model, "features.0")
    # test_model.to(0)

    # showcaseInterface(iterations, test_model, img)
