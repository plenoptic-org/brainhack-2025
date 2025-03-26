import plenoptic as po

from demos.example_torchinterface import TorchInterface, showcaseInterface
from torchvision.models.feature_extraction import get_graph_node_names

import torchvision

def tv_demo(iterations=100):
    # Load from torchvision
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    transform = torchvision.models.ResNet50_Weights.IMAGENET1K_V1.transforms()

    train_nodes, eval_nodes = get_graph_node_names(model)
        # print(eval_nodes[:20]) #list layer names
    test_model = TorchInterface(model, 'layer1.0.relu_2', transform)
    test_model.to(0);
    
    # Load image as if it were color by replicating across 3 channels
    img = po.data.einstein(as_gray=False).to(0)
    showcaseInterface(iterations, test_model, img, img.shape)

