import plenoptic as po

from demos.example_torchinterface import TorchInterface, showcaseInterface
from torchvision.models.feature_extraction import get_graph_node_names

import robustbench

def rb_demo(iterations=100):
    model = robustbench.load_model(model_name='Bartoldson2024Adversarial_WRN-94-16', dataset='cifar10', threat_model='Linf')
    model.to(0)
    
    img = po.data.einstein(as_gray=False).to(0)
    train_nodes, eval_nodes = get_graph_node_names(model)
        # eval_nodes[:20]

    test_model = TorchInterface(model, "layer.2.block.14.conv_1")
    test_model = TorchInterface(model, "layer.0.block.1.batchnorm")
    test_model.to(0)

    showcaseInterface(iterations, test_model, img, None)
