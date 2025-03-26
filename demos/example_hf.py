import plenoptic as po

from demos.example_torchinterface import TorchInterface, showcaseInterface
from torchvision.models.feature_extraction import get_graph_node_names

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

def hf_demo(iterations=100):
    # Load from Huggingface Hub
    model = timm.create_model("hf-hub:nateraw/resnet50-oxford-iiit-pet", pretrained=True)
    
    # Set model to eval mode for inference
    model.eval()
    
    # Create Transform
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    # Load image as if it were color by replicating across 3 channels
    img = po.data.einstein(as_gray=False).to(0)
        # img = po.data.einstein().to(0)
        # img = img.repeat(1, 3, 1, 1)    

    # print(img.shape)
    # po.imshow(img); #256 x 256
    img_resnet_ready = transform(img)
    # po.imshow(img_resnet_ready); #224 x 224

    train_nodes, eval_nodes = get_graph_node_names(model)
    # print(eval_nodes) #list layer names
    test_model = TorchInterface(model, "layer2", transform)
    test_model.to(0);
    showcaseInterface(iterations, test_model, img, (1, 3, 224, 224))

