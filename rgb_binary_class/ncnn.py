# On your Mac — re-export with ONNX simplifier + opset 17
# pip install onnxsim

import torch
import torchvision.models as models
import onnx
from onnxsim import simplify

# Load your trained weights
model = models.mobilenet_v3_small()
model.classifier[-1] = torch.nn.Linear(1024, 2)
model.load_state_dict(torch.load("/Users/mac/Documents/PFE/rgb_binary_class/best_model_mobileNEtEnhanced.pt", map_location="cpu"))
model.eval()

# Export with opset 17 (better ARM kernel support in ORT)
dummy = torch.zeros(1, 3, 128, 128)
torch.onnx.export(
    model, dummy, "classifier_opt.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)

# Simplify — fuses ops, removes dead nodes
model_onnx = onnx.load("classifier_opt.onnx")
model_simplified, check = simplify(model_onnx)
assert check
onnx.save(model_simplified, "classifier_simplified.onnx")
print("Done")