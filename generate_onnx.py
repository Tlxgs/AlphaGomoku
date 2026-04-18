import torch

from model import PolicyValueNet
model_path = "backup/model.pt"
model = PolicyValueNet.load_model(model_path, device="cpu")
model.eval()
model = model.float()
if hasattr(model, '_orig_mod'):
    model = model._orig_mod
dummy_input = torch.randn(1, 2, 15, 15)
fixed_onnx_path = "model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    fixed_onnx_path,
    input_names=['input'],
    output_names=['policy_logits', 'value'],
    opset_version=14,
    do_constant_folding=True,
)
print("静态形状 ONNX 导出成功")
