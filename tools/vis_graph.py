import netron
import os

graph_path = ''
onnx_path = os.path.join(graph_path, "model_vis.onnx")
netron.start(onnx_path)