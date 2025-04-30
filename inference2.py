import onnxruntime as ort
import numpy as np

x_d = np.float32([0.967568147250241,	-0.292564641522697,	-0.268022593291007,	-0.0680025161509444,	0.482420778264618,	1.34009067136198])
y_d = 0.190


Cs = [0.967568147250241,	-0.607397104011979,	-0.356972794479088,	-0.968478529510599,	0.482420778264618,	-41.6814052660887]  # Center values
Ss = [2.52232492135314,	0.564375712674976,	0.406153017236974,	0.900476013359655,	41.2322962227346,	45.8786387945936]  # Scale values

x_n = (x_d - Cs) / Ss

ort_sess = ort.InferenceSession("base_model_norm.onnx")
y = ort_sess.run(None, {'x': (np.expand_dims(np.float32(x_n), axis=0))})


ort_sess2 = ort.InferenceSession("base_model_denorm.onnx")
y2 = ort_sess2.run(None, {'x': (np.expand_dims(np.float32(x_d), axis=0))})


