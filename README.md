# RetinaFace ONNX Export and Inference

This repository helps to convert retinface with `mobilenet` or `resnet50` backbones to `onnx`.

## 1. Install dependencies

```sh
pip3 install -r requirements.txt
```

## 2. Download weights

```sh
cd weights
./download-weights.sh
```

## 3. Export to onnx

```sh
# mobilenet
python3 export_onnx.py --trained_model weights/mobilenet0.25_Final.pth --network mobile0.25

# resnet50
python3 export_onnx.py --trained_model weights/Resnet50_Final.pth --network resnet50
```

## 4. Run inference

```sh
python3 inference_onnx.py
```