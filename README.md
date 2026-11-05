# Minimal BEVFusion in PyTorch

## Requirements

- pytorch
- cuda
- matplotlib
- onnxruntime (optional)

## Usages

### Inference

1. Download the official checkpoint from https://github.com/mit-han-lab/bevfusion/tree/main and save it to `data\checkpoints`.

2. Run the following command to compile and install the operators:

    ```bash
    scripts\install_ops.bat or scripts\install_ops.sh
    ```

3. Run the following command to launch the demo:

    ```bash
    python test_det.py --checkpoint data/checkpoints/bevfusion-det.pth --samples data/samples --config config/convfuser.yaml --device cuda --backend auto --no-show 
    ```

	or

    ```bash
    python test_seg.py - --checkpoint data/checkpoints/bevfusion-seg.pth --samples data/samples --device cuda --backend auto --no-show
    ```

See `.\test_det.py` and `.\test_seg.py`   for more details.

### ONNX

Run the following command to export the model in `.onnx` format:

```bash
python .\tools\export_onnx.py --checkpoint data/checkpoints/bevfusion-det.pth
```

## Supported checkpoints

### 3D Object Detection (on nuScenes)

| Model                                                        | Modality | Inference time | Checkpoint                                                   |
| ------------------------------------------------------------ | -------- | -------------- | ------------------------------------------------------------ |
| [BEVFusion](https://github.com/mit-han-lab/bevfusion/blob/main/configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml) | C+L      | ~107ms         | [Link](https://www.dropbox.com/scl/fi/ulaz9z4wdwtypjhx7xdi3/bevfusion-det.pth?rlkey=ovusfi2rchjub5oafogou255v&dl=1) |

### BEV Map Segmentation (on nuScenes)

| Model                                                        | Modality | Inference time | Checkpoint                                                   |
| ------------------------------------------------------------ | -------- | -------------- | ------------------------------------------------------------ |
| [BEVFusion](https://github.com/mit-han-lab/bevfusion/blob/main/configs/nuscenes/seg/fusion-bev256d2-lss.yaml) | C+L      | ~101ms         | [Link](https://www.dropbox.com/scl/fi/8lgd1hkod2a15mwry0fvd/bevfusion-seg.pth?rlkey=2tmgw7mcrlwy9qoqeui63tay9&dl=1) |

## Disclaimer

CAUTION: This repository has not been thoroughly tested or validated. Use it with caution and at your own risk. The authors assume no responsibility for any issues, losses, or damages arising from its use.

## License and provenance

The existing repository [LICENSE](LICENSE) is MIT, copyright 2026 Uzuki, and is preserved. [Third-party provenance](third_party/README.md) records the retained [BEVFusion Apache-2.0 license notice](third_party/BEVFusion-LICENSE). Original model sources and checkpoints remain external resources; their respective terms still apply.
