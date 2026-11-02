# Minimal BEVFusion in PyTorch

## Usages

1. Download the official checkpoint from https://github.com/mit-han-lab/bevfusion/tree/main and save it to `data\checkpoints`.
2. Run the following command to launch the demo:

```bash
python .\test.py --checkpoint data/checkpoints/bevfusion-seg.pth --samples data/samples
```

See `.\test.py` for more details.

## Supported checkpoints

### BEV Map Segmentation (on nuScenes)

| Model                                                        | Modality | Checkpoint                                                   |
| ------------------------------------------------------------ | -------- | ------------------------------------------------------------ |
| [BEVFusion](https://github.com/mit-han-lab/bevfusion/blob/main/configs/nuscenes/seg/fusion-bev256d2-lss.yaml) | C+L      | [Link](https://www.dropbox.com/scl/fi/8lgd1hkod2a15mwry0fvd/bevfusion-seg.pth?rlkey=2tmgw7mcrlwy9qoqeui63tay9&dl=1) |

## Disclaimer

CAUTION: This repository has not been thoroughly tested or validated. Use it with caution and at your own risk. The authors assume no responsibility for any issues, losses, or damages arising from its use.

## License and provenance

The existing repository [LICENSE](LICENSE) is MIT, copyright 2026 Uzuki, and is preserved. [Third-party provenance](third_party/README.md) records the retained [BEVFusion Apache-2.0 license notice](third_party/BEVFusion-LICENSE). Original model sources and checkpoints remain external resources; their respective terms still apply.
