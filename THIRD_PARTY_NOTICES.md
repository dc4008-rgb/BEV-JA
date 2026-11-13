# Third-party notices

## Included in this repository

### AdvRM — https://github.com/a-c-a-c/AdvRM (MIT License)

- `packages/base/parent_v17/01_公共数据集/car_00.jpg` and `car_00_mask.jpg` are unmodified copies of
  `asset/obstacle/car_00.jpg` and `asset/obstacle/car_00_mask.jpg` from the AdvRM repository
  (the vehicle cut-out and its mask used in all experiments).
- `packages/base/parent_v17/01_公共数据集/advrm_insertion.py` implements the AdvRM vehicle-insertion procedure, and
  `packages/base/parent_v17/03_方法/production/advrm_regularizer.py` the feature regularizer of the AdvRM objective.

```
MIT License

Copyright (c) 2024 a-c-a-c

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Used but not included

| Component | License / terms | How the experiments obtained it |
|---|---|---|
| Hugging Face `diffusers` 0.38.0 (the SD3 pipeline and flow-matching scheduler that `packages/runner/sd35_conditional_two_clip_v1.py` follows) | Apache-2.0 | `pip` |
| Stable Diffusion 3.5 Medium weights | Stability AI Community License (gated) | Hugging Face |
| YOLOv5 runtime source and `yolov5s.pt` | AGPL-3.0 (Ultralytics) | official release, kept outside this repository |
| torchvision detectors (Faster R-CNN, SSD300, FCOS) | BSD-3-Clause, pretrained weights under their own terms | torchvision |
| Monodepth2, DPT-Large, Depth Anything V1, ZoeDepth | each under its own license | official releases |
| KITTI object-detection images | CC BY-NC-SA 3.0 | official download |
| nuScenes v1.0-mini | nuScenes terms of use (non-commercial) | official download |

Verbatim third-party files that the original source tree keeps only as review evidence
(Adversarial Robustness Toolbox `robust_dpatch.py`; UMBC Vision contextual-patch files) are not published here;
they are listed in `OMITTED_FILES.md`.
