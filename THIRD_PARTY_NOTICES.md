# Graph WaveNet

The Graph WaveNet implementation in `trafficbench/models.py` is adapted from
Zonghan Wu's original PyTorch implementation:
https://github.com/nnzhan/Graph-WaveNet/blob/master/model.py

Changes include modern `Conv2d` usage for four-dimensional tensors, registered
graph-support buffers, consistent input/output layouts, and a smaller public API.
The architecture retains gated dilated temporal convolutions, two fixed directed
diffusion supports, learned adaptive adjacency, residual/skip connections, batch
normalization, and the original default channel widths.

Copyright (c) <2019> <Zonghan Wu>

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

# Data

METR-LA observations are downloaded from a pinned public mirror of the dataset
distributed with DCRNN. Sensor adjacency, road distances and coordinates are from
the original DCRNN repository. Download URLs, revisions and checksums are recorded
in `data/metr-la/manifest.json`. Cite the DCRNN paper when using its dataset.

# Liquid-network dependency

RGL-Flow uses `ncps==1.0.1` as a dependency, including the original PyTorch CfC
implementation by Mathias Lechner and Ramin Hasani. It is licensed under Apache
License 2.0: https://github.com/mlech26l/ncps. The liquid-network implementation is
not claimed as original work in this project.
