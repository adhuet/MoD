## GPU v1.1
### Minor memory management changes
v1.0: All memory allocations and transfers are done for each frame computation. Background computations are repeated for each frame.

Changes:
- [x] Background is processed only once
- [x] Memory is allocated and freed only once (predictable based on frame dimensions).

Notes:
- Only two Memcpy per frame are done: 1 for frame to d_frame, 1 for d_labels to labels. The rest of cudaMem ops are done only once outside the loop.

Effect:
Improvment of around +20fps, Memory mangement drops from 12% exec time to 6%, reduction of ~18% in total exec time.