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

## GPU v1.2
### Shared/Constant Memory Usage for simple kernels
v1.1: some constant buffers can be moved to constant memory, and most of the "easy" kernels rely on global memory.
- grayscaleGPU consists of 4 incompressible memory access (3 read + 1 write) for each thread, so shared memory does not bring much here, and impact on performance is just negligible (less than 0.5% exec time on v1.1)
- blurGPU has the second highest exec cost (~20%). The gaussian blur matrix can be moved to constant memory and even be computed at compilation time on cpu, as the parameters will be left unchanged. Tiling might prove useful as every thread rely on locality to compute output.
- diffGPU and thresholdGPU, we can apply similar reasoning as grayscale, so no optimization
- morph has the highest exec time, but dilateGPU and erodeGPU are two kernels similar to blurGPU, so same reasoning can be applied (constant memory + tiling)
- connectedComponents, most of the exec time comes from the Memcpy for the symbollic labelled image, the rest is negligible before performance

Changes:
- [x] Use shared memory tiling to blurGPU (v1.1.1)
- [x] Use shared memory tiling to dilateGPU (v1.1.1)
- [x] Use shared memory tiling to erodeGPU (v1.1.1)
- [x] Feed blur matrix and morph kernel into constant memory (v1.1.2)
- [x] Fix shared memory tiling (v1.2)
<!-- - [ ] Compute host matrix and kernel at compile time with constexpr (v1.1.3) -->

Notes:
- On 32x32 blocks, the use of shared memory for blur actually shows a serious drop in performance for reasons yet unknown. This might be a memory spill issue, and the shared memory might mess with the cache. Will need to complete the rest of the optimizations to be sure.
- This modification applied to the morph kernels had the same impact. Performance drops drastically. Needs some further investigation.

Effect:
- Using shared memory drastically impacted performance, almost doubling the exec time for both blur and morph kernels
- Using constant memory to store kernel mask improved performance, with similar memory bandwidth impact for copy
- Overall, performance decreased. GPU V1.3 will focus on implementing solely constant memory usage, and drop the modification fo v1.2 for shared memory.
- After fix, performance is better as expected originally. Increase of ~55fps, and half the time for the new kernels.

## GPU v1.3
### Prefetching for shared memory kernels

v1.2: shared memory had a nice impact on performance, but blur and morph kernels still represent the majority of the exec time. I want to try various optimization for those in the following versions.
First one to tackle is prefectching. Right now, each block load a shared memory tile to be used in the current computation. The idea would be to have twice the amount of shared memory (which is possible taking into account our numbers) to handle two tiles and hide latency even more.

Changes:
- [ ] Implement shared memory prefetching for the blur and morph kernels

Notes:
- This project is coming to an end, and due to time restrictions, this version might be postponed indefinitely.

Changes:

## Future versions
### Various optimizations

Connected components is in itself and algorithm that is fairly optimized, and computation time stays low compared to other kernels.
The morph and blur shared memory kernels represent the bulk of the computation, but are basically simple 2D convolutions. A lot of optimizations can be applied to speed them up even more:
- prefecthing, and playing with instruction mix
- loop unrolling, specifically for memory loading, as the load itself relies on few registers, and can further improve instruction mix
- memory coalescing, the access strides cannot be simpler here, and memory is accessed is a coalesced way already, so there is not much to be done
- registers and shared memory optimizations, verifying occupancy with benchmark tools (Nvidia NSight), maybe play around cache/shared/constant size and see what's happening
- memory communication, it is already fairly optimized, with 2 memcpys per frame, which is the minimum for this computation, unless bboxes generation can be done on the GPU
- kernel launch overhead, all kernels are executed sequentially, and there is not much to be done here