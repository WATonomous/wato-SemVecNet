# wato_monorepo integration

Mirrors how `wato-cuda-bevfusion` is integrated: the monorepo clones this repo in
`docker/perception.Dockerfile`, and a thin vendor package builds `CUDA-SemVecNet/`
and exposes the imported target `cuda_semvecnet_vendor::semvecnet_core`.

Steps (in `~/Desktop/code/wato_monorepo`, nothing here is applied automatically):

1. Copy the vendor package:
   `cp -r integration/cuda_semvecnet_vendor <monorepo>/src/perception/semvecnet/cuda_semvecnet_vendor`
2. In `docker/perception.Dockerfile`, `source` stage, next to the bevfusion clone, add:
   ```
   git clone https://github.com/WATonomous/wato-SemVecNet.git wato-SemVecNet && \
   git -C wato-SemVecNet checkout <commit-or-branch> && \
   touch wato-SemVecNet/COLCON_IGNORE
   ```
   (`COLCON_IGNORE` stops colcon from picking up this repo's own files; the vendor package builds it.)
3. Add `cuda_semvecnet_vendor` (and your wrapper package) to `bloom/jazzy.ignored` and `bloom/rolling.ignored`.
4. In the ROS wrapper package:
   ```cmake
   find_package(cuda_semvecnet_vendor REQUIRED)
   target_link_libraries(<target> PUBLIC cuda_semvecnet_vendor::semvecnet_core)
   ```
5. The `.plan` is not baked into the image. Mount it and pass its path via a ROS parameter.
   Engines are tied to the TensorRT version + GPU, so build it with the monorepo's TensorRT.

Note: the vendor package needs at least one source file in `CUDA-SemVecNet/src/semvecnet/`,
otherwise there is no `libsemvecnet_core.so` to import.
