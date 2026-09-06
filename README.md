# Ultra-Fast Astro Stacker (Tailored Hardware Build)

Customized high-performance image stacker optimized specifically for:
- **CPU:** AMD Ryzen 5 5600 (6 Cores / 12 Threads, Zen 3 Architecture)
- **RAM:** 32 GB DDR4 (Large in-memory frame cache)
- **GPU:** NVIDIA RTX 3050 6GB

## Hardware Optimizations Applied
- **Thread Scheduling:** Explicitly mapped to **12 threads** to fully saturate all SMT threads on Zen 3 without thread contention.
- **AVX2 & FMA Flags:** Vector instructions compiled for Zen 3 execution units.
- **In-Memory Buffering:** Takes full advantage of 32 GB RAM to eliminate disk I/O bottlenecks during calibration and sigma clipping.

## Windows Local Build (Visual Studio MSVC)

```cmd
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -A x64 -DOpenCV_DIR="C:/path/to/opencv/build" ..
cmake --build . --config Release
```

## Usage

```cmd
astro_stacker.exe output.png -light L1.jpg L2.jpg -dark D1.jpg -flat F1.jpg -bias B1.jpg
```
