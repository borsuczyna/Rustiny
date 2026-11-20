# Milestone 1 measurements

Machine: Windows 11, NVIDIA GeForce RTX 5050 Laptop GPU (discrete, selected) + Intel UHD 770, Vulkan loader 1.4.363. Release build (`--release`), validation off. One machine, small samples; no Unity comparison was performed, so **no claim about Unity is made**.

| Metric | Result | Design target | Verdict |
|---|---|---|---|
| Process start to exit, 2 frames rendered (n=5, ms), before start-up work | 3012 (cold), 1802, 1446, 1620, 1592; median of warm runs ~1.6 s | empty editor warm start < 1 s | see start-up section below |
| Start of `main` to ready (n=6, ms), after start-up work | 943-1109, median ~1010 | empty editor warm start < 1 s | **borderline, not reliably met** |
| Idle CPU (editor settled, 5 s window) | 31 ms CPU per 5 s = 0.63 % of one core | no needless work when idle | acceptable, not zero |
| Working set, idle | 173 MB | - | recorded only |
| Frame CPU time (debug-opt build, 89 frames, validation on) | median 6.3 ms, P95 13.2 ms, P99 63.9 ms (first frames: font atlas upload) | stable frame pacing | startup spike present |
| Validation (validation + sync validation, 120 frames incl. resize) | 0 errors | 0 errors | met |

## Start-up work (after "Przyspiesz start")

Measured by the `startup phases` log line (release, validation off, `--smoke-frames 2`, same machine). "Ready" = first frame can be drawn; it is counted from the start of `main`, so it excludes OS process creation, and the editor window is not guaranteed to have painted yet.

| Phase (ms) | Before (single thread) | After (instance on worker thread) |
|---|---|---|
| Window creation | 65-75 | 60-125 (runs in parallel with the instance now) |
| Vulkan loader + instance | 290-410 | 310-390 on the worker; the main thread waits only 240-330 |
| Device | 92-136 | 88-138 |
| First swapchain + frame resources | 380-550 | 384-486 |
| ImGui | 12-22 | 11-21 |
| **Ready, from start of `main`** | **980-1300** | **943-1109 (n=6, median ~1010)** |

Change made: `Instance::new` (Vulkan loader, layer and extension enumeration, `vkCreateInstance`) now runs on a worker thread started before the window is created, so the window creation cost is hidden behind it. Also removed the temporary swapchain timing code.

Verdict: about 100-150 ms faster, roughly 10 %. The < 1 s target is **borderline and not reliably met** (3 of 6 runs below 1 s). Small samples on one machine; wall-clock for the whole 2-frame smoke run is still 1.5-2.1 s because teardown (device idle, swapchain and instance destruction) adds roughly 0.5 s that a normal user does not wait for at start but does at exit.

What is left is mostly the driver: `vkCreateSwapchainKHR` takes 370-450 ms regardless of present mode (hybrid NVIDIA + Intel laptop), and instance creation 300-400 ms with broken third-party layer manifests on this machine. Disabling implicit layers did not help (neutral or slower), so that idea was dropped.

Ideas not done, because they were not measured to help or have a cost: creating the device on the worker as well (no gain while the window is already hidden behind the instance), showing the window only after the first frame (perceived, not real, time), and a swapchain created at a smaller image count (not tried).
