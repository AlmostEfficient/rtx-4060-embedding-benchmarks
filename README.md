# RTX 4060 embedding benchmarks

i wanted to know if an rtx 4060 is any good for running embedding models locally. i searched and found zero real numbers, just spec sheets and guesses. so i measured it, with an m4 mac mini next to it for comparison.

**tl;dr:**

- same javascript code on both machines: the 4060 embeds images **2.2x faster** than the mac mini (7.38 vs 3.31 frames/s).
- native cuda on the 4060: **5.7x** the mac's inference speed (19.48 vs 3.41 frames/s, inference only).
- 8 gb of vram is way more than you need for this. the best config peaked at 2.7 gb.
- use **fp16** weights, not q4. q4 was slower on both gpus.

tested october 2026. one model, small sample. caveats at the bottom.

## what i ran

**model:** [EmbeddingGemma 2](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX) (`onnx-community/embeddinggemma-2-ONNX`). google's small multimodal embedding model: a 170M param vision encoder plus a 270M param text model. each image goes in, one 768-d vector comes out. image budget was 140 tokens, which works out to 120 soft tokens (1,260 patches) for a 16:9 frame.

**workload:** 36 jpegs pulled from real video at 1 fps. a mix of talking heads, silent b-roll, drone shots and fast-cut edits. max 768 px on the long side. every run did 1 warmup pass, then 3 timed passes over all 36.

the images and model files were sha-256 matched across both machines, so both boxes did the exact same work.

| | mac mini | 4060 box |
|---|---|---|
| chip | apple m4, 10 cores | ryzen 5 7600, 6c / 12t |
| ram | 16 gb unified | 32 gb |
| gpu | integrated m4 gpu | **desktop** rtx 4060, 8 gb |
| os | macos 26.5 | windows 10 pro 19045, nvidia driver 610.88 |

**software:**

- **javascript (both machines):** bun 1.3.14 + transformers.js 4.3.1 on the webgpu backend. identical code and lockfile.
- **cuda (4060 only):** python 3.12.5 + `onnxruntime-gpu` 1.26.0, with cuda 12.9 and cudnn 9 pulled in by pip.

## results

### same code, both machines (jpeg in, vector out)

this is the full per-image cost: jpeg decode, preprocessing, inference, and reading the vector back. it doesn't include model load time or pulling frames out of a video file.

text model is q4 in every row. "vision" is the precision of the vision encoder.

| vision | machine | batch 1 | batch 4 | batch 8 |
|---|---|---:|---:|---:|
| q4 | mac mini, cpu | 0.61 fps | | |
| q4 | mac mini, webgpu | 1.07 fps | | |
| q4 | 4060, webgpu | 6.03 fps | 5.94 fps | 6.31 fps |
| fp16 | mac mini, webgpu | 3.19 fps | **3.31 fps** | 3.22 fps |
| fp16 | 4060, webgpu | 6.97 fps | **7.38 fps** | 7.37 fps |

best vs best: **7.38 / 3.31 = 2.23x** for the 4060.

if you compare both machines at q4 you get 5.6x, but that's mostly the mac choking on q4. fp16 is the fair fight.

### native cuda on the 4060

for this one i exported the exact preprocessed tensors from the mac, so both sides got byte-identical input. these numbers are **inference only**: no jpeg decode, no preprocessing. host/gpu transfers and readback are included, and profiling was on.

| vision | batch 1 | batch 4 | batch 8 |
|---|---:|---:|---:|
| q4 | 10.41 fps | 12.10 fps | 12.17 fps |
| fp16 | 15.00 fps | **19.48 fps** | 19.32 fps |

the mac's inference stage at the same config (fp16 vision, q4 text, batch 4) does 3.41 fps. so **19.48 / 3.41 = 5.72x**.

at batch 4 the vision encoder takes 45 ms per image and the text model takes 6 ms. the vision side is basically the whole cost.

### what that means in hours

at 1 fps, an hour of video is 3,600 frames.

| setup | per hour of footage | 100k images |
|---|---:|---:|
| mac mini, webgpu, fp16 | 18 min | 8.4 h |
| 4060, webgpu, fp16 | 8 min | 3.8 h |
| 4060, cuda, fp16 (inference only) | 3 min | 1.4 h |

the cuda row is inference only. decode + preprocessing measured about 7.6 ms per image on one js thread. if you do that in series before inference, you'd land around 17 fps. if you prep on other cores while the gpu runs, you should stay close to 19. i haven't measured that end to end yet, so treat 17 to 19 as an estimate.

## things i learned

**q4 is slower than fp16 on both gpus.** smaller file, slower run. on the mac, switching the vision encoder from q4 to fp16 tripled throughput (1.07 → 3.19 fps). on cuda, q4 → fp16 went from 12.1 to 19.5 fps at batch 4. my guess is that q4 weights get dequantized on the fly (onnx's `MatMulNBits` op), so you pay compute to save memory you didn't need saved. the text model stayed q4 everywhere since it's only 6 ms of the 51.

**batch 4 is the sweet spot.** going from batch 1 to 4 helped on cuda (15.0 → 19.5 fps). batch 8 added nothing and used more vram (2.7 → 4.3 gb).

**8 gb of vram is plenty for a model this size.** the fp16 vision encoder plus q4 text model is about 510 mb of weights. peak sampled vram was 2.7 gb on the best config and 4.8 gb on the worst one i tried (q4, batch 8).

**it stays cool, at least for short runs.** temps were sampled every 2 s. the best cuda run peaked at 58°C, and the hottest anything got was 66°C (webgpu, fp16, batch 1). gpu utilization hit 96 to 100% at batch 4 and up. power draw: the driver reported `N/A`, so i can't tell you watts or frames per joule.

**webgpu leaves a lot on the table on nvidia.** same gpu, same weights, same batch: the webgpu inference stage did 7.8 fps, cuda did 19.5. if you own a 4060, use cuda.

**cuda from javascript on windows wasn't an option.** when i tested, the windows build of `onnxruntime-node` shipped directml and webgpu but no cuda. so the cuda runs are python. it was painless, though. no system cuda toolkit, no driver changes:

```powershell
python -m venv .venv-cuda
& ./.venv-cuda/Scripts/python.exe -m pip install "onnxruntime-gpu[cuda,cudnn]==1.26.0" numpy
```

then call `ort.preload_dlls()` before creating a session, so it finds the cuda dlls inside the venv.

**not everything runs on the gpu.** about 1.8% of node executions fell back to cpu in both the vision and text graphs. it didn't stop the 4060 from winning, but it's not a pure gpu run.

**vectors match across machines.** with the same precision config, mac webgpu vs 4060 cuda had a mean cosine of 0.99982 (min 0.99903). mac vs 4060 at q4 on webgpu was basically identical (mean above 0.9999999999). so you can embed half your library on the mac and half on the 4060 and search them as one index.

**but don't mix precisions in one index.** in an earlier test, q4 vision + q4 text vs full fp32 was only 0.978 cosine. with this model, a good match and a bad match are often just ~0.07 apart, so that drift is enough to reshuffle results. pick one config and re-embed if you switch.

**mac notes:** don't run this on the m4 cpu (0.61 fps). in an earlier test on an m4 macbook, the coreml backend was about 3.5x slower than plain cpu on the vision encoder, so skip that too. webgpu with fp16 vision is the way.

## so what's the 4060 good for?

- **bulk image or video frame embedding with small models (under ~1B params).** this is what i tested. 100k images in under 4 hours on webgpu, ~1.4 hours of inference on cuda.
- **a box on your lan that does the embedding.** send it frames, get vectors back, search on whatever machine you want. since the vectors match, it doesn't matter which machine embedded what.
- **running cuda-only stuff at all.** a lot of ml tooling assumes nvidia, and cuda doesn't exist on a mac.

and the mac mini? 3.3 fps is slower, but 100k images in 8.4 hours is still one night. if you already have one, it works. if you're buying something for bulk embedding, the 4060 is the faster option.

## what i didn't test

- llms, whisper, ocr, or text-only embedding models
- the full video → frames → vectors → index pipeline
- whether fp16 changes search quality (the vectors agree, but i didn't re-judge results)
- long sustained jobs. each timed run was short (about 8 s on cuda, about 45 s on mac fp16), so the temps above aren't long-run thermals
- power usage
- the laptop 4060. it runs at a different power limit, so expect lower numbers

## caveats

- **36 images is a small sample.** it's enough to compare relative speed. it's not a promise for your dataset.
- **the cuda comparison is a different software stack** (python + onnxruntime 1.26) than the js runs. so 5.7x is "4060 + cuda stack vs mac + webgpu stack", not pure hardware.
- **cuda numbers had profiling on**, which adds a little overhead. the real number is probably slightly better.
- **one model.** bigger models or different architectures will scale differently.
