# Needle Browser Artifacts

The Needle demo can allocate deterministic same-shape weights locally, or load
converted checkpoint tensors from any public origin such as Cloudflare R2.

Convert the upstream pickle checkpoint:

```bash
python scripts/export_needle_for_r2.py checkpoints/needle.pkl /tmp/needle-r2
```

The converter writes `manifest.json` plus one raw `bfloat16` tensor file per
jax-js parameter, matching Needle's upstream checkpoint loader. Upload the whole
directory to this R2 prefix with public read access:

```text
data/demos/needle/needle/params/v1/
```

The canonical public manifest URL is:

```text
https://media.tanh.xyz/data/demos/needle/needle/params/v1/manifest.json
```

The app uses that URL by default. To override it for another bucket or local
artifact host, build with:

```bash
VITE_NEEDLE_MODEL_MANIFEST_URL=https://example.r2.dev/data/demos/needle/needle/params/v1/manifest.json bun run build
```

The browser loader expects the manifest tensor names used by
`src/demos/needle/models.ts`, for example `embedding`,
`encoderLayers.0.selfAttn.qProj`, and `decoderLayers.7.crossAttn.outProj`.
