# SEA-RAFT v8.1 — strict safetensors shared BN alias fix

## Cause
SEA-RAFT `BasicBlock` registers a BatchNorm as both `bn3` and
`downsample.1`. A `.safetensors` model export may deduplicate the shared
storage, retaining `bn3.*` but dropping `downsample.1.*` aliases.
This appears as 16 missing BN tensor paths (`fnet/cnet` × `layer2/layer3`
× 4 fields) during PyTorch strict loading.

## Fix
Only if the model itself aliases the *same tensor* under both names,
recover the missing key from its checkpoint `bn3.*` sibling; validate
shapes/dtypes and retain `load_state_dict(..., strict=True)`.
If any genuine weight remains missing, strict loading still raises.
The 16 missing fields represent four BN aliases × four non-scalar BN
fields. The corresponding `num_batches_tracked` buffers may also be
restored (20 total keys) even if PyTorch did not list them in its error.

## Deploy
Run in project root; save original file first:

```bash
cp scripts/sea_raft_fullbackbone_v8.py scripts/sea_raft_fullbackbone_v8.py.before_alias_fix
ununzip -o Event-JEPA_SEARAFT_v8_1_StrictAliasFix.zip -d .
python -m pytest -q tests/test_v8_1_shared_bn_alias.py
```

Rerun the exact earlier `--mode eval-pretrained` command (choose a new
output directory). Check console for `[RESTORED SAFETENSORS SHARED BN ALIASES]`
and `[STRICT FULL RAFT LOAD]`. Output should show empty `missing` and
`unexpected`. If other missing keys appear, do not switch to strict=False;
inspect model/checkpoint versions.

Important: separate `resnet34-b627a593.pth` auto-download is standard
ImageNet initialization inside the upstream constructor, not the full
SEA-RAFT checkpoint. The latter still comes from
`pretrained/searaft_M/model.safetensors`.
