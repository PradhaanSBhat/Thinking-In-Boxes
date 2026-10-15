# Third-party code in `preparation/`

## `preparation/meshes/`

Derived from the **3D-Fixer** dataset toolkit, which in turn copied it from
**TRELLIS**. Cloning either repository is no longer required.

| Upstream file | Vendored as | What was kept |
|---|---|---|
| `dataset_toolkits/build_metadata.py` | `objaverse/metadata.py` | Building `metadata.csv` from the ARSG-110K listing, and merging the `downloaded_*.csv` shard records back into it |
| `dataset_toolkits/download.py` | `objaverse/download.py` | Resume filtering (`local_path` is null) and rank/world-size sharding |
| `dataset_toolkits/datasets/ObjaverseXL.py` | `objaverse/download.py` | Reading the ARSG-110K CSV and calling the Objaverse-XL downloader |

Everything else in those files — rendering, voxelisation, latent/feature
extraction, aesthetic scoring, the `--from_file` metadata rebuild and the
other dataset backends (ABO, HSSD, Toys4k, 3D-FUTURE) — is unused by this
pipeline and was not copied. Dropping it also drops the `utils3d` and
`easydict` dependencies.

Sources and licences:

- 3D-Fixer — https://github.com/HorizonRobotics/3D-Fixer — Apache License 2.0,
  Copyright (c) 2026 Horizon Robotics, D-Robotics and 3D-Fixer Contributors.
- TRELLIS — https://github.com/microsoft/TRELLIS — MIT License,
  Copyright (c) the TRELLIS authors.

Both licences require this attribution to be preserved. The Apache-2.0 text
is at https://www.apache.org/licenses/LICENSE-2.0.
