# CHANGELOG


## v1.21.3 (2025-05-23)

### Performance Improvements

- Kill ray if initialized to free up memory
  ([`3d8fa4b`](https://github.com/mbari-org/sdcat/commit/3d8fa4b3b53f89b13791989ddfe3f8999cacf3c1))


## v1.21.2 (2025-05-22)

### Performance Improvements

- Improved cluster batch performance for GPU acceleration
  ([`5a90dd9`](https://github.com/mbari-org/sdcat/commit/5a90dd98b330950f0cc2a42664b85d55044d06d4))


## v1.21.1 (2025-05-22)

### Performance Improvements

- Improved cluster cropping performance by moving from modin to pandas and grouping by frame and
  more logging
  ([`3713e7e`](https://github.com/mbari-org/sdcat/commit/3713e7e1ed15f750c90480d1f25c57a0910f5700))


## v1.21.0 (2025-05-22)

### Documentation

- More detail on SAHI
  ([`04eee46`](https://github.com/mbari-org/sdcat/commit/04eee46c43454e82b1ba8b780cb2341dd7cd85d4))

### Features

- Fast batch clustering ([#25](https://github.com/mbari-org/sdcat/pull/25),
  [`38cc5b5`](https://github.com/mbari-org/sdcat/commit/38cc5b5ca0dce187114feca7cd01cf12cbe4bcd1))

Optimized batch clustering with improved reporting. New features include:

* Clustering large archives. Tested on 1.5 million ROIs * Clustered results saved to parquet format
  to facilitate better downstream processing * Better aggregation and reporting on cluster summary
  in human readable JSON format, e.g. { "dataset": { "output": "/data/output",
  "clustering_algorithm": "HDBSCAN", "clustering_parameters": { "min_cluster_size": 2,
  "min_samples": 1, "cluster_selection_method": "leaf", "metric": "precomputed", "algorithm":
  "best", "alpha": 1.3, "cluster_selection_epsilon": 0.0 }, "feature_embedding_model":
  "MBARI-org/mbari-uav-vit-b-16", "roi": true, "input": [ "/data/input ], "image_count": 328 },
  "statistics": { "total_clusters": 4, "cluster_coverage": "1.23 (122.94%)", "top_predictions": {
  "class": "Shark", "percentage": "3.35%" } }, "sdcat_version": "1.20.4" }


## v1.20.3 (2025-03-19)

### Bug Fixes

- Remove last cluster embedding correctly and fetch embeddings from dataframe
  ([`e955744`](https://github.com/mbari-org/sdcat/commit/e955744ca9ea961b654eacc58d7b12a0df55ea69))

### Performance Improvements

- Adjust min similarity to a more conservative .9
  ([`697f138`](https://github.com/mbari-org/sdcat/commit/697f13839a1e09f4cd5d25d021146a40fb4eb7e3))


## v1.20.2 (2025-03-13)

### Bug Fixes

- Handle cases where no detections exists for specified start/end images by skipping over
  ([`87be2b7`](https://github.com/mbari-org/sdcat/commit/87be2b7f9547c268ce5afa74065acdce18970b9f))

### Performance Improvements

- Always assign noise cluster and correct label assign
  ([`5dd83bb`](https://github.com/mbari-org/sdcat/commit/5dd83bb30d63d0c2b227e714c1328fa3f042db6a))

- Better defaults for cluster alpha and epsilon
  ([`237890d`](https://github.com/mbari-org/sdcat/commit/237890df9db70c3babed4f36e8d9af5cff0a646c))

- Change combined detection+classify score to average
  ([`8def7a2`](https://github.com/mbari-org/sdcat/commit/8def7a2624c8dc7ac94853c40f3ea9e04911afa2))


## v1.20.1 (2025-03-12)

### Bug Fixes

- Correct arg for weighted score in cluster
  ([`658ef95`](https://github.com/mbari-org/sdcat/commit/658ef9552b5c66d6f656ce461738569d1c7dfa2c))


## v1.20.0 (2025-03-12)

### Bug Fixes

- Remove cuda in cluster vits
  ([`d449a86`](https://github.com/mbari-org/sdcat/commit/d449a86394e6f560a2ea4f37cb1fa77c1b166a18))

### Chores

- Clean up imports
  ([`18da1aa`](https://github.com/mbari-org/sdcat/commit/18da1aadac42ee3a4102c9cc14f7dc07a3306624))

### Continuous Integration

- Only run pytest on push to main branch
  ([`fcfb28d`](https://github.com/mbari-org/sdcat/commit/fcfb28d5cd3555ee7397bea1645691fe6a88adbe))

### Documentation

- Updated workflow diagram
  ([`91ebe6a`](https://github.com/mbari-org/sdcat/commit/91ebe6a512b10d306a04e95ed373890b6e424ab7))

### Features

- Add weight_vits option to weight the scores from the detection model in the vits classification
  model
  ([`9bffe3d`](https://github.com/mbari-org/sdcat/commit/9bffe3d589a94081f0c9e9a9eb49e5a31e3e7f22))

- Added feature merge
  ([`90d182d`](https://github.com/mbari-org/sdcat/commit/90d182db13d8e56660f349bd62426dd18dc81b38))

- Added hdbscan algorithm choice for clustering
  ([`40b9ba2`](https://github.com/mbari-org/sdcat/commit/40b9ba21f685be52ee7f644fc2f25e2a4c3a7859))

- Added hdbscan algorithm choice for clustering
  ([`722da61`](https://github.com/mbari-org/sdcat/commit/722da6163aba7cfae937508a16d1e18209326349))

- Added min-sample-size argument to allow for parameter sweeps
  ([`2cf9771`](https://github.com/mbari-org/sdcat/commit/2cf9771b40090375f146e02082156a5757ae9044))

- Rename to weighted_score and add back in the noise reassignment for higher coverage
  ([`0a361a0`](https://github.com/mbari-org/sdcat/commit/0a361a0eee69fd08a821dc96cb31decea7fa777a))

### Performance Improvements

- Improved cluster coverage, weighted classification scores, and more options for running cluster
  sweeps
  ([`a94e4a9`](https://github.com/mbari-org/sdcat/commit/a94e4a91faed622e335bc8da450da1a5d61097e3))

Performance - Better handling of noise cluster and merging similar clusters. This should improve
  cluster coverage and generate somewhat larger clusters with foundation models.

Features - new arg to sdcat cluster `--algorithm` default "best"; prims_kdtree or boruvka_kdtree may
  be worth trying - new arg to sdcat cluster `--min-sample-size` which was only supported in the
  .ini file - new arg to sdcat cluster `--weighted-score` which will weight the classification score
  with the detection score from the ViTS models through multiplication


## v1.19.1 (2025-02-26)

### Bug Fixes

- Correct handling of bounded end image
  ([`f510e16`](https://github.com/mbari-org/sdcat/commit/f510e160197ac8b819376395de8cdaae2ac0072a))

### Features

- Second pass merge cluster
  ([`a58ce18`](https://github.com/mbari-org/sdcat/commit/a58ce189473b2a66fda10096d631bbffbafee2d6))


## v1.19.0 (2025-02-26)

### Features

- Added softcluster ([#19](https://github.com/mbari-org/sdcat/pull/19),
  [`cfaf784`](https://github.com/mbari-org/sdcat/commit/cfaf784c831bb61ee5998776a00c644f5824f801))

* perf: better defaults for finer-grained clustering with google model

* feat: added soft clustering for leaf method only

* fix: remove default as this overrides what is in the .ini file

* perf: add batch size as command option --batch-size; default is 32 but best size depends on
  GPU/model memory

* fix: correct args for multiproc

* perf: combine soft/fuzzy and cosine sim

* docs: update workflow diagram with soft/fuzzy algorithm

* fix: handle models that only output top 1

* fix: only capture top 2 classes and scores

* chore: merged changes from main


## v1.18.2 (2025-02-20)

### Bug Fixes

- Only capture top 2 classes and scores
  ([`9b85463`](https://github.com/mbari-org/sdcat/commit/9b854638b2303dd03752195d13bb3e68bd3dc291))


## v1.18.1 (2025-02-20)

### Bug Fixes

- Handle models that only output top 1 and default to cuda if available if not specified for
  clustering
  ([`164480a`](https://github.com/mbari-org/sdcat/commit/164480a2b25544e05152f136757cfdb2c1d989ef))


## v1.18.0 (2025-02-20)

### Features

- Add support for --save-roi --roi-size ([#18](https://github.com/mbari-org/sdcat/pull/18),
  [`9a801ac`](https://github.com/mbari-org/sdcat/commit/9a801ac34710f82d1b62bcd561253887858aa6cf))

Added `--save-roi` and `--roi-size `options to sdcat detect. This saves the crops in a location
  compatible with the clustering stage, but can also be used outside of sdcat. Data saved to crops

├── det_filtered # The filtered detections from the model ├── crops # Crops of the detections

- Trigger release for --save-roi
  ([`8240a74`](https://github.com/mbari-org/sdcat/commit/8240a74d6363578509cc29196244b6c28d34d2c2))


## v1.17.0 (2025-02-07)

### Build System

- Relaxed requirements for compatibility with mbari-aidata since these are often used together
  ([`8bf55e3`](https://github.com/mbari-org/sdcat/commit/8bf55e3c6e29660766e87622b3cc791772767d70))

### Features

- Trigger release to pypi with latest deps
  ([`2490823`](https://github.com/mbari-org/sdcat/commit/249082389403626c963012cbc5205bb8d2ba24e4))


## v1.16.3 (2025-01-27)

### Build System

- Updated poetry lock
  ([`b9a04e6`](https://github.com/mbari-org/sdcat/commit/b9a04e6bc3f2afab75bd9e3aea88b850967d7908))

### Performance Improvements

- Bump sahi to support YOLOv11
  ([`d36b494`](https://github.com/mbari-org/sdcat/commit/d36b4942daac1614bd2b3cbac6e17c2300e06c21))


## v1.16.2 (2025-01-14)

### Performance Improvements

- Better handling of cuda devices by id across both detection and clustering commands with --device
  cuda:0
  ([`ae8e395`](https://github.com/mbari-org/sdcat/commit/ae8e3958cca9751b0c4d1548174a30ca974636a8))


## v1.16.1 (2025-01-13)

### Bug Fixes

- Correct argument order to create_model and added types for float/int args in detect
  ([`6bb93bb`](https://github.com/mbari-org/sdcat/commit/6bb93bba765991d3255404a209c329441e4fb175))


## v1.16.0 (2025-01-11)

### Features

- Added support for auto-detecting detection model types from huggingface and loading models from a
  directory. If models do not have the model type encoded in the name, e.g. yolov5 the --model-type
  yolov5 must be used
  ([`3ea7612`](https://github.com/mbari-org/sdcat/commit/3ea76120344e53936bc9cd63cea8815106abf312))


## v1.15.0 (2025-01-10)

### Features

- Add second score and class assignment for roi cluster
  ([`8412941`](https://github.com/mbari-org/sdcat/commit/84129413fc87730f2b83a2fbdddf6d317614b269))


## v1.14.2 (2025-01-10)

### Bug Fixes

- Copy rois to crop path to avoid removal
  ([`00ca30f`](https://github.com/mbari-org/sdcat/commit/00ca30f288d94ab6b6325101a1242ff4280ed9e1))

### Performance Improvements

- Remove only dark and blurry
  ([`1e0de1f`](https://github.com/mbari-org/sdcat/commit/1e0de1fff4b2039d9b6e9c0569e8332bfe4eb29e))


## v1.14.1 (2024-12-07)

### Bug Fixes

- Correct clean_vision for roi and added check for is_low_information_issue and
  is_near_duplicates_issue
  ([`849d432`](https://github.com/mbari-org/sdcat/commit/849d4323dabb79dc98a53b5bf52066ba08271f13))


## v1.14.0 (2024-11-27)

### Features

- Remove dark and blurry examples in clustering using cleanvision
  ([`c04fab7`](https://github.com/mbari-org/sdcat/commit/c04fab70749799891d0b59e78a0a47c76613fe5d))


## v1.13.2 (2024-11-23)

### Bug Fixes

- Correct vits assignment
  ([`d35ae96`](https://github.com/mbari-org/sdcat/commit/d35ae96b8d64f154621d7dfdc5ea2625fa12cf19))


## v1.13.1 (2024-11-23)

### Bug Fixes

- Handle index out of range on vits assign
  ([`2aee31a`](https://github.com/mbari-org/sdcat/commit/2aee31aa2b468a6b693a9710dfc896e5cdde236c))

### Refactoring

- Rename vits prediction option to --use-vits
  ([`a1e1301`](https://github.com/mbari-org/sdcat/commit/a1e13018dbe6b18b0b91f738ebe88ab1417f2def))


## v1.13.0 (2024-11-21)

### Bug Fixes

- Assign exemplar to crop
  ([`05b97c9`](https://github.com/mbari-org/sdcat/commit/05b97c9529c049ab62e9096ea0d90310f16b39ab))

- Correct handling of remove corner
  ([`60df1ae`](https://github.com/mbari-org/sdcat/commit/60df1aeed140928c3b741e6873bd89a7117ebaec))

- Correct order of file/byte for running vss
  ([`1d896ee`](https://github.com/mbari-org/sdcat/commit/1d896ee11fbab5965e3862b7923ed3f662722694))

### Build System

- Updated poetry
  ([`2d97ed5`](https://github.com/mbari-org/sdcat/commit/2d97ed50a89ec3e8514631d072ac2d0b1ccd4c67))

### Chores

- Switch projects
  ([`22090f3`](https://github.com/mbari-org/sdcat/commit/22090f35d2d9a6758d894a7049bc305d639e08cc))

### Features

- Added pass through of vss server and renaming cluster id in exemplar output
  ([`67c4202`](https://github.com/mbari-org/sdcat/commit/67c4202c6ca7d661ee2cdfadb922b62d7789efa1))

- Added support for assigning predictions to clusters
  ([`1b674fe`](https://github.com/mbari-org/sdcat/commit/1b674fe1ace2d0c5ab3e1a70c6e082b1064df065))

- Assign unknowns via vss
  ([`947a8a3`](https://github.com/mbari-org/sdcat/commit/947a8a387150f4c908380647664ece8dd5d57113))

### Performance Improvements

- Assign everything, not just clusters and assign top prediction if there is more than a .05 spread
  in the top 2
  ([`f6cf171`](https://github.com/mbari-org/sdcat/commit/f6cf171b832a598a51255aed25e3d9dc64a3f778))


## v1.12.1 (2024-10-29)

### Performance Improvements

- Always run saliency on multiproc regardless of cpu or gpu as it is not gpu enabled
  ([`b7e913e`](https://github.com/mbari-org/sdcat/commit/b7e913e855899c90c89eaa3f32d6156a8ffc6849))


## v1.12.0 (2024-10-29)

### Bug Fixes

- Correct boolen for remove corners
  ([`4161e62`](https://github.com/mbari-org/sdcat/commit/4161e62aec5856beea58c9e3b251f58d9dbc93ce))

### Chores

- Default to include all detections in cluster
  ([`0bf446a`](https://github.com/mbari-org/sdcat/commit/0bf446a99eabe071e5056c4fb2486ffa0891219d))

### Continuous Integration

- Add yolov5 to build
  ([`b8c2eba`](https://github.com/mbari-org/sdcat/commit/b8c2ebac0f3d1f4c2208e87a01fe563e509841e4))

### Features

- Added --model MBARI/yolov5x6-uavs-oneclass to detection
  ([`5fcd915`](https://github.com/mbari-org/sdcat/commit/5fcd915f29b223bd85d33e2f1e093c3ced517603))


## v1.11.1 (2024-09-25)

### Bug Fixes

- Correct CUDA HDBSCAN fit
  ([`c6a8db3`](https://github.com/mbari-org/sdcat/commit/c6a8db33e274c356a4c5d8b26a7f380f615fb6e1))

### Continuous Integration

- Clean up poetry build before docker
  ([`c74efc8`](https://github.com/mbari-org/sdcat/commit/c74efc82f5ed4f5db4bde4ad61a06a8f5b84c597))

- Disable release
  ([`cbf6394`](https://github.com/mbari-org/sdcat/commit/cbf63942d760e52ad89ee4c04660ef8e00559c34))

- Revert to only only amd cuda build
  ([`dd133f0`](https://github.com/mbari-org/sdcat/commit/dd133f086babd4e05113499f1a74e6a852980cbf))

- Revert to only only amd cuda build
  ([`f181dfc`](https://github.com/mbari-org/sdcat/commit/f181dfc231f333b7633d36de00f46b2bf2d4b517))

- Slimmer docker add optional imports
  ([`48ae61a`](https://github.com/mbari-org/sdcat/commit/48ae61acc6eeb9df4235484d13679e7565030db8))


## v1.11.0 (2024-09-18)

### Features

- Add to cluster command option --skip-visualization since this takes some time and is not needed
  for prod ml workflows
  ([`48d1f19`](https://github.com/mbari-org/sdcat/commit/48d1f1955510ad2269def64dee9e14b96db36c0f))


## v1.10.5 (2024-09-13)

### Bug Fixes

- Minor fix to correct debug statement and correct order arg for cluster
  ([`645e979`](https://github.com/mbari-org/sdcat/commit/645e9790909496e8f3a385174270bb70f069a165))


## v1.10.4 (2024-09-05)

### Bug Fixes

- Move last cluster removal to exemplars only
  ([`96fc5db`](https://github.com/mbari-org/sdcat/commit/96fc5dbf5953bf823b929e1d41958bcfa9978698))


## v1.10.3 (2024-09-03)

### Bug Fixes

- Do not load the last cluster in exemplars
  ([`bbc3a9a`](https://github.com/mbari-org/sdcat/commit/bbc3a9a0fd73c1ef3852f19928c01c983c34d241))


## v1.10.2 (2024-08-22)

### Bug Fixes

- Trigger release with changes to __init__.py
  ([`e275d9c`](https://github.com/mbari-org/sdcat/commit/e275d9cec3094c37d3bca337ebfe2bfb8364b5d1))


## v1.10.1 (2024-08-22)

### Bug Fixes

- Trigger release with changes to pyproject.toml to update __version__
  ([`6b7f641`](https://github.com/mbari-org/sdcat/commit/6b7f641c192846167926d7c7c7920c0680eb128f))

### Chores

- Revert changes to author list
  ([`b7cd5a5`](https://github.com/mbari-org/sdcat/commit/b7cd5a588743e6173e1045d57abd26d8e20ec16f))


## v1.10.0 (2024-08-22)

### Bug Fixes

- Better handling of testing log write permissions
  ([`728b145`](https://github.com/mbari-org/sdcat/commit/728b14526b931ee895865b059a129f966b923e4d))

### Build System

- Adjust test path
  ([`83119fb`](https://github.com/mbari-org/sdcat/commit/83119fbbf479a2f09543355b2872a9f58a2f34ce))

### Chores

- Moved tests to root of project
  ([`ccf2601`](https://github.com/mbari-org/sdcat/commit/ccf2601eb292591824064b42f8ccd940a2128537))

### Documentation

- Added clustering table, transformer paper, and other minor revisions
  ([`56b4e17`](https://github.com/mbari-org/sdcat/commit/56b4e170b19d5b2d9c6984ce8c5cc7e1bc370c93))

### Features

- Added latest 30 and 18k models
  ([`673a7f6`](https://github.com/mbari-org/sdcat/commit/673a7f64a9b69f45e0edd72e381c8fe3fce01e75))

### Performance Improvements

- Removed color image normalization in preprocessing for SAHI and extract min_std and block_size for
  more flexible application across projects
  ([`d9d954a`](https://github.com/mbari-org/sdcat/commit/d9d954a24ceed9431a357340ca7e959a9111d062))


## v1.9.4 (2024-08-05)

### Bug Fixes

- Pass through device correctly for clustering
  ([`0996b8c`](https://github.com/mbari-org/sdcat/commit/0996b8cfb4adffa0f57276fb9c0b87a97a7fbcb6))

### Build System

- Install cuda enabled torch
  ([`c01259b`](https://github.com/mbari-org/sdcat/commit/c01259b1a7cea197612e446330f9d6f2ea3457ae))

### Documentation

- Changing 2 parameters to ISIIS images
  ([`417bbf4`](https://github.com/mbari-org/sdcat/commit/417bbf4e4eecbc8a61de2d8b66d3cabfb1604fbc))

### Testing

- Adjust per changing for ISIIS images
  ([`de1ba65`](https://github.com/mbari-org/sdcat/commit/de1ba65f8dff5c133bac2b03fe6b5a539619824b))

- Adjust plankton detections
  ([`25ce6bd`](https://github.com/mbari-org/sdcat/commit/25ce6bd32836fcd4eb288179e47905f87c255248))


## v1.9.3 (2024-07-31)

### Bug Fixes

- Skip over HDBSCAN for small cluster
  ([`cbf1ea3`](https://github.com/mbari-org/sdcat/commit/cbf1ea3381aef94a17eb2ae27924df2238c117a9))

### Continuous Integration

- Disable docker build until resolve issue with out of space
  ([`59e160f`](https://github.com/mbari-org/sdcat/commit/59e160f2a84f7de8e1d7f2f18e155a6a4b3b8402))


## v1.9.2 (2024-07-31)

### Bug Fixes

- Handle small cluster dataset reduction plot by switching to PCA and reduce build to only linux
  ([`16ab4de`](https://github.com/mbari-org/sdcat/commit/16ab4ded38b62665007cc5e5b1b6d1eec5865803))


## v1.9.1 (2024-07-31)

### Bug Fixes

- Correct handling of single cluster
  ([`f6ede19`](https://github.com/mbari-org/sdcat/commit/f6ede19b84b5f8808f34efa9fbf8e36479f5c8d8))


## v1.9.0 (2024-07-31)

### Continuous Integration

- Switch to entrypoint
  ([`02d4eb2`](https://github.com/mbari-org/sdcat/commit/02d4eb2aca56acbe22b452b8f690292210a8243e))

### Documentation

- Minor update to intro
  ([`ec75265`](https://github.com/mbari-org/sdcat/commit/ec7526584df6d8d63c10c75f672151fb5dc91aa3))

### Features

- Added support for arg --cluster-selection-method eof/leaf to support small clusters which work
  better with eof
  ([`6480ac2`](https://github.com/mbari-org/sdcat/commit/6480ac279d1383f2c08691998fb12ca2705bffeb))


## v1.8.4 (2024-07-29)

### Performance Improvements

- Migrated to transformers library with batch size of 8, moved some imports to only where needed for
  some speed-up, and removed unused activation maps.
  ([`c5fe725`](https://github.com/mbari-org/sdcat/commit/c5fe72523fdfb1eb7b674565e5724686a4ec65d1))


## v1.8.3 (2024-07-23)

### Bug Fixes

- Handle bad image crops (zero length). Fixes #13.
  ([`c02394c`](https://github.com/mbari-org/sdcat/commit/c02394c88e7f446d4fb1ddaea1581e5922ebf030))

### Continuous Integration

- Switch to PyPi release and skip over large build
  ([`25d3da3`](https://github.com/mbari-org/sdcat/commit/25d3da3b25c2cf8e534e7fa93a5eeb76f77672f1))

### Documentation

- Fix typo
  ([`5506bad`](https://github.com/mbari-org/sdcat/commit/5506badfa072ecedfa5ee6c1dbcd977dcb2ff90c))

- Shorten readme and update workflow to reflect latest
  ([`152624e`](https://github.com/mbari-org/sdcat/commit/152624e159c5f535ddc49132d719ab2b1452d06b))


## v1.8.2 (2024-07-22)

### Bug Fixes

- Fixes a bug in handling zero clusters which occasionally happes for very small roi collections.
  Closes #12.
  ([`0f69805`](https://github.com/mbari-org/sdcat/commit/0f698052af54af3f44a8ec2faa335e2f66077b5b))

### Continuous Integration

- Slim with two stage docker build, added explicit semantic release steps for clarity, and test pypi
  for pip install
  ([`ff2a46c`](https://github.com/mbari-org/sdcat/commit/ff2a46cdc911dae721357ff27793e6f1ef3d8c01))

### Documentation

- More detail on detection and model table
  ([`981beb2`](https://github.com/mbari-org/sdcat/commit/981beb2252dec546873cdbf136a6d24a1f2e6c4b))

- Updated cluster to include latest roi
  ([`b02464e`](https://github.com/mbari-org/sdcat/commit/b02464e99ad50db709e53a6395811ff1c7dcc754))


## v1.8.1 (2024-07-19)

### Bug Fixes

- Bump version to 1.8.0 still failing in ci
  ([`ed405e8`](https://github.com/mbari-org/sdcat/commit/ed405e8b5c892746680e1b2985c0842cb0581668))

### Continuous Integration

- Add semantic release action to write to repo
  ([`66be480`](https://github.com/mbari-org/sdcat/commit/66be480ffe72c45fc330b89757555533bc1d76d1))


## v1.8.0 (2024-07-19)

### Build System

- Slim docker images a tad, switch to generic docker user, and replace with miniconda install for
  better torch/cuda support
  ([`782848d`](https://github.com/mbari-org/sdcat/commit/782848d9ac2187b0c298e358ef642943e4e3f88f))

### Continuous Integration

- Add pip package called sdcat with poetry build and other minor corrections in comments
  ([`dbe494a`](https://github.com/mbari-org/sdcat/commit/dbe494a2112de76839ec37b079e5507908cab1ba))

- Minor change to more specific CUDA docker version in docker tag
  ([`dd2286b`](https://github.com/mbari-org/sdcat/commit/dd2286b92c496e5cbddbec73388524a42750170f))

- Simplified docker build within pypi size limits with pip build and renamed accidentally commited
  backup file
  ([`cfcb581`](https://github.com/mbari-org/sdcat/commit/cfcb58187e016287280aa2501732c38ea5d0c518))

- Updated poetry lock file with pytest
  ([`a8b62b0`](https://github.com/mbari-org/sdcat/commit/a8b62b04b75192ee50cff8dd705d395eab370b09))

### Features

- Auto default logs for different use cases: testing, local dev, and to /tmp on permission failure
  ([`0c57d45`](https://github.com/mbari-org/sdcat/commit/0c57d4552b90875796a11c1e2899a07c14f1bf53))


## v1.7.0 (2024-07-10)

### Build System

- Fix docker build
  ([`0d7ab83`](https://github.com/mbari-org/sdcat/commit/0d7ab83fc57d331fc58dabb8eb8291790b22bc3e))

- Removed unused imports and bump torch to python3.11 compatible and
  ([`57c6c2b`](https://github.com/mbari-org/sdcat/commit/57c6c2b4232730ffec878b5c3c75491f1ff42c87))

### Continuous Integration

- Remove arm cuda build
  ([`131d3bd`](https://github.com/mbari-org/sdcat/commit/131d3bdb69ddbaeaa88e38580e215f13fdb44284))

### Features

- Added tsne as an option, defaulted to false as it is generating too many clusters
  ([`8f974f3`](https://github.com/mbari-org/sdcat/commit/8f974f346b1ab1d1d1c0d16cad21494f8f5f17c7))


## v1.6.0 (2024-06-26)

### Build System

- Add sys path for convenience
  ([`d668246`](https://github.com/mbari-org/sdcat/commit/d668246fa3b6591723504d9dc26fbe9d58acc24f))

### Chores

- Updated with links visible in dockerhub
  ([`53c91b0`](https://github.com/mbari-org/sdcat/commit/53c91b077d3c195844402271861c340fc504582f))

### Continuous Integration

- Add CPU only build
  ([`8750d94`](https://github.com/mbari-org/sdcat/commit/8750d94d98ec2a5cf51b83cdf8a3220d97b6a0b7))

- Add docker hub description
  ([`9695146`](https://github.com/mbari-org/sdcat/commit/9695146473f5f0fe49122b0322d4570ae836ff4b))

- Adding in Docker CUDA12 build semantic release for amd/arm64
  ([`38aebba`](https://github.com/mbari-org/sdcat/commit/38aebbaff101b86b873b5c35208312babf7159f4))

- Consistent image tag
  ([`63165fb`](https://github.com/mbari-org/sdcat/commit/63165fb9f14f6dbff027f87041bf1676fe4d89fb))

- Correct addition of docker user
  ([`07aff58`](https://github.com/mbari-org/sdcat/commit/07aff585297d000810ceca0cfa6f45dca4307239))

- Correct order of tags in build
  ([`86f100c`](https://github.com/mbari-org/sdcat/commit/86f100c2f5f213efeba6efad0415b7824d726097))

- Correct order of tags in build and reduce dockerhub short description to < 100
  ([`86bedcf`](https://github.com/mbari-org/sdcat/commit/86bedcff3f3dd4afc8675a6dcc99430f37b8ed36))

- Correct pypi name for MulticoreTSNE
  ([`4d0702c`](https://github.com/mbari-org/sdcat/commit/4d0702c0344b016879be4c0a784328f0da67a2b8))

- Fix release version from tag
  ([`5d083df`](https://github.com/mbari-org/sdcat/commit/5d083df54902dba4abb91400732ccd82715fdd5a))

- Install tsne separately and clean
  ([`17f5a6e`](https://github.com/mbari-org/sdcat/commit/17f5a6ebb0c40a7b172b3461e808b766396f20a5))

- Pin build for torch
  ([`3f32cb6`](https://github.com/mbari-org/sdcat/commit/3f32cb6e4264c838f096105c99dc0f8f1db7157d))

- Remove pass through of version to dockerhub
  ([`a43cba7`](https://github.com/mbari-org/sdcat/commit/a43cba7ac1c9599fa48230e0bc8789b49bd3f0cd))

### Documentation

- Minor update to correct log format
  ([`6cf6453`](https://github.com/mbari-org/sdcat/commit/6cf64537ca551a56f1bd6bba2d47d42cc4dcb7e7))

- Updated workflow diagram to reflect tSNE
  ([`f1cecf8`](https://github.com/mbari-org/sdcat/commit/f1cecf866fe7e9b41e71dfb198910ef44a30bc65))

### Features

- Added export of the examplars, handle small clustering input by bypassing tSNE which fails, and
  make dino_vits8 the default
  ([`2306316`](https://github.com/mbari-org/sdcat/commit/2306316f4218244c8fc340261140de9d4af0dc8b))


## v1.5.0 (2024-06-03)


## v1.4.1 (2024-06-03)

### Bug Fixes

- Conditional import of multicore tsne
  ([`76ec895`](https://github.com/mbari-org/sdcat/commit/76ec89589319664a98c8523312a859aa3475b1c2))

### Chores

- Minor change to remove unused import
  ([`d87b1de`](https://github.com/mbari-org/sdcat/commit/d87b1de2a3b07f02dbfde7611a8a1506d9a027e5))


## v1.4.0 (2024-05-30)

### Bug Fixes

- Added image width/height and fixed multiproc to square
  ([`b7e2f21`](https://github.com/mbari-org/sdcat/commit/b7e2f21e136abc80d1e88133928cc68d2f30eebf))

- Check for det columns
  ([`ff9a29d`](https://github.com/mbari-org/sdcat/commit/ff9a29dbe6c81cee0e0038401b9a99309c2baecd))

- Correct PIL image path
  ([`c0da6b6`](https://github.com/mbari-org/sdcat/commit/c0da6b67a79c636606a207571873ebb6bf8ebb0b))

- Image size in int not float needed for resize
  ([`cf3ebd3`](https://github.com/mbari-org/sdcat/commit/cf3ebd37436ddd7e160a0b7b0ea14ac29ba64fe3))

- Image size in int not float needed for resize
  ([`18fc005`](https://github.com/mbari-org/sdcat/commit/18fc005a5fc39a14327e47b6d0225ea6a79317ab))

- Path to string
  ([`977ef32`](https://github.com/mbari-org/sdcat/commit/977ef32902f82d4816716118a34b36a8978210a6))

- Removed unused args for start/end frame
  ([`f91ac03`](https://github.com/mbari-org/sdcat/commit/f91ac03dcd1a3ff735b982cc2b0647eb8dc32938))

- Roi_dir needs to support lists
  ([`3808bf9`](https://github.com/mbari-org/sdcat/commit/3808bf9531ed27d55e3dbf1e4ebd9cafc9c6454a))

### Chores

- All options with dashes instead of underscores
  ([`8dbc5ad`](https://github.com/mbari-org/sdcat/commit/8dbc5ad6b0145bf56f690779ab5790d537c3d2e7))

- Remove area for ROI since approximations are probably a bad idea
  ([`a54d88f`](https://github.com/mbari-org/sdcat/commit/a54d88fa4463c8937288565d06a85f274be0c324))

### Features

- Added square black pad resize for roi
  ([`473aa34`](https://github.com/mbari-org/sdcat/commit/473aa34c6e5efd02bb9983b55f358d3afea89ed8))

- Added tsne reduction
  ([`d0f7647`](https://github.com/mbari-org/sdcat/commit/d0f764735d1f0dc91f8090b92367434724b34376))

- Initial commit of new option to cluster roi only
  ([`f08053c`](https://github.com/mbari-org/sdcat/commit/f08053c27d6c7c5750416165ed20a40c542f23cd))

### Performance Improvements

- Switch to multicore tSNE
  ([`9ec0ca9`](https://github.com/mbari-org/sdcat/commit/9ec0ca9107f9c9e5a6a8ea90c24f711302b1c392))

### Refactoring

- Minor renaming for clarity
  ([`6bb998a`](https://github.com/mbari-org/sdcat/commit/6bb998af4245b21eb90195465e382788e612e988))


## v1.3.0 (2024-05-01)

### Features

- Added pass through of slicing overlap and postprocess_match_metric
  ([`f3b14bb`](https://github.com/mbari-org/sdcat/commit/f3b14bbf953a5eb18955339c36279cdabd761374))

### Testing

- Adjust detection count per NMS output fix
  ([`9651c7a`](https://github.com/mbari-org/sdcat/commit/9651c7a4c8166a6fe3894d620b1f42726474d592))


## v1.2.2 (2024-05-01)

### Bug Fixes

- Allow for override of detect params with config.ini;correct save NMS output and detect single
  image
  ([`4ab8780`](https://github.com/mbari-org/sdcat/commit/4ab878033ac423918c7a1eb1eb2416a435779276))

- More detail on cluster args, allow for override with config.ini, and set CUDA_VISIBLE_DEVICES in
  case not set
  ([`40a96ef`](https://github.com/mbari-org/sdcat/commit/40a96efcfd4fd9f7cf2fb8aecb11970d5c227671))

### Refactoring

- Minor refactor of cuda device init
  ([`0f29e68`](https://github.com/mbari-org/sdcat/commit/0f29e688daac36353eb66e85ecf6a4cdd2d61574))


## v1.2.1 (2024-04-29)

### Bug Fixes

- Sorted order needed to make start/end filtering work
  ([`9caea73`](https://github.com/mbari-org/sdcat/commit/9caea73bc3263077e057886eee2ce2a95e1e139d))


## v1.2.0 (2024-04-29)

### Documentation

- Erasing some prints in the code
  ([`622c119`](https://github.com/mbari-org/sdcat/commit/622c1191a9055e06fa2bd818ae5ba407cca607bb))

### Features

- Added support for specifying start/ending file names to support skipping image blocks
  ([`f344865`](https://github.com/mbari-org/sdcat/commit/f344865812ede812a4bff7b15fd24f71b0d8c418))


## v1.1.0 (2024-04-23)

### Features

- New features for clustering
  ([`b74d1b1`](https://github.com/mbari-org/sdcat/commit/b74d1b1e3025db401d996ee49c8e79b9d26c8998))


## v1.0.9 (2024-04-22)

### Bug Fixes

- Deleting the os version
  ([`e63bb73`](https://github.com/mbari-org/sdcat/commit/e63bb738e9d8e6c08b1c9913788182b681b550d6))


## v1.0.8 (2024-04-22)

### Bug Fixes

- Adding .tiff images and vizualization tools with tqdm
  ([`d0b99e2`](https://github.com/mbari-org/sdcat/commit/d0b99e20e1c1a27eee3ba210d0d6724c4e509a1d))


## v1.0.7 (2024-03-11)

### Bug Fixes

- Added missing config.ini default
  ([`ae8b726`](https://github.com/mbari-org/sdcat/commit/ae8b726fedb88459add154ca4add99948f79cc50))

### Build System

- Bumped version
  ([`8ba97a8`](https://github.com/mbari-org/sdcat/commit/8ba97a879c94d643543f8c9ff3fa14edcbc31d73))

### Testing

- Revised PYTHONPATH and switch to codfish semantic release
  ([`8eb6d42`](https://github.com/mbari-org/sdcat/commit/8eb6d420b906331baf983b639926d74281189ac5))


## v1.0.6 (2024-03-05)

### Bug Fixes

- Correct some bugs calling libaries
  ([`d69171b`](https://github.com/mbari-org/sdcat/commit/d69171ba9b9caafa218439df810f9b73ace99023))

### Build System

- Bumped version
  ([`404036b`](https://github.com/mbari-org/sdcat/commit/404036bc673cf541c7f7e45d96d8bbd13a1399b6))

### Chores

- **release**: 1.0.6 [skip ci]
  ([`ff2c3bd`](https://github.com/mbari-org/sdcat/commit/ff2c3bdf286f2bd6442c737cca825704c81a9779))

## [1.0.6](https://github.com/mbari-org/sdcat/compare/v1.0.5...v1.0.6) (2024-03-05)

### Bug Fixes

* correct some bugs calling libaries
  ([d69171b](https://github.com/mbari-org/sdcat/commit/d69171ba9b9caafa218439df810f9b73ace99023))


## v1.0.5 (2024-03-05)

### Bug Fixes

- Correct some bugs calling libaries
  ([`0668bda`](https://github.com/mbari-org/sdcat/commit/0668bda21986e6a291f89e0c4a67305d223047ad))

### Build System

- Bumped version
  ([`b7a6740`](https://github.com/mbari-org/sdcat/commit/b7a674041376a38c19ca58436d089312342df8f0))

### Chores

- **release**: 1.0.5 [skip ci]
  ([`1258290`](https://github.com/mbari-org/sdcat/commit/12582901b051fb6407b90add813cc9ae7bb82e94))

## [1.0.5](https://github.com/mbari-org/sdcat/compare/v1.0.4...v1.0.5) (2024-03-05)

### Bug Fixes

* correct some bugs calling libaries
  ([0668bda](https://github.com/mbari-org/sdcat/commit/0668bda21986e6a291f89e0c4a67305d223047ad))


## v1.0.4 (2024-03-05)

### Bug Fixes

- Correct more imports
  ([`107a470`](https://github.com/mbari-org/sdcat/commit/107a470a12abaac020a1bcb7e9749581e882eea8))

### Build System

- Bumped version
  ([`f1c17d3`](https://github.com/mbari-org/sdcat/commit/f1c17d37b1efecedcd01e1ecef55e9e73e8e96da))

### Chores

- **release**: 1.0.4 [skip ci]
  ([`5e76d32`](https://github.com/mbari-org/sdcat/commit/5e76d32af346cb01b2b7e5c5972923c1b6b4b5bc))

## [1.0.4](https://github.com/mbari-org/sdcat/compare/v1.0.3...v1.0.4) (2024-03-05)

### Bug Fixes

* correct more imports
  ([107a470](https://github.com/mbari-org/sdcat/commit/107a470a12abaac020a1bcb7e9749581e882eea8))


## v1.0.3 (2024-03-05)

### Bug Fixes

- Correct import paths
  ([`6ca36e1`](https://github.com/mbari-org/sdcat/commit/6ca36e13a321cf8a4460a2598cf8f1e6cf73c2c3))

### Build System

- Bumped version
  ([`af998f4`](https://github.com/mbari-org/sdcat/commit/af998f4bc730660a537c7d3ba818b10585771e27))

- Minor rev to pytest to try to fix PYTHONPATH
  ([`ed5dec7`](https://github.com/mbari-org/sdcat/commit/ed5dec79fd4d973d654a1b06e5ebe6f242e8aa02))

### Chores

- Correct comment
  ([`1004d95`](https://github.com/mbari-org/sdcat/commit/1004d9588cf46e4864faa9444d077f7df0179de1))

- Correct comment on multiproc
  ([`a826de9`](https://github.com/mbari-org/sdcat/commit/a826de930ab851a3ca77cff36f6cf4a99170ba7d))

- Removed unused code
  ([`ccb95e3`](https://github.com/mbari-org/sdcat/commit/ccb95e33f2f4657eaf7cbbf237765d1a8bb077dc))

- **release**: 1.0.3 [skip ci]
  ([`3dce93a`](https://github.com/mbari-org/sdcat/commit/3dce93a216cde0790f77fa267f1e005b28358e39))

## [1.0.3](https://github.com/mbari-org/sdcat/compare/v1.0.2...v1.0.3) (2024-03-05)

### Bug Fixes

* correct import paths
  ([6ca36e1](https://github.com/mbari-org/sdcat/commit/6ca36e13a321cf8a4460a2598cf8f1e6cf73c2c3))

### Documentation

- Added sahi example and constrain python to 3.9-3.11
  ([`240d630`](https://github.com/mbari-org/sdcat/commit/240d6302a393ef89b7202257c16335c3d71c28f8))

### Testing

- Add missing PYTHONPATH
  ([`4371400`](https://github.com/mbari-org/sdcat/commit/437140021ff3295a3142d94f9ada2f400af2c213))

- Fixed path to main and allow pytest to be run manually
  ([`f586fd4`](https://github.com/mbari-org/sdcat/commit/f586fd4903a8bb9f470047e944308920a955c628))

- Minor typo fix in pytest.yml
  ([`a108717`](https://github.com/mbari-org/sdcat/commit/a108717b7e93fc8efeb27a8e175c87f4d7be23c3))


## v1.0.2 (2024-02-17)

### Bug Fixes

- Pass through config agnostic flat
  ([`7b5f940`](https://github.com/mbari-org/sdcat/commit/7b5f940bae4a10591f0c7c14b87f2a60b14d69c2))

- Remove unused arguments
  ([`ad06613`](https://github.com/mbari-org/sdcat/commit/ad066138a233b0bcb79fda18c50b54da658fe545))

### Build System

- Bumped version
  ([`721b1b3`](https://github.com/mbari-org/sdcat/commit/721b1b30599cdde6c285fdded2564c844efdf7e4))

### Chores

- **release**: 1.0.2 [skip ci]
  ([`3d9488d`](https://github.com/mbari-org/sdcat/commit/3d9488dc76a9749f6e81d5be897ef9cb5441a172))

## [1.0.2](https://github.com/mbari-org/sdcat/compare/v1.0.1...v1.0.2) (2024-02-17)

### Bug Fixes

* pass through config agnostic flat
  ([7b5f940](https://github.com/mbari-org/sdcat/commit/7b5f940bae4a10591f0c7c14b87f2a60b14d69c2)) *
  remove unused arguments
  ([ad06613](https://github.com/mbari-org/sdcat/commit/ad066138a233b0bcb79fda18c50b54da658fe545))

### Documentation

- Added spec removal ref
  ([`987ad8a`](https://github.com/mbari-org/sdcat/commit/987ad8a2ced3506995b2c0d68920ab4e33194939))


## v1.0.1 (2024-02-15)

### Bug Fixes

- Added missing outer block for sahi pool
  ([`91f88ef`](https://github.com/mbari-org/sdcat/commit/91f88efdffde043c365b4b8871c4d6eae0e64f25))

### Build System

- Bumped version
  ([`5508b33`](https://github.com/mbari-org/sdcat/commit/5508b337df46bb4387483777e28392d757e0202f))

### Chores

- **release**: 1.0.1 [skip ci]
  ([`f9f8639`](https://github.com/mbari-org/sdcat/commit/f9f8639c1d0cfa254cc3eb98d91225aad69defe6))

## [1.0.1](https://github.com/mbari-org/sdcat/compare/v1.0.0...v1.0.1) (2024-02-15)

### Bug Fixes

* added missing outer block for sahi pool
  ([91f88ef](https://github.com/mbari-org/sdcat/commit/91f88efdffde043c365b4b8871c4d6eae0e64f25))

### Documentation

- Added ref to HDBSCAN paper
  ([`4a4edbc`](https://github.com/mbari-org/sdcat/commit/4a4edbca2d910dbdf7919716375f896d7bfde3da))

- Minor reformatting
  ([`7afeca2`](https://github.com/mbari-org/sdcat/commit/7afeca2b56845dee95b882f777fe3a449ffd5477))


## v1.0.0 (2024-02-14)

### Build System

- Added release and pytest
  ([`1bb6f28`](https://github.com/mbari-org/sdcat/commit/1bb6f28de4c6d017b2e617fe50a82d557675586e))

### Chores

- **release**: 1.0.0 [skip ci]
  ([`0c32f9b`](https://github.com/mbari-org/sdcat/commit/0c32f9b208ce7a0a6864b657b948b32a99422ca8))

# 1.0.0 (2024-02-14)

### Features

* initial commit
  ([148f17f](https://github.com/mbari-org/sdcat/commit/148f17f4a1e5af2a03380de964ff1140052d53b8))

### Documentation

- Added example images and cluster workflow diagram
  ([`b228480`](https://github.com/mbari-org/sdcat/commit/b228480926cfce24d3e8575c786d1b952e8be256))

- Correct links to image and reset version history
  ([`2c98876`](https://github.com/mbari-org/sdcat/commit/2c9887661b6d6fa0a562c470e6282a00f54d5d3a))

### Features

- Initial commit
  ([`148f17f`](https://github.com/mbari-org/sdcat/commit/148f17f4a1e5af2a03380de964ff1140052d53b8))
