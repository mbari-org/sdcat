# CHANGELOG

## v1.10.2 (2024-08-22)

### Fix

* fix: trigger release with changes to __init__.py ([`e275d9c`](https://github.com/mbari-org/sdcat/commit/e275d9cec3094c37d3bca337ebfe2bfb8364b5d1))

## v1.10.1 (2024-08-22)

### Chore

* chore: revert changes to author list ([`b7cd5a5`](https://github.com/mbari-org/sdcat/commit/b7cd5a588743e6173e1045d57abd26d8e20ec16f))

### Fix

* fix: trigger release with changes to pyproject.toml to update __version__ ([`6b7f641`](https://github.com/mbari-org/sdcat/commit/6b7f641c192846167926d7c7c7920c0680eb128f))

## v1.10.0 (2024-08-22)

### Build

* build: adjust test path ([`83119fb`](https://github.com/mbari-org/sdcat/commit/83119fbbf479a2f09543355b2872a9f58a2f34ce))

### Chore

* chore: moved tests to root of project ([`ccf2601`](https://github.com/mbari-org/sdcat/commit/ccf2601eb292591824064b42f8ccd940a2128537))

### Documentation

* docs: added clustering table, transformer paper, and other minor revisions ([`56b4e17`](https://github.com/mbari-org/sdcat/commit/56b4e170b19d5b2d9c6984ce8c5cc7e1bc370c93))

### Feature

* feat: added latest 30 and 18k models ([`673a7f6`](https://github.com/mbari-org/sdcat/commit/673a7f64a9b69f45e0edd72e381c8fe3fce01e75))

### Fix

* fix: better handling of testing log write permissions ([`728b145`](https://github.com/mbari-org/sdcat/commit/728b14526b931ee895865b059a129f966b923e4d))

### Performance

* perf: removed color image normalization in preprocessing for SAHI and extract min_std and block_size for more flexible application across projects ([`d9d954a`](https://github.com/mbari-org/sdcat/commit/d9d954a24ceed9431a357340ca7e959a9111d062))

## v1.9.4 (2024-08-05)

### Build

* build: install cuda enabled torch ([`c01259b`](https://github.com/mbari-org/sdcat/commit/c01259b1a7cea197612e446330f9d6f2ea3457ae))

### Documentation

* docs: changing 2 parameters to ISIIS images ([`417bbf4`](https://github.com/mbari-org/sdcat/commit/417bbf4e4eecbc8a61de2d8b66d3cabfb1604fbc))

### Fix

* fix: pass through device correctly for clustering ([`0996b8c`](https://github.com/mbari-org/sdcat/commit/0996b8cfb4adffa0f57276fb9c0b87a97a7fbcb6))

### Test

* test: adjust plankton detections ([`25ce6bd`](https://github.com/mbari-org/sdcat/commit/25ce6bd32836fcd4eb288179e47905f87c255248))

* test: adjust per changing for  ISIIS images ([`de1ba65`](https://github.com/mbari-org/sdcat/commit/de1ba65f8dff5c133bac2b03fe6b5a539619824b))

## v1.9.3 (2024-07-31)

### Ci

* ci: disable docker build until resolve issue with out of space ([`59e160f`](https://github.com/mbari-org/sdcat/commit/59e160f2a84f7de8e1d7f2f18e155a6a4b3b8402))

### Fix

* fix: skip over HDBSCAN for small cluster ([`cbf1ea3`](https://github.com/mbari-org/sdcat/commit/cbf1ea3381aef94a17eb2ae27924df2238c117a9))

## v1.9.2 (2024-07-31)

### Fix

* fix: handle small cluster dataset reduction plot by switching to PCA and reduce build to only linux ([`16ab4de`](https://github.com/mbari-org/sdcat/commit/16ab4ded38b62665007cc5e5b1b6d1eec5865803))

## v1.9.1 (2024-07-31)

### Fix

* fix: correct handling of single cluster ([`f6ede19`](https://github.com/mbari-org/sdcat/commit/f6ede19b84b5f8808f34efa9fbf8e36479f5c8d8))

## v1.9.0 (2024-07-31)

### Ci

* ci: switch to entrypoint ([`02d4eb2`](https://github.com/mbari-org/sdcat/commit/02d4eb2aca56acbe22b452b8f690292210a8243e))

### Documentation

* docs: minor update to intro ([`ec75265`](https://github.com/mbari-org/sdcat/commit/ec7526584df6d8d63c10c75f672151fb5dc91aa3))

### Feature

* feat: added support for arg --cluster-selection-method eof/leaf to support small clusters which work better with eof ([`6480ac2`](https://github.com/mbari-org/sdcat/commit/6480ac279d1383f2c08691998fb12ca2705bffeb))

## v1.8.4 (2024-07-29)

### Performance

* perf: migrated to transformers library with batch size of 8, moved some imports to only where needed for some speed-up, and removed unused activation maps. ([`c5fe725`](https://github.com/mbari-org/sdcat/commit/c5fe72523fdfb1eb7b674565e5724686a4ec65d1))

### Unknown

* Merge pull request #14 from mbari-org/vitbatch

perf: transformer batching ([`427931e`](https://github.com/mbari-org/sdcat/commit/427931e3f21e744e887489051ca02d1f8f39f894))

## v1.8.3 (2024-07-23)

### Ci

* ci: switch to PyPi release and skip over large build ([`25d3da3`](https://github.com/mbari-org/sdcat/commit/25d3da3b25c2cf8e534e7fa93a5eeb76f77672f1))

### Documentation

* docs: fix typo ([`5506bad`](https://github.com/mbari-org/sdcat/commit/5506badfa072ecedfa5ee6c1dbcd977dcb2ff90c))

* docs: shorten readme and update workflow to reflect latest ([`152624e`](https://github.com/mbari-org/sdcat/commit/152624e159c5f535ddc49132d719ab2b1452d06b))

### Fix

* fix: handle bad image crops (zero length). Fixes #13. ([`c02394c`](https://github.com/mbari-org/sdcat/commit/c02394c88e7f446d4fb1ddaea1581e5922ebf030))

## v1.8.2 (2024-07-22)

### Ci

* ci: slim with two stage docker build, added explicit semantic release steps for clarity, and test pypi for pip install ([`ff2a46c`](https://github.com/mbari-org/sdcat/commit/ff2a46cdc911dae721357ff27793e6f1ef3d8c01))

### Documentation

* docs: updated cluster to include latest roi ([`b02464e`](https://github.com/mbari-org/sdcat/commit/b02464e99ad50db709e53a6395811ff1c7dcc754))

* docs: more detail on detection and model table ([`981beb2`](https://github.com/mbari-org/sdcat/commit/981beb2252dec546873cdbf136a6d24a1f2e6c4b))

### Fix

* fix: fixes a bug in handling zero clusters which occasionally happes for very small roi collections. Closes #12. ([`0f69805`](https://github.com/mbari-org/sdcat/commit/0f698052af54af3f44a8ec2faa335e2f66077b5b))

## v1.8.1 (2024-07-19)

### Ci

* ci: add semantic release action to write to repo ([`66be480`](https://github.com/mbari-org/sdcat/commit/66be480ffe72c45fc330b89757555533bc1d76d1))

### Fix

* fix: bump version to 1.8.0 still failing in ci ([`ed405e8`](https://github.com/mbari-org/sdcat/commit/ed405e8b5c892746680e1b2985c0842cb0581668))

## v1.8.0 (2024-07-19)

### Build

* build: slim docker images a tad, switch to generic docker user, and replace with miniconda install for better torch/cuda support ([`782848d`](https://github.com/mbari-org/sdcat/commit/782848d9ac2187b0c298e358ef642943e4e3f88f))

### Ci

* ci: simplified docker build within pypi size limits with pip build and renamed accidentally commited backup file ([`cfcb581`](https://github.com/mbari-org/sdcat/commit/cfcb58187e016287280aa2501732c38ea5d0c518))

* ci: minor change to more specific CUDA docker version in docker tag ([`dd2286b`](https://github.com/mbari-org/sdcat/commit/dd2286b92c496e5cbddbec73388524a42750170f))

* ci: updated poetry lock file with pytest ([`a8b62b0`](https://github.com/mbari-org/sdcat/commit/a8b62b04b75192ee50cff8dd705d395eab370b09))

* ci: add pip package called sdcat with poetry build and other minor corrections in comments ([`dbe494a`](https://github.com/mbari-org/sdcat/commit/dbe494a2112de76839ec37b079e5507908cab1ba))

### Feature

* feat: auto default logs for different use cases: testing, local dev, and to /tmp on permission failure ([`0c57d45`](https://github.com/mbari-org/sdcat/commit/0c57d4552b90875796a11c1e2899a07c14f1bf53))

### Unknown

* Merge pull request #11 from mbari-org/pipbuild

ci: add pip package called sdcat with poetry build and other minor co… ([`e220587`](https://github.com/mbari-org/sdcat/commit/e2205871a7fdafc2ca9dd2d769f6d43a73930c1d))

## v1.7.0 (2024-07-10)

### Build

* build: removed unused imports and bump torch to python3.11 compatible and ([`57c6c2b`](https://github.com/mbari-org/sdcat/commit/57c6c2b4232730ffec878b5c3c75491f1ff42c87))

* build: fix docker build ([`0d7ab83`](https://github.com/mbari-org/sdcat/commit/0d7ab83fc57d331fc58dabb8eb8291790b22bc3e))

### Ci

* ci: remove arm cuda build ([`131d3bd`](https://github.com/mbari-org/sdcat/commit/131d3bdb69ddbaeaa88e38580e215f13fdb44284))

### Feature

* feat: added tsne as an option, defaulted to false as it is generating too many clusters ([`8f974f3`](https://github.com/mbari-org/sdcat/commit/8f974f346b1ab1d1d1c0d16cad21494f8f5f17c7))

## v1.6.0 (2024-06-26)

### Build

* build: add sys path for convenience ([`d668246`](https://github.com/mbari-org/sdcat/commit/d668246fa3b6591723504d9dc26fbe9d58acc24f))

### Chore

* chore: updated with links visible in dockerhub ([`53c91b0`](https://github.com/mbari-org/sdcat/commit/53c91b077d3c195844402271861c340fc504582f))

### Ci

* ci: fix release version from tag ([`5d083df`](https://github.com/mbari-org/sdcat/commit/5d083df54902dba4abb91400732ccd82715fdd5a))

* ci: remove pass through of version to dockerhub ([`a43cba7`](https://github.com/mbari-org/sdcat/commit/a43cba7ac1c9599fa48230e0bc8789b49bd3f0cd))

* ci: correct order of tags in build and reduce dockerhub short description to &lt; 100 ([`86bedcf`](https://github.com/mbari-org/sdcat/commit/86bedcff3f3dd4afc8675a6dcc99430f37b8ed36))

* ci: correct order of tags in build ([`86f100c`](https://github.com/mbari-org/sdcat/commit/86f100c2f5f213efeba6efad0415b7824d726097))

* ci: consistent image tag ([`63165fb`](https://github.com/mbari-org/sdcat/commit/63165fb9f14f6dbff027f87041bf1676fe4d89fb))

* ci: install tsne separately and clean ([`17f5a6e`](https://github.com/mbari-org/sdcat/commit/17f5a6ebb0c40a7b172b3461e808b766396f20a5))

* ci: pin build for torch ([`3f32cb6`](https://github.com/mbari-org/sdcat/commit/3f32cb6e4264c838f096105c99dc0f8f1db7157d))

* ci: add docker hub description ([`9695146`](https://github.com/mbari-org/sdcat/commit/9695146473f5f0fe49122b0322d4570ae836ff4b))

* ci: add CPU only build ([`8750d94`](https://github.com/mbari-org/sdcat/commit/8750d94d98ec2a5cf51b83cdf8a3220d97b6a0b7))

* ci: correct pypi name for MulticoreTSNE ([`4d0702c`](https://github.com/mbari-org/sdcat/commit/4d0702c0344b016879be4c0a784328f0da67a2b8))

* ci: correct addition of docker user ([`07aff58`](https://github.com/mbari-org/sdcat/commit/07aff585297d000810ceca0cfa6f45dca4307239))

* ci: adding in Docker CUDA12 build semantic release for amd/arm64 ([`38aebba`](https://github.com/mbari-org/sdcat/commit/38aebbaff101b86b873b5c35208312babf7159f4))

### Documentation

* docs: minor update to correct log format ([`6cf6453`](https://github.com/mbari-org/sdcat/commit/6cf64537ca551a56f1bd6bba2d47d42cc4dcb7e7))

* docs: updated workflow diagram to reflect tSNE ([`f1cecf8`](https://github.com/mbari-org/sdcat/commit/f1cecf866fe7e9b41e71dfb198910ef44a30bc65))

### Feature

* feat: added export of the examplars, handle small clustering input by bypassing tSNE which fails, and make dino_vits8 the default ([`2306316`](https://github.com/mbari-org/sdcat/commit/2306316f4218244c8fc340261140de9d4af0dc8b))

### Unknown

* Merge pull request #8 from mbari-org/docker

Docker automated builds ([`fd1e359`](https://github.com/mbari-org/sdcat/commit/fd1e359f7b6f21a0a50597d274e6567825afb605))

## v1.5.0 (2024-06-03)

### Unknown

* Merge pull request #7 from mbari-org/roicluster

RoiCluster ([`a9476c2`](https://github.com/mbari-org/sdcat/commit/a9476c20fea7f0c483c9de52fca1359fdd28ba7a))

* Merge branch &#39;main&#39; into roicluster ([`bd1696c`](https://github.com/mbari-org/sdcat/commit/bd1696cf9534e3263608b986084f12b765c3432e))

## v1.4.1 (2024-06-03)

### Chore

* chore: minor change to remove unused import ([`d87b1de`](https://github.com/mbari-org/sdcat/commit/d87b1de2a3b07f02dbfde7611a8a1506d9a027e5))

### Fix

* fix: conditional import of multicore tsne ([`76ec895`](https://github.com/mbari-org/sdcat/commit/76ec89589319664a98c8523312a859aa3475b1c2))

## v1.4.0 (2024-05-30)

### Chore

* chore: remove area for ROI since approximations are probably a bad idea ([`a54d88f`](https://github.com/mbari-org/sdcat/commit/a54d88fa4463c8937288565d06a85f274be0c324))

* chore: all options with dashes instead of underscores ([`8dbc5ad`](https://github.com/mbari-org/sdcat/commit/8dbc5ad6b0145bf56f690779ab5790d537c3d2e7))

### Feature

* feat: added tsne reduction ([`d0f7647`](https://github.com/mbari-org/sdcat/commit/d0f764735d1f0dc91f8090b92367434724b34376))

* feat: added square black pad resize for roi ([`473aa34`](https://github.com/mbari-org/sdcat/commit/473aa34c6e5efd02bb9983b55f358d3afea89ed8))

* feat: initial commit of new option to cluster roi only ([`f08053c`](https://github.com/mbari-org/sdcat/commit/f08053c27d6c7c5750416165ed20a40c542f23cd))

### Fix

* fix: check for det columns ([`ff9a29d`](https://github.com/mbari-org/sdcat/commit/ff9a29dbe6c81cee0e0038401b9a99309c2baecd))

* fix: image size in int not float needed for resize ([`cf3ebd3`](https://github.com/mbari-org/sdcat/commit/cf3ebd37436ddd7e160a0b7b0ea14ac29ba64fe3))

* fix: image size in int not float needed for resize ([`18fc005`](https://github.com/mbari-org/sdcat/commit/18fc005a5fc39a14327e47b6d0225ea6a79317ab))

* fix: correct PIL image path ([`c0da6b6`](https://github.com/mbari-org/sdcat/commit/c0da6b67a79c636606a207571873ebb6bf8ebb0b))

* fix: added image width/height and fixed multiproc to square ([`b7e2f21`](https://github.com/mbari-org/sdcat/commit/b7e2f21e136abc80d1e88133928cc68d2f30eebf))

* fix: path to string ([`977ef32`](https://github.com/mbari-org/sdcat/commit/977ef32902f82d4816716118a34b36a8978210a6))

* fix: roi_dir needs to support lists ([`3808bf9`](https://github.com/mbari-org/sdcat/commit/3808bf9531ed27d55e3dbf1e4ebd9cafc9c6454a))

* fix: removed unused args for start/end frame ([`f91ac03`](https://github.com/mbari-org/sdcat/commit/f91ac03dcd1a3ff735b982cc2b0647eb8dc32938))

### Performance

* perf: switch to multicore tSNE ([`9ec0ca9`](https://github.com/mbari-org/sdcat/commit/9ec0ca9107f9c9e5a6a8ea90c24f711302b1c392))

### Refactor

* refactor: minor renaming for clarity ([`6bb998a`](https://github.com/mbari-org/sdcat/commit/6bb998af4245b21eb90195465e382788e612e988))

### Unknown

* Merge pull request #6 from mbari-org/tsne

Tsne ([`48ca9f8`](https://github.com/mbari-org/sdcat/commit/48ca9f82ac9fee2fb54aefa435a1d1ac2e2027a3))

## v1.3.0 (2024-05-01)

### Feature

* feat: added pass through of slicing overlap and postprocess_match_metric ([`f3b14bb`](https://github.com/mbari-org/sdcat/commit/f3b14bbf953a5eb18955339c36279cdabd761374))

### Test

* test: adjust detection count per NMS output fix ([`9651c7a`](https://github.com/mbari-org/sdcat/commit/9651c7a4c8166a6fe3894d620b1f42726474d592))

## v1.2.2 (2024-05-01)

### Fix

* fix: allow for override of detect params with config.ini;correct save NMS output and detect single image ([`4ab8780`](https://github.com/mbari-org/sdcat/commit/4ab878033ac423918c7a1eb1eb2416a435779276))

* fix: more detail on cluster args, allow for override with config.ini, and set CUDA_VISIBLE_DEVICES in case not set ([`40a96ef`](https://github.com/mbari-org/sdcat/commit/40a96efcfd4fd9f7cf2fb8aecb11970d5c227671))

### Refactor

* refactor: minor refactor of cuda device init ([`0f29e68`](https://github.com/mbari-org/sdcat/commit/0f29e688daac36353eb66e85ecf6a4cdd2d61574))

## v1.2.1 (2024-04-29)

### Fix

* fix: sorted order needed to make start/end filtering work ([`9caea73`](https://github.com/mbari-org/sdcat/commit/9caea73bc3263077e057886eee2ce2a95e1e139d))

## v1.2.0 (2024-04-29)

### Documentation

* docs: Erasing some prints in the code ([`622c119`](https://github.com/mbari-org/sdcat/commit/622c1191a9055e06fa2bd818ae5ba407cca607bb))

### Feature

* feat: added support for specifying start/ending file names to support skipping image blocks ([`f344865`](https://github.com/mbari-org/sdcat/commit/f344865812ede812a4bff7b15fd24f71b0d8c418))

## v1.1.0 (2024-04-23)

### Feature

* feat: New features for clustering ([`b74d1b1`](https://github.com/mbari-org/sdcat/commit/b74d1b1e3025db401d996ee49c8e79b9d26c8998))

## v1.0.9 (2024-04-22)

### Fix

* fix: deleting the os version ([`e63bb73`](https://github.com/mbari-org/sdcat/commit/e63bb738e9d8e6c08b1c9913788182b681b550d6))

## v1.0.8 (2024-04-22)

### Fix

* fix: Adding .tiff images and vizualization tools with tqdm ([`d0b99e2`](https://github.com/mbari-org/sdcat/commit/d0b99e20e1c1a27eee3ba210d0d6724c4e509a1d))

### Unknown

* Including clahe in the commands ([`ea92e2f`](https://github.com/mbari-org/sdcat/commit/ea92e2fde01e646bc3506890a8279eb881fbd3c1))

## v1.0.7 (2024-03-11)

### Build

* build: bumped version ([`8ba97a8`](https://github.com/mbari-org/sdcat/commit/8ba97a879c94d643543f8c9ff3fa14edcbc31d73))

### Fix

* fix: added missing config.ini default ([`ae8b726`](https://github.com/mbari-org/sdcat/commit/ae8b726fedb88459add154ca4add99948f79cc50))

### Test

* test: revised PYTHONPATH and switch to codfish semantic release ([`8eb6d42`](https://github.com/mbari-org/sdcat/commit/8eb6d420b906331baf983b639926d74281189ac5))

### Unknown

* Testing #2 ([`0d8b2f4`](https://github.com/mbari-org/sdcat/commit/0d8b2f49fecee4370d5d74dfb74171216591f5e7))

## v1.0.6 (2024-03-05)

### Build

* build: bumped version ([`404036b`](https://github.com/mbari-org/sdcat/commit/404036bc673cf541c7f7e45d96d8bbd13a1399b6))

### Chore

* chore(release): 1.0.6 [skip ci]

## [1.0.6](https://github.com/mbari-org/sdcat/compare/v1.0.5...v1.0.6) (2024-03-05)

### Bug Fixes

* correct some bugs calling libaries ([d69171b](https://github.com/mbari-org/sdcat/commit/d69171ba9b9caafa218439df810f9b73ace99023)) ([`ff2c3bd`](https://github.com/mbari-org/sdcat/commit/ff2c3bdf286f2bd6442c737cca825704c81a9779))

### Fix

* fix: correct some bugs calling libaries ([`d69171b`](https://github.com/mbari-org/sdcat/commit/d69171ba9b9caafa218439df810f9b73ace99023))

## v1.0.5 (2024-03-05)

### Build

* build: bumped version ([`b7a6740`](https://github.com/mbari-org/sdcat/commit/b7a674041376a38c19ca58436d089312342df8f0))

### Chore

* chore(release): 1.0.5 [skip ci]

## [1.0.5](https://github.com/mbari-org/sdcat/compare/v1.0.4...v1.0.5) (2024-03-05)

### Bug Fixes

* correct some bugs calling libaries ([0668bda](https://github.com/mbari-org/sdcat/commit/0668bda21986e6a291f89e0c4a67305d223047ad)) ([`1258290`](https://github.com/mbari-org/sdcat/commit/12582901b051fb6407b90add813cc9ae7bb82e94))

### Fix

* fix: correct some bugs calling libaries ([`0668bda`](https://github.com/mbari-org/sdcat/commit/0668bda21986e6a291f89e0c4a67305d223047ad))

### Unknown

* Fixing dependencies ([`f069f0d`](https://github.com/mbari-org/sdcat/commit/f069f0dbca3896d98b8c0d01417ce0743365c31f))

## v1.0.4 (2024-03-05)

### Build

* build: bumped version ([`f1c17d3`](https://github.com/mbari-org/sdcat/commit/f1c17d37b1efecedcd01e1ecef55e9e73e8e96da))

### Chore

* chore(release): 1.0.4 [skip ci]

## [1.0.4](https://github.com/mbari-org/sdcat/compare/v1.0.3...v1.0.4) (2024-03-05)

### Bug Fixes

* correct more imports ([107a470](https://github.com/mbari-org/sdcat/commit/107a470a12abaac020a1bcb7e9749581e882eea8)) ([`5e76d32`](https://github.com/mbari-org/sdcat/commit/5e76d32af346cb01b2b7e5c5972923c1b6b4b5bc))

### Fix

* fix: correct more imports ([`107a470`](https://github.com/mbari-org/sdcat/commit/107a470a12abaac020a1bcb7e9749581e882eea8))

## v1.0.3 (2024-03-05)

### Build

* build: minor rev to pytest to try to fix PYTHONPATH ([`ed5dec7`](https://github.com/mbari-org/sdcat/commit/ed5dec79fd4d973d654a1b06e5ebe6f242e8aa02))

* build: bumped version ([`af998f4`](https://github.com/mbari-org/sdcat/commit/af998f4bc730660a537c7d3ba818b10585771e27))

### Chore

* chore(release): 1.0.3 [skip ci]

## [1.0.3](https://github.com/mbari-org/sdcat/compare/v1.0.2...v1.0.3) (2024-03-05)

### Bug Fixes

* correct import paths ([6ca36e1](https://github.com/mbari-org/sdcat/commit/6ca36e13a321cf8a4460a2598cf8f1e6cf73c2c3)) ([`3dce93a`](https://github.com/mbari-org/sdcat/commit/3dce93a216cde0790f77fa267f1e005b28358e39))

* chore: removed unused code ([`ccb95e3`](https://github.com/mbari-org/sdcat/commit/ccb95e33f2f4657eaf7cbbf237765d1a8bb077dc))

* chore: correct comment on multiproc ([`a826de9`](https://github.com/mbari-org/sdcat/commit/a826de930ab851a3ca77cff36f6cf4a99170ba7d))

* chore: correct comment ([`1004d95`](https://github.com/mbari-org/sdcat/commit/1004d9588cf46e4864faa9444d077f7df0179de1))

### Documentation

* docs: added sahi example and constrain python to 3.9-3.11 ([`240d630`](https://github.com/mbari-org/sdcat/commit/240d6302a393ef89b7202257c16335c3d71c28f8))

### Fix

* fix: correct import paths ([`6ca36e1`](https://github.com/mbari-org/sdcat/commit/6ca36e13a321cf8a4460a2598cf8f1e6cf73c2c3))

### Test

* test: add missing PYTHONPATH ([`4371400`](https://github.com/mbari-org/sdcat/commit/437140021ff3295a3142d94f9ada2f400af2c213))

* test: minor typo fix in pytest.yml ([`a108717`](https://github.com/mbari-org/sdcat/commit/a108717b7e93fc8efeb27a8e175c87f4d7be23c3))

* test: fixed path to main and allow pytest to be run manually ([`f586fd4`](https://github.com/mbari-org/sdcat/commit/f586fd4903a8bb9f470047e944308920a955c628))

## v1.0.2 (2024-02-17)

### Build

* build: bumped version ([`721b1b3`](https://github.com/mbari-org/sdcat/commit/721b1b30599cdde6c285fdded2564c844efdf7e4))

### Chore

* chore(release): 1.0.2 [skip ci]

## [1.0.2](https://github.com/mbari-org/sdcat/compare/v1.0.1...v1.0.2) (2024-02-17)

### Bug Fixes

* pass through config agnostic flat ([7b5f940](https://github.com/mbari-org/sdcat/commit/7b5f940bae4a10591f0c7c14b87f2a60b14d69c2))
* remove unused arguments ([ad06613](https://github.com/mbari-org/sdcat/commit/ad066138a233b0bcb79fda18c50b54da658fe545)) ([`3d9488d`](https://github.com/mbari-org/sdcat/commit/3d9488dc76a9749f6e81d5be897ef9cb5441a172))

### Documentation

* docs: added spec removal ref ([`987ad8a`](https://github.com/mbari-org/sdcat/commit/987ad8a2ced3506995b2c0d68920ab4e33194939))

### Fix

* fix: remove unused arguments ([`ad06613`](https://github.com/mbari-org/sdcat/commit/ad066138a233b0bcb79fda18c50b54da658fe545))

* fix: pass through config agnostic flat ([`7b5f940`](https://github.com/mbari-org/sdcat/commit/7b5f940bae4a10591f0c7c14b87f2a60b14d69c2))

## v1.0.1 (2024-02-15)

### Build

* build: bumped version ([`5508b33`](https://github.com/mbari-org/sdcat/commit/5508b337df46bb4387483777e28392d757e0202f))

### Chore

* chore(release): 1.0.1 [skip ci]

## [1.0.1](https://github.com/mbari-org/sdcat/compare/v1.0.0...v1.0.1) (2024-02-15)

### Bug Fixes

* added missing outer block for sahi pool ([91f88ef](https://github.com/mbari-org/sdcat/commit/91f88efdffde043c365b4b8871c4d6eae0e64f25)) ([`f9f8639`](https://github.com/mbari-org/sdcat/commit/f9f8639c1d0cfa254cc3eb98d91225aad69defe6))

### Documentation

* docs: minor reformatting ([`7afeca2`](https://github.com/mbari-org/sdcat/commit/7afeca2b56845dee95b882f777fe3a449ffd5477))

* docs: added ref to HDBSCAN paper ([`4a4edbc`](https://github.com/mbari-org/sdcat/commit/4a4edbca2d910dbdf7919716375f896d7bfde3da))

### Fix

* fix: added missing outer block for sahi pool ([`91f88ef`](https://github.com/mbari-org/sdcat/commit/91f88efdffde043c365b4b8871c4d6eae0e64f25))

## v1.0.0 (2024-02-14)

### Build

* build: added release and pytest ([`1bb6f28`](https://github.com/mbari-org/sdcat/commit/1bb6f28de4c6d017b2e617fe50a82d557675586e))

### Chore

* chore(release): 1.0.0 [skip ci]

# 1.0.0 (2024-02-14)

### Features

* initial commit ([148f17f](https://github.com/mbari-org/sdcat/commit/148f17f4a1e5af2a03380de964ff1140052d53b8)) ([`0c32f9b`](https://github.com/mbari-org/sdcat/commit/0c32f9b208ce7a0a6864b657b948b32a99422ca8))

### Documentation

* docs: correct links to image and reset version history ([`2c98876`](https://github.com/mbari-org/sdcat/commit/2c9887661b6d6fa0a562c470e6282a00f54d5d3a))

* docs: added example images and cluster workflow diagram ([`b228480`](https://github.com/mbari-org/sdcat/commit/b228480926cfce24d3e8575c786d1b952e8be256))

### Feature

* feat: initial commit ([`148f17f`](https://github.com/mbari-org/sdcat/commit/148f17f4a1e5af2a03380de964ff1140052d53b8))
