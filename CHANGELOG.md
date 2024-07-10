# CHANGELOG

## v1.7.0 (2024-07-10)

### Build

* build: removed unused imports and bump torch to python3.11 compatible and ([`57c6c2b`](https://github.com/mbari-org/sdcat/commit/57c6c2b4232730ffec878b5c3c75491f1ff42c87))

* build: fix docker build ([`0d7ab83`](https://github.com/mbari-org/sdcat/commit/0d7ab83fc57d331fc58dabb8eb8291790b22bc3e))

### Feature

* feat: added tsne as an option, defaulted to false as it is generating too many clusters ([`8f974f3`](https://github.com/mbari-org/sdcat/commit/8f974f346b1ab1d1d1c0d16cad21494f8f5f17c7))

## v1.6.0 (2024-06-26)

### Build

* build: add sys path for convenience ([`d668246`](https://github.com/mbari-org/sdcat/commit/d668246fa3b6591723504d9dc26fbe9d58acc24f))

### Documentation

* docs: minor update to correct log format ([`6cf6453`](https://github.com/mbari-org/sdcat/commit/6cf64537ca551a56f1bd6bba2d47d42cc4dcb7e7))

* docs: updated workflow diagram to reflect tSNE ([`f1cecf8`](https://github.com/mbari-org/sdcat/commit/f1cecf866fe7e9b41e71dfb198910ef44a30bc65))

### Feature

* feat: added export of the examplars, handle small clustering input by bypassing tSNE which fails, and make dino_vits8 the default ([`2306316`](https://github.com/mbari-org/sdcat/commit/2306316f4218244c8fc340261140de9d4af0dc8b))

## v1.5.0 (2024-06-03)

## v1.4.1 (2024-06-03)

### Fix

* fix: conditional import of multicore tsne ([`76ec895`](https://github.com/mbari-org/sdcat/commit/76ec89589319664a98c8523312a859aa3475b1c2))

## v1.4.0 (2024-05-30)

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

## v1.3.0 (2024-05-01)

### Feature

* feat: added pass through of slicing overlap and postprocess_match_metric ([`f3b14bb`](https://github.com/mbari-org/sdcat/commit/f3b14bbf953a5eb18955339c36279cdabd761374))

## v1.2.2 (2024-05-01)

### Fix

* fix: allow for override of detect params with config.ini;correct save NMS output and detect single image ([`4ab8780`](https://github.com/mbari-org/sdcat/commit/4ab878033ac423918c7a1eb1eb2416a435779276))

* fix: more detail on cluster args, allow for override with config.ini, and set CUDA_VISIBLE_DEVICES in case not set ([`40a96ef`](https://github.com/mbari-org/sdcat/commit/40a96efcfd4fd9f7cf2fb8aecb11970d5c227671))

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

### Unknown

* Testing #2 ([`0d8b2f4`](https://github.com/mbari-org/sdcat/commit/0d8b2f49fecee4370d5d74dfb74171216591f5e7))

## v1.0.6 (2024-03-05)

### Build

* build: bumped version ([`404036b`](https://github.com/mbari-org/sdcat/commit/404036bc673cf541c7f7e45d96d8bbd13a1399b6))

### Fix

* fix: correct some bugs calling libaries ([`d69171b`](https://github.com/mbari-org/sdcat/commit/d69171ba9b9caafa218439df810f9b73ace99023))

## v1.0.5 (2024-03-05)

### Build

* build: bumped version ([`b7a6740`](https://github.com/mbari-org/sdcat/commit/b7a674041376a38c19ca58436d089312342df8f0))

### Fix

* fix: correct some bugs calling libaries ([`0668bda`](https://github.com/mbari-org/sdcat/commit/0668bda21986e6a291f89e0c4a67305d223047ad))

### Unknown

* Fixing dependencies ([`f069f0d`](https://github.com/mbari-org/sdcat/commit/f069f0dbca3896d98b8c0d01417ce0743365c31f))

## v1.0.4 (2024-03-05)

### Build

* build: bumped version ([`f1c17d3`](https://github.com/mbari-org/sdcat/commit/f1c17d37b1efecedcd01e1ecef55e9e73e8e96da))

### Fix

* fix: correct more imports ([`107a470`](https://github.com/mbari-org/sdcat/commit/107a470a12abaac020a1bcb7e9749581e882eea8))

## v1.0.3 (2024-03-05)

### Build

* build: minor rev to pytest to try to fix PYTHONPATH ([`ed5dec7`](https://github.com/mbari-org/sdcat/commit/ed5dec79fd4d973d654a1b06e5ebe6f242e8aa02))

* build: bumped version ([`af998f4`](https://github.com/mbari-org/sdcat/commit/af998f4bc730660a537c7d3ba818b10585771e27))

### Documentation

* docs: added sahi example and constrain python to 3.9-3.11 ([`240d630`](https://github.com/mbari-org/sdcat/commit/240d6302a393ef89b7202257c16335c3d71c28f8))

### Fix

* fix: correct import paths ([`6ca36e1`](https://github.com/mbari-org/sdcat/commit/6ca36e13a321cf8a4460a2598cf8f1e6cf73c2c3))

## v1.0.2 (2024-02-17)

### Build

* build: bumped version ([`721b1b3`](https://github.com/mbari-org/sdcat/commit/721b1b30599cdde6c285fdded2564c844efdf7e4))

### Documentation

* docs: added spec removal ref ([`987ad8a`](https://github.com/mbari-org/sdcat/commit/987ad8a2ced3506995b2c0d68920ab4e33194939))

### Fix

* fix: remove unused arguments ([`ad06613`](https://github.com/mbari-org/sdcat/commit/ad066138a233b0bcb79fda18c50b54da658fe545))

* fix: pass through config agnostic flat ([`7b5f940`](https://github.com/mbari-org/sdcat/commit/7b5f940bae4a10591f0c7c14b87f2a60b14d69c2))

## v1.0.1 (2024-02-15)

### Build

* build: bumped version ([`5508b33`](https://github.com/mbari-org/sdcat/commit/5508b337df46bb4387483777e28392d757e0202f))

### Documentation

* docs: minor reformatting ([`7afeca2`](https://github.com/mbari-org/sdcat/commit/7afeca2b56845dee95b882f777fe3a449ffd5477))

* docs: added ref to HDBSCAN paper ([`4a4edbc`](https://github.com/mbari-org/sdcat/commit/4a4edbca2d910dbdf7919716375f896d7bfde3da))

### Fix

* fix: added missing outer block for sahi pool ([`91f88ef`](https://github.com/mbari-org/sdcat/commit/91f88efdffde043c365b4b8871c4d6eae0e64f25))

## v1.0.0 (2024-02-14)

### Build

* build: added release and pytest ([`1bb6f28`](https://github.com/mbari-org/sdcat/commit/1bb6f28de4c6d017b2e617fe50a82d557675586e))

### Documentation

* docs: correct links to image and reset version history ([`2c98876`](https://github.com/mbari-org/sdcat/commit/2c9887661b6d6fa0a562c470e6282a00f54d5d3a))

* docs: added example images and cluster workflow diagram ([`b228480`](https://github.com/mbari-org/sdcat/commit/b228480926cfce24d3e8575c786d1b952e8be256))

### Feature

* feat: initial commit ([`148f17f`](https://github.com/mbari-org/sdcat/commit/148f17f4a1e5af2a03380de964ff1140052d53b8))
