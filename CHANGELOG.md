# CHANGELOG



## v1.28.0 (2026-01-21)

### Feature

* feat: rfdetr supports

This adds support for running RF-DETR models from huggingface. For an example, see the MBARI UAV project repo here: https://huggingface.co/MBARI-org/rf-detrLarge-uavs-detectv0.   Also now supports defining the detection model in the config.ini file in the detect section, etc.. 

```ini
[detect] 
# Detection model
model = MBARI-org/rf-detrLarge-uavs-detectv0
``` ([`b464677`](https://github.com/mbari-org/sdcat/commit/b464677b916ab71b4e4e5c8cdd8873aa09febf3f))


## v1.27.11 (2025-12-14)

### Fix

* fix: handle cuda or cuda:1, cuda:0 device arguments ([`a60337d`](https://github.com/mbari-org/sdcat/commit/a60337db95d6c6f4f0938780081c425d5e4e933e))

### Unknown

* example vss processing ([`5c0a75e`](https://github.com/mbari-org/sdcat/commit/5c0a75e827a256e864e740c87e32a869c915ebc4))


## v1.27.10 (2025-09-05)

### Documentation

* docs: minor correction to JSON ([`c84c260`](https://github.com/mbari-org/sdcat/commit/c84c260c6335e5aae3cb811495f7f4a1faa09146))

* docs: minor reformatting ([`13a0371`](https://github.com/mbari-org/sdcat/commit/13a0371a9b614151c3e9f773d70126b8cda7a1f0))

* docs: minor reformatting ([`c09d411`](https://github.com/mbari-org/sdcat/commit/c09d411a5d002639b2d30c46d4ca2c582485ee08))

* docs: added more detail on visualization examples and analysis ([`3460cf3`](https://github.com/mbari-org/sdcat/commit/3460cf3f707c939f830d73e62aea32cb4aab2cc1))

* docs: some reorg of order &amp; reformatting in README.md for clarity; better description of batching ([`f0d5260`](https://github.com/mbari-org/sdcat/commit/f0d5260740d7f77b26958973f96b04f5e4e3adb1))

* docs: typo ([`f33c384`](https://github.com/mbari-org/sdcat/commit/f33c384f800b8f44d73393bf27f15e9d6ebf4572))

* docs: minor addition to badge and intro ([`62de3ff`](https://github.com/mbari-org/sdcat/commit/62de3ff9708df9e439b82674923dbb6c4dfb0bc6))

### Fix

* fix: handle out of bounds cluster merge ([`4681cdd`](https://github.com/mbari-org/sdcat/commit/4681cdd8a00c2587d77950a4bd4f2ee48e939769))

### Unknown

* docs: ([`b9b8aa9`](https://github.com/mbari-org/sdcat/commit/b9b8aa91e65a10c486a5820e0868c51791a2e2ba))


## v1.27.9 (2025-07-03)

### Fix

* fix: triggering release with README.md minor changes moving performance to separate section and usage ([`5e93945`](https://github.com/mbari-org/sdcat/commit/5e93945dd3d4293ee4db426d6ecab6be798cb31b))

### Unknown

* fix; triggering release with changelog ([`d816180`](https://github.com/mbari-org/sdcat/commit/d816180c97a15eeed38f85f632e47b262108da4a))


## v1.27.8 (2025-06-24)

### Performance

* perf: add support for rotated exemplar merge; this is a performance enhancement that can reduce total clusters by merging clusters based on the median of the exemplar and its rotated variant ([`798df04`](https://github.com/mbari-org/sdcat/commit/798df049808d4e41e49205d2b1b82fa391df42bb))


## v1.27.7 (2025-06-23)

### Fix

* fix: added missing changelog for build and triggering release to generate ([`297e63a`](https://github.com/mbari-org/sdcat/commit/297e63a49fda350616c69298c49d5409baa1d37f))


## v1.27.6 (2025-06-23)

### Performance

* perf: mostly a refactor and ruff format but remove reassigning exemplars after noise assignment which should never be different ([`09c871b`](https://github.com/mbari-org/sdcat/commit/09c871b5723add3d5cf4c20e1e58c206780145c3))

* perf: more aggressive duplicate removal and add low-information removal with cleanvision ([`dbc52d1`](https://github.com/mbari-org/sdcat/commit/dbc52d19b75733b9db0d03bad7a0a77ba0bfa8cb))


## v1.27.5 (2025-06-20)

### Fix

* fix: disable verbose mode for rapids; verbose mode causes and exit condition ([`58a9b4a`](https://github.com/mbari-org/sdcat/commit/58a9b4abe72a9a395414e5035223c37712cf45a0))


## v1.27.4 (2025-06-18)

### Fix

* fix: ensure the indices for noise and non noise are in alignment, some simplification on logic and correction to info ([`7731d60`](https://github.com/mbari-org/sdcat/commit/7731d6024357ddb2c1161ec2a73ca4207ee51383))


## v1.27.3 (2025-06-10)

### Fix

* fix: return 0 for empty detections instead of None to support aggregation ([`95200cc`](https://github.com/mbari-org/sdcat/commit/95200ccbc294c93160dc626ccc50d187a823c9ac))


## v1.27.2 (2025-06-05)

### Documentation

* docs: added precommit check example in DEVELOPMENT.md ([`5df3cd8`](https://github.com/mbari-org/sdcat/commit/5df3cd89a8ff6596e1b935292b561ce5232352f5))

### Fix

* fix: 1 image per 1 process for detection test ([`6eecc3a`](https://github.com/mbari-org/sdcat/commit/6eecc3a15d43a95c967032a3fa72a17d6f66e2d3))

### Performance

* perf: drop all cleaned remove_bad_images data, allow for highest max prevalence of 1 to see all cleanvision issues and require --use-cuhdbscan to use cuda enabled version as it is faster, but does not handle custom metric needed for cosine similarity ([`f36f127`](https://github.com/mbari-org/sdcat/commit/f36f12770c544fa9cc36769822d112e2906b67a1))


## v1.27.1 (2025-06-03)

### Fix

* fix: correct pass through of duplicates ([`193dc90`](https://github.com/mbari-org/sdcat/commit/193dc90d3dc55c122504d8b65d7fbd2e2775021a))


## v1.27.0 (2025-06-03)

### Build

* build: ruff format and add precommit hook ([`9d446c3`](https://github.com/mbari-org/sdcat/commit/9d446c33d4715eccdcc5c420ff5fab84093b59a4))

* build: enable only cuda build and remove duplicate Docker entrypoint ([`8f4aba0`](https://github.com/mbari-org/sdcat/commit/8f4aba0e60858c712a4eff92a48d698af7047fab))

### Feature

* feat: save command line text to summary.json for clustering for provenance ([`3437c52`](https://github.com/mbari-org/sdcat/commit/3437c52cf1fb1fe0c5c31b0923b04fb9d10bf8dd))


## v1.26.2 (2025-06-03)

### Fix

* fix: add --use-vits command ([`05cba0c`](https://github.com/mbari-org/sdcat/commit/05cba0cd89ab706ca1785fa2b03e00e7bf94b1e8))


## v1.26.1 (2025-06-03)

### Performance

* perf: adjust near duplicate threshold to be more lenient (more matches) for  remove_bad_images=True settings ([`b4eecf1`](https://github.com/mbari-org/sdcat/commit/b4eecf1e58a24d22860f3040d42f1d68b14d5f40))


## v1.26.0 (2025-06-02)

### Feature

* feat: added 100 dimensional reduction PCA to sdcat cluster roi/detections command with --use-pca ([`a090796`](https://github.com/mbari-org/sdcat/commit/a0907966cd4443d3f73d4a4dbafafcf58b168752))


## v1.25.0 (2025-06-02)

### Feature

* feat: triggering release with latest ([`c5a87c3`](https://github.com/mbari-org/sdcat/commit/c5a87c3c100b5a40e6c357ed35bd9bfa6eadbec4))


## v1.24.2 (2025-05-27)

### Performance

* perf: force cosine precomputed for CPU only clustering as this improves the quality of the clusters ([`a83c580`](https://github.com/mbari-org/sdcat/commit/a83c58049c8919d4bb803bc787a03a3d4d8bbaaf))


## v1.24.1 (2025-05-25)

### Fix

* fix: correct arg for batch size on cpu ([`5f5df19`](https://github.com/mbari-org/sdcat/commit/5f5df191593b18ebf1d3b4af0e743eace0401443))

* fix: remove device check in commands.py which fails for multi gpu; handled later in multi gpu model instantiation so no longer needed ([`bbdeaee`](https://github.com/mbari-org/sdcat/commit/bbdeaee5a07abfde06f21637115546859097a01f))


## v1.24.0 (2025-05-24)

### Feature

* feat: add support for cleaning near duplicates and specifying allowable_classes in the cluster section of the .ini (same formatting as the detection section). Both can significantly reduce the number of images that are clustered, thus reducing the computational overhead and output that may be loaded or visualized downstream ([`58270f3`](https://github.com/mbari-org/sdcat/commit/58270f3cf562230e3100a4194aad08cda4508b37))


## v1.23.0 (2025-05-23)

### Feature

* feat: drop weighted score option as it was not useful, improved performance loading vits model results, and added suport to set the batches sizes for vits with --vits-batch-size and the clustering wiht --hdbscan-batch-size ([`6dfa2fa`](https://github.com/mbari-org/sdcat/commit/6dfa2faa9d06487681e64a2d5635df297352d571))


## v1.22.0 (2025-05-23)

### Feature

* feat: plot noise cluster ([`aee021f`](https://github.com/mbari-org/sdcat/commit/aee021fc1ac1090addd45179713fb0fcd1151764))


## v1.21.3 (2025-05-23)

### Performance

* perf: kill ray if initialized to free up memory ([`3d8fa4b`](https://github.com/mbari-org/sdcat/commit/3d8fa4b3b53f89b13791989ddfe3f8999cacf3c1))


## v1.21.2 (2025-05-22)

### Performance

* perf: improved cluster batch performance for GPU acceleration ([`5a90dd9`](https://github.com/mbari-org/sdcat/commit/5a90dd98b330950f0cc2a42664b85d55044d06d4))


## v1.21.1 (2025-05-22)

### Performance

* perf: improved cluster cropping performance by moving from modin to pandas and grouping by frame and more logging ([`3713e7e`](https://github.com/mbari-org/sdcat/commit/3713e7e1ed15f750c90480d1f25c57a0910f5700))


## v1.21.0 (2025-05-22)

### Documentation

* docs: more detail on SAHI ([`04eee46`](https://github.com/mbari-org/sdcat/commit/04eee46c43454e82b1ba8b780cb2341dd7cd85d4))

### Feature

* feat: fast batch clustering (#25)

Optimized batch clustering with improved reporting. New features include:

* Clustering large archives. Tested on 1.5 million ROIs
* Clustered results saved to parquet format to facilitate better downstream processing
* Better aggregation and reporting on cluster summary in human readable JSON format, e.g.  
{
    &#34;dataset&#34;: {
        &#34;output&#34;: &#34;/data/output&#34;,
        &#34;clustering_algorithm&#34;: &#34;HDBSCAN&#34;,
        &#34;clustering_parameters&#34;: {
            &#34;min_cluster_size&#34;: 2,
            &#34;min_samples&#34;: 1,
            &#34;cluster_selection_method&#34;: &#34;leaf&#34;,
            &#34;metric&#34;: &#34;precomputed&#34;,
            &#34;algorithm&#34;: &#34;best&#34;,
            &#34;alpha&#34;: 1.3,
            &#34;cluster_selection_epsilon&#34;: 0.0
        },
        &#34;feature_embedding_model&#34;: &#34;MBARI-org/mbari-uav-vit-b-16&#34;,
        &#34;roi&#34;: true,
        &#34;input&#34;: [
            &#34;/data/input
        ],
        &#34;image_count&#34;: 328
    },
    &#34;statistics&#34;: {
        &#34;total_clusters&#34;: 4,
        &#34;cluster_coverage&#34;: &#34;1.23 (122.94%)&#34;,
        &#34;top_predictions&#34;: {
            &#34;class&#34;: &#34;Shark&#34;,
            &#34;percentage&#34;: &#34;3.35%&#34;
        }
    },
    &#34;sdcat_version&#34;: &#34;1.20.4&#34;
} ([`38cc5b5`](https://github.com/mbari-org/sdcat/commit/38cc5b5ca0dce187114feca7cd01cf12cbe4bcd1))


## v1.20.3 (2025-03-19)

### Fix

* fix: remove last cluster embedding correctly and fetch embeddings from dataframe ([`e955744`](https://github.com/mbari-org/sdcat/commit/e955744ca9ea961b654eacc58d7b12a0df55ea69))

### Performance

* perf: adjust min similarity to a more conservative .9 ([`697f138`](https://github.com/mbari-org/sdcat/commit/697f13839a1e09f4cd5d25d021146a40fb4eb7e3))


## v1.20.2 (2025-03-13)

### Fix

* fix: handle cases where no detections exists for specified start/end images by skipping over ([`87be2b7`](https://github.com/mbari-org/sdcat/commit/87be2b7f9547c268ce5afa74065acdce18970b9f))

### Performance

* perf: change combined detection+classify score to average ([`8def7a2`](https://github.com/mbari-org/sdcat/commit/8def7a2624c8dc7ac94853c40f3ea9e04911afa2))

* perf: better defaults for cluster alpha and epsilon ([`237890d`](https://github.com/mbari-org/sdcat/commit/237890df9db70c3babed4f36e8d9af5cff0a646c))

* perf: always assign noise cluster and correct label assign ([`5dd83bb`](https://github.com/mbari-org/sdcat/commit/5dd83bb30d63d0c2b227e714c1328fa3f042db6a))


## v1.20.1 (2025-03-12)

### Fix

* fix: correct arg for weighted score in cluster ([`658ef95`](https://github.com/mbari-org/sdcat/commit/658ef9552b5c66d6f656ce461738569d1c7dfa2c))


## v1.20.0 (2025-03-12)

### Documentation

* docs: updated workflow diagram ([`91ebe6a`](https://github.com/mbari-org/sdcat/commit/91ebe6a512b10d306a04e95ed373890b6e424ab7))

### Feature

* feat: rename to weighted_score and add back in the noise reassignment for higher coverage ([`0a361a0`](https://github.com/mbari-org/sdcat/commit/0a361a0eee69fd08a821dc96cb31decea7fa777a))

* feat: add weight_vits option to weight the scores from the detection model in the vits classification model ([`9bffe3d`](https://github.com/mbari-org/sdcat/commit/9bffe3d589a94081f0c9e9a9eb49e5a31e3e7f22))

* feat: added min-sample-size argument to allow for parameter sweeps ([`2cf9771`](https://github.com/mbari-org/sdcat/commit/2cf9771b40090375f146e02082156a5757ae9044))

* feat: added feature merge ([`90d182d`](https://github.com/mbari-org/sdcat/commit/90d182db13d8e56660f349bd62426dd18dc81b38))

* feat: added hdbscan algorithm choice for clustering ([`40b9ba2`](https://github.com/mbari-org/sdcat/commit/40b9ba21f685be52ee7f644fc2f25e2a4c3a7859))

* feat: added hdbscan algorithm choice for clustering ([`722da61`](https://github.com/mbari-org/sdcat/commit/722da6163aba7cfae937508a16d1e18209326349))

### Fix

* fix: remove cuda in cluster vits ([`d449a86`](https://github.com/mbari-org/sdcat/commit/d449a86394e6f560a2ea4f37cb1fa77c1b166a18))

### Performance

* perf: improved cluster coverage, weighted classification scores, and more options for running cluster sweeps

Performance
- Better handling of noise cluster and merging similar clusters.  This should improve cluster coverage and generate somewhat larger clusters with foundation models.

Features
- new arg to sdcat cluster `--algorithm`  default &#34;best&#34;; prims_kdtree or boruvka_kdtree may be worth trying
- new arg to sdcat cluster `--min-sample-size` which was only supported in the .ini file
- new arg to sdcat cluster `--weighted-score` which will weight the classification score with the detection score from the ViTS models through multiplication ([`a94e4a9`](https://github.com/mbari-org/sdcat/commit/a94e4a91faed622e335bc8da450da1a5d61097e3))


## v1.19.1 (2025-02-26)

### Feature

* feat: second pass merge cluster ([`a58ce18`](https://github.com/mbari-org/sdcat/commit/a58ce189473b2a66fda10096d631bbffbafee2d6))

### Fix

* fix: correct handling of bounded end image ([`f510e16`](https://github.com/mbari-org/sdcat/commit/f510e160197ac8b819376395de8cdaae2ac0072a))


## v1.19.0 (2025-02-26)

### Feature

* feat: added softcluster (#19)

* perf: better defaults for finer-grained clustering with google model

* feat: added soft clustering for leaf method only

* fix: remove default as this overrides what is in the .ini file

* perf: add batch size as command option --batch-size; default is 32 but best size depends on GPU/model memory

* fix: correct args for multiproc

* perf: combine soft/fuzzy and cosine sim

* docs: update workflow diagram with soft/fuzzy algorithm

* fix: handle models that only output top 1

* fix: only capture top 2 classes and scores

* chore: merged changes from main ([`cfaf784`](https://github.com/mbari-org/sdcat/commit/cfaf784c831bb61ee5998776a00c644f5824f801))


## v1.18.2 (2025-02-20)

### Fix

* fix: only capture top 2 classes and scores ([`9b85463`](https://github.com/mbari-org/sdcat/commit/9b854638b2303dd03752195d13bb3e68bd3dc291))


## v1.18.1 (2025-02-20)

### Fix

* fix: handle models that only output top 1 and default to cuda if available if not specified for clustering ([`164480a`](https://github.com/mbari-org/sdcat/commit/164480a2b25544e05152f136757cfdb2c1d989ef))


## v1.18.0 (2025-02-20)

### Feature

* feat: trigger release for --save-roi ([`8240a74`](https://github.com/mbari-org/sdcat/commit/8240a74d6363578509cc29196244b6c28d34d2c2))

* feat: add support for --save-roi  --roi-size (#18)

Added `--save-roi` and `--roi-size `options to sdcat detect. This saves the crops in a location compatible with the clustering stage, but can also be used outside of sdcat.  Data saved to crops

     ├── det_filtered                    # The filtered detections from the model
            ├── crops                       # Crops of the detections ([`9a801ac`](https://github.com/mbari-org/sdcat/commit/9a801ac34710f82d1b62bcd561253887858aa6cf))


## v1.17.0 (2025-02-07)

### Build

* build: relaxed requirements for compatibility with mbari-aidata since these are often used together ([`8bf55e3`](https://github.com/mbari-org/sdcat/commit/8bf55e3c6e29660766e87622b3cc791772767d70))

### Feature

* feat: trigger release to pypi with latest deps ([`2490823`](https://github.com/mbari-org/sdcat/commit/249082389403626c963012cbc5205bb8d2ba24e4))


## v1.16.3 (2025-01-27)

### Build

* build: updated poetry lock ([`b9a04e6`](https://github.com/mbari-org/sdcat/commit/b9a04e6bc3f2afab75bd9e3aea88b850967d7908))

### Performance

* perf: bump sahi to support YOLOv11 ([`d36b494`](https://github.com/mbari-org/sdcat/commit/d36b4942daac1614bd2b3cbac6e17c2300e06c21))


## v1.16.2 (2025-01-14)

### Performance

* perf: better handling of cuda devices by id across both detection and clustering commands with --device cuda:0 ([`ae8e395`](https://github.com/mbari-org/sdcat/commit/ae8e3958cca9751b0c4d1548174a30ca974636a8))


## v1.16.1 (2025-01-13)

### Fix

* fix: correct argument order to create_model and added types for float/int args in detect ([`6bb93bb`](https://github.com/mbari-org/sdcat/commit/6bb93bba765991d3255404a209c329441e4fb175))


## v1.16.0 (2025-01-11)

### Feature

* feat: added support for auto-detecting detection model types from huggingface and loading models from a directory. If models do not have the model type encoded in the name, e.g. yolov5 the --model-type yolov5 must be used ([`3ea7612`](https://github.com/mbari-org/sdcat/commit/3ea76120344e53936bc9cd63cea8815106abf312))


## v1.15.0 (2025-01-10)

### Feature

* feat: add second score and class assignment for roi cluster ([`8412941`](https://github.com/mbari-org/sdcat/commit/84129413fc87730f2b83a2fbdddf6d317614b269))


## v1.14.2 (2025-01-10)

### Fix

* fix: copy rois to crop path to avoid removal ([`00ca30f`](https://github.com/mbari-org/sdcat/commit/00ca30f288d94ab6b6325101a1242ff4280ed9e1))

### Performance

* perf: remove only dark and blurry ([`1e0de1f`](https://github.com/mbari-org/sdcat/commit/1e0de1fff4b2039d9b6e9c0569e8332bfe4eb29e))


## v1.14.1 (2024-12-07)

### Fix

* fix: correct clean_vision for roi and added check for is_low_information_issue and is_near_duplicates_issue ([`849d432`](https://github.com/mbari-org/sdcat/commit/849d4323dabb79dc98a53b5bf52066ba08271f13))


## v1.14.0 (2024-11-27)

### Feature

* feat: remove dark and blurry examples in clustering using cleanvision ([`c04fab7`](https://github.com/mbari-org/sdcat/commit/c04fab70749799891d0b59e78a0a47c76613fe5d))


## v1.13.2 (2024-11-23)

### Fix

* fix: correct vits assignment ([`d35ae96`](https://github.com/mbari-org/sdcat/commit/d35ae96b8d64f154621d7dfdc5ea2625fa12cf19))


## v1.13.1 (2024-11-23)

### Fix

* fix: handle index out of range on vits assign ([`2aee31a`](https://github.com/mbari-org/sdcat/commit/2aee31aa2b468a6b693a9710dfc896e5cdde236c))


## v1.13.0 (2024-11-21)

### Build

* build: updated poetry ([`2d97ed5`](https://github.com/mbari-org/sdcat/commit/2d97ed50a89ec3e8514631d072ac2d0b1ccd4c67))

### Feature

* feat: added support for assigning predictions to clusters ([`1b674fe`](https://github.com/mbari-org/sdcat/commit/1b674fe1ace2d0c5ab3e1a70c6e082b1064df065))

### Fix

* fix: correct order of file/byte for running vss ([`1d896ee`](https://github.com/mbari-org/sdcat/commit/1d896ee11fbab5965e3862b7923ed3f662722694))


## v1.12.1 (2024-10-29)

### Performance

* perf: always run saliency on multiproc regardless of cpu or gpu as it is not gpu enabled ([`b7e913e`](https://github.com/mbari-org/sdcat/commit/b7e913e855899c90c89eaa3f32d6156a8ffc6849))


## v1.12.0 (2024-10-29)

### Feature

* feat: added --model MBARI/yolov5x6-uavs-oneclass to detection ([`5fcd915`](https://github.com/mbari-org/sdcat/commit/5fcd915f29b223bd85d33e2f1e093c3ced517603))

* feat: assign unknowns via vss ([`947a8a3`](https://github.com/mbari-org/sdcat/commit/947a8a387150f4c908380647664ece8dd5d57113))

* feat: added pass through of vss server and renaming cluster id in exemplar output ([`67c4202`](https://github.com/mbari-org/sdcat/commit/67c4202c6ca7d661ee2cdfadb922b62d7789efa1))

### Fix

* fix: correct boolen for remove corners ([`4161e62`](https://github.com/mbari-org/sdcat/commit/4161e62aec5856beea58c9e3b251f58d9dbc93ce))

* fix: correct handling of remove corner ([`60df1ae`](https://github.com/mbari-org/sdcat/commit/60df1aeed140928c3b741e6873bd89a7117ebaec))

* fix: assign exemplar to crop ([`05b97c9`](https://github.com/mbari-org/sdcat/commit/05b97c9529c049ab62e9096ea0d90310f16b39ab))

### Performance

* perf: assign everything, not just clusters and assign top prediction if there is more than a .05 spread in the top 2 ([`f6cf171`](https://github.com/mbari-org/sdcat/commit/f6cf171b832a598a51255aed25e3d9dc64a3f778))

### Unknown

* per: batch 32 ([`a959b0f`](https://github.com/mbari-org/sdcat/commit/a959b0f78df0820ebbe6b36fe9be7e2b989ca5b3))


## v1.11.1 (2024-09-25)

### Fix

* fix: correct CUDA HDBSCAN fit ([`c6a8db3`](https://github.com/mbari-org/sdcat/commit/c6a8db33e274c356a4c5d8b26a7f380f615fb6e1))


## v1.11.0 (2024-09-18)

### Feature

* feat: add to cluster command option --skip-visualization since this takes some time and is not needed for prod ml workflows ([`48d1f19`](https://github.com/mbari-org/sdcat/commit/48d1f1955510ad2269def64dee9e14b96db36c0f))


## v1.10.5 (2024-09-13)

### Fix

* fix: minor fix to correct debug statement and correct order arg for cluster ([`645e979`](https://github.com/mbari-org/sdcat/commit/645e9790909496e8f3a385174270bb70f069a165))


## v1.10.4 (2024-09-05)

### Fix

* fix: move last cluster removal to exemplars only ([`96fc5db`](https://github.com/mbari-org/sdcat/commit/96fc5dbf5953bf823b929e1d41958bcfa9978698))


## v1.10.3 (2024-09-03)

### Fix

* fix: do not load the last cluster in exemplars ([`bbc3a9a`](https://github.com/mbari-org/sdcat/commit/bbc3a9a0fd73c1ef3852f19928c01c983c34d241))


## v1.10.2 (2024-08-22)

### Fix

* fix: trigger release with changes to __init__.py ([`e275d9c`](https://github.com/mbari-org/sdcat/commit/e275d9cec3094c37d3bca337ebfe2bfb8364b5d1))


## v1.10.1 (2024-08-22)

### Fix

* fix: trigger release with changes to pyproject.toml to update __version__ ([`6b7f641`](https://github.com/mbari-org/sdcat/commit/6b7f641c192846167926d7c7c7920c0680eb128f))


## v1.10.0 (2024-08-22)

### Build

* build: adjust test path ([`83119fb`](https://github.com/mbari-org/sdcat/commit/83119fbbf479a2f09543355b2872a9f58a2f34ce))

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


## v1.9.3 (2024-07-31)

### Fix

* fix: skip over HDBSCAN for small cluster ([`cbf1ea3`](https://github.com/mbari-org/sdcat/commit/cbf1ea3381aef94a17eb2ae27924df2238c117a9))


## v1.9.2 (2024-07-31)

### Fix

* fix: handle small cluster dataset reduction plot by switching to PCA and reduce build to only linux ([`16ab4de`](https://github.com/mbari-org/sdcat/commit/16ab4ded38b62665007cc5e5b1b6d1eec5865803))


## v1.9.1 (2024-07-31)

### Fix

* fix: correct handling of single cluster ([`f6ede19`](https://github.com/mbari-org/sdcat/commit/f6ede19b84b5f8808f34efa9fbf8e36479f5c8d8))


## v1.9.0 (2024-07-31)

### Documentation

* docs: minor update to intro ([`ec75265`](https://github.com/mbari-org/sdcat/commit/ec7526584df6d8d63c10c75f672151fb5dc91aa3))

### Feature

* feat: added support for arg --cluster-selection-method eof/leaf to support small clusters which work better with eof ([`6480ac2`](https://github.com/mbari-org/sdcat/commit/6480ac279d1383f2c08691998fb12ca2705bffeb))


## v1.8.4 (2024-07-29)

### Performance

* perf: migrated to transformers library with batch size of 8, moved some imports to only where needed for some speed-up, and removed unused activation maps. ([`c5fe725`](https://github.com/mbari-org/sdcat/commit/c5fe72523fdfb1eb7b674565e5724686a4ec65d1))


## v1.8.3 (2024-07-23)

### Documentation

* docs: fix typo ([`5506bad`](https://github.com/mbari-org/sdcat/commit/5506badfa072ecedfa5ee6c1dbcd977dcb2ff90c))

* docs: shorten readme and update workflow to reflect latest ([`152624e`](https://github.com/mbari-org/sdcat/commit/152624e159c5f535ddc49132d719ab2b1452d06b))

### Fix

* fix: handle bad image crops (zero length). Fixes #13. ([`c02394c`](https://github.com/mbari-org/sdcat/commit/c02394c88e7f446d4fb1ddaea1581e5922ebf030))


## v1.8.2 (2024-07-22)

### Documentation

* docs: updated cluster to include latest roi ([`b02464e`](https://github.com/mbari-org/sdcat/commit/b02464e99ad50db709e53a6395811ff1c7dcc754))

* docs: more detail on detection and model table ([`981beb2`](https://github.com/mbari-org/sdcat/commit/981beb2252dec546873cdbf136a6d24a1f2e6c4b))

### Fix

* fix: fixes a bug in handling zero clusters which occasionally happes for very small roi collections. Closes #12. ([`0f69805`](https://github.com/mbari-org/sdcat/commit/0f698052af54af3f44a8ec2faa335e2f66077b5b))


## v1.8.1 (2024-07-19)

### Fix

* fix: bump version to 1.8.0 still failing in ci ([`ed405e8`](https://github.com/mbari-org/sdcat/commit/ed405e8b5c892746680e1b2985c0842cb0581668))


## v1.8.0 (2024-07-19)

### Build

* build: slim docker images a tad, switch to generic docker user, and replace with miniconda install for better torch/cuda support ([`782848d`](https://github.com/mbari-org/sdcat/commit/782848d9ac2187b0c298e358ef642943e4e3f88f))

### Feature

* feat: auto default logs for different use cases: testing, local dev, and to /tmp on permission failure ([`0c57d45`](https://github.com/mbari-org/sdcat/commit/0c57d4552b90875796a11c1e2899a07c14f1bf53))


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


## v1.5.0 (2024-06-04)


## v1.4.1 (2024-06-04)

### Fix

* fix: conditional import of multicore tsne ([`76ec895`](https://github.com/mbari-org/sdcat/commit/76ec89589319664a98c8523312a859aa3475b1c2))


## v1.4.0 (2024-05-31)

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


## v1.3.0 (2024-05-02)

### Feature

* feat: added pass through of slicing overlap and postprocess_match_metric ([`f3b14bb`](https://github.com/mbari-org/sdcat/commit/f3b14bbf953a5eb18955339c36279cdabd761374))


## v1.2.2 (2024-05-02)

### Fix

* fix: allow for override of detect params with config.ini;correct save NMS output and detect single image ([`4ab8780`](https://github.com/mbari-org/sdcat/commit/4ab878033ac423918c7a1eb1eb2416a435779276))

* fix: more detail on cluster args, allow for override with config.ini, and set CUDA_VISIBLE_DEVICES in case not set ([`40a96ef`](https://github.com/mbari-org/sdcat/commit/40a96efcfd4fd9f7cf2fb8aecb11970d5c227671))


## v1.2.1 (2024-04-30)

### Fix

* fix: sorted order needed to make start/end filtering work ([`9caea73`](https://github.com/mbari-org/sdcat/commit/9caea73bc3263077e057886eee2ce2a95e1e139d))


## v1.2.0 (2024-04-30)

### Documentation

* docs: Erasing some prints in the code ([`622c119`](https://github.com/mbari-org/sdcat/commit/622c1191a9055e06fa2bd818ae5ba407cca607bb))

### Feature

* feat: added support for specifying start/ending file names to support skipping image blocks ([`f344865`](https://github.com/mbari-org/sdcat/commit/f344865812ede812a4bff7b15fd24f71b0d8c418))


## v1.1.0 (2024-04-23)

### Feature

* feat: New features for clustering ([`b74d1b1`](https://github.com/mbari-org/sdcat/commit/b74d1b1e3025db401d996ee49c8e79b9d26c8998))


## v1.0.9 (2024-04-23)

### Fix

* fix: deleting the os version ([`e63bb73`](https://github.com/mbari-org/sdcat/commit/e63bb738e9d8e6c08b1c9913788182b681b550d6))


## v1.0.8 (2024-04-23)

### Fix

* fix: Adding .tiff images and vizualization tools with tqdm ([`d0b99e2`](https://github.com/mbari-org/sdcat/commit/d0b99e20e1c1a27eee3ba210d0d6724c4e509a1d))

### Unknown

* Including clahe in the commands ([`ea92e2f`](https://github.com/mbari-org/sdcat/commit/ea92e2fde01e646bc3506890a8279eb881fbd3c1))


## v1.0.7 (2024-03-12)

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
