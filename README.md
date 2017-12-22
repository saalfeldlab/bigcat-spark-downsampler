# (BigCAT) Spark Downsampler
Command-line tool for downsampling label data (stored as LabelMultisetTypes in an [N5](https://github.com/saalfeldlab/n5) dataset) with Spark.

## Compile

Compile and install the master branch (version 0.0.3-SNAPSHOT) of [BigCAT](https://github.com/saalfeldlab/bigcat).

```
mvn clean install
```

 - Finally, clone this repository, and it should compile.

 - To make a "fat jar" with all dependencies added, run:

```
mvn clean compile assembly:single
```

## Downsampling

![Downsampling Operation](/img/downsampling_example.png)

<img align="right" src="/img/labelmultiset_structure.png" height=175 />

Downsampling works by merging groups of `LabelMultisetEntryList`'s:

For instance, in a block of pixels where each list has one entry,
with a count of 1 and a label of either 83 or 64, they will be downsampled
into a single list with 2 entries, one of which will have a label of 83
and the corresponding count, and the other will have a label of 64 and its
corresponding count.

The motivation for this method of downsampling is that it doesn't technically
lose information other than where within a downsampled block each label is.

When converting to a color for visualization purposes, this method also
allows a weighted color to be calculated from the colors of each label within
a downsampled pixel.

## Usage

```
java -Dspark.master=[spark_master] -jar target/bigcat-spark-downsampler-0.0.2-SNAPSHOT-jar-with-dependencies.jar [args]
```

#### Arguments

-  `--compression`, `-c`
   Compression type to use in output N5 dataset. Default: `RAW`
   
-  `--factor`, `-f`
   Factor by which to downscale the input image
   (**Required**)
   
-  `--idatasetname`, `--idata`, `-id`
   Input dataset name (N5 relative path from group)
   (**Required**)
   
-  `--igroupname`, `--igroup`, `-ig`
   Input group name (N5 group)
   (**Required**)
   
-  `--odatasetname`, `--odata`, `-od`
   Output dataset name (N5 relative path from group)
   (**Required**)
   
-  `--ogroupname`, `--ogroup`, `-og`
   Output group name (N5 group). Defaults to input group name
   
-  `--parallelblocks`, `-pb`
   Size of the blocks (in cells) to parallelize with Spark. Defaults to [16, 16, ... 16]

Note that the `spark.master` property must be set when running as well. See [here](http://spark.apache.org/docs/latest/submitting-applications.html#master-urls) for more information on Spark Master URLs.

#### Example

```
java -Dspark.master=local[*] -jar target/bigcat-spark-downsampler-0.0.2-SNAPSHOT-jar-with-dependencies.jar -ig ~/cremi-n5/ -id sampleA-fullres -od sampleA-8x8x2 -f 8,8,2 -c GZIP -pb 4,4,4
```

Would downsample the N5 label dataset at `~/cremi-n5/sampleA-fulres` by a factor of 8x8x2, and write to an N5 dataset (with GZIP compression) at `~/cremi-n5/sampleA-8x8x2`.

Note that the output group name is not specified, and defaults to the same as the input group name.

Also, `spark.master` is set to `local[*]`, which, according to [Spark documentation](http://spark.apache.org/docs/latest/submitting-applications.html#master-urls), will
> Run Spark locally with as many worker threads as logical cores on your machine.

Parallel block size simply determines the size of each block to parallelize with, 4x4x4 yields blocks of 64 cells each. This will have no impact on the output dataset, but if it is set too high relative to the input dataset, there may not be enough blocks for each worker thread to have something to work on (thus wasting the parallelism).
