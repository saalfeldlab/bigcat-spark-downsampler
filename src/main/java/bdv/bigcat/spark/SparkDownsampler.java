package bdv.bigcat.spark;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.janelia.saalfeldlab.n5.CompressionType;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.N5FSReader;
import org.janelia.saalfeldlab.n5.N5FSWriter;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;

public class SparkDownsampler
{
	static public class Parameters {

		@Parameter( names = { "--igroupname", "--igroup", "-ig" }, description = "Input group name (N5 group)" )
		public String inputGroupName = null;

		@Parameter( names = { "--idatasetname", "--idata", "-id" }, description = "Input dataset name (N5 relative path from group)" )
		public String inputDatasetName = null;

		@Parameter( names = { "--ogroupname", "--ogroup", "-og" }, description = "Output group name (N5 group)")
		public String outputGroupName = null;

		@Parameter( names = { "--odatasetname", "--odata", "-od" }, description = "Output dataset name (N5 relative path from group)")
		public String outputDatasetName = null;

		@Parameter( names = { "--factor", "-f"}, description = "Factor by which to downscale the input image" )
		public List<Integer> factor = new ArrayList<Integer>();

		@Parameter( names = { "--parallelblocks", "-pb"}, description = "Size of the blocks (in cells) to parallelize with Spark" )
		public List<Integer> parallelBlockSize = new ArrayList<Integer>();

		@Parameter( names = { "--compression", "-c"}, description = "Compression type to use in output N5 dataset" )
		public String compressionType = "RAW";


		public boolean init() {
			if(inputGroupName == null || inputDatasetName == null || factor.size()==0)
				return false;

			if(parallelBlockSize.size()==0)
				for(int i = 0; i < factor.size(); i ++)
					parallelBlockSize.add(16);

			if(outputGroupName == null)
				outputGroupName = inputGroupName; // default to using same group

			if(outputDatasetName == null) {
				outputDatasetName = inputDatasetName + "-downscaled-";
				for(int i = 0; i < factor.size(); i ++)
					outputDatasetName += (i!=0?"x":"") + i;
			}
			return true;
		}
	}


	public static void main(String[] args) throws IOException {
		final Parameters params = new Parameters();
		JCommander commander = new JCommander( params, args );

		if(!params.init()) {
			commander.usage();
			return;
		}

		SparkConf conf = new SparkConf().setAppName( "SparkDownsampler" );
		JavaSparkContext sc = new JavaSparkContext( conf );
		SparkDownsampler.downsample(sc,
				new N5FSReader(params.inputGroupName), params.inputGroupName, params.inputDatasetName,
				params.factor.stream().mapToLong(i->i).toArray(), params.parallelBlockSize.stream().mapToInt(i->i).toArray(),
				params.outputGroupName, params.outputDatasetName,
				CompressionType.valueOf(params.compressionType));
	}

	public static void downsample(JavaSparkContext sc,
			N5Reader reader, String readGroupName, String readDatasetName,
			long[] downsampleFactor, int[] parallelSize,
			String outputGroupName, String outputDatasetName,
			CompressionType compressionType) throws IOException  {

		DatasetAttributes attributes = reader.getDatasetAttributes(readDatasetName);

		long[] dimensions = attributes.getDimensions();
		int[] blockSize = attributes.getBlockSize();

		List<DownsampleBlock> parallelizeSections = new ArrayList<DownsampleBlock>();

		int nDim = attributes.getNumDimensions();

		long[] downsampledDimensions = new long[nDim];
		Arrays.setAll(downsampledDimensions, i -> (int)(Math.ceil((double)dimensions[i]/downsampleFactor[i])));

		final long[] offset = new long[nDim];
		int[] actualSize = new int[nDim];

		for(int d = 0; d < nDim; ) {

			for(int i = 0; i < nDim; i ++)
				actualSize[i] = (int) Math.min(parallelSize[i]*blockSize[i], downsampledDimensions[i]-offset[i]);

			parallelizeSections.add(new DownsampleBlock(offset.clone(), actualSize.clone()));

			for( d = 0; d < nDim; d++) {
				offset[d] += parallelSize[d]*blockSize[d];
				if(offset[d] < downsampledDimensions[d])
					break;
				else
					offset[d] = 0;
			}
		}

		N5Writer writer = new N5FSWriter(outputGroupName);
		writer.createDataset(outputDatasetName, downsampledDimensions, blockSize, DataType.UINT8, compressionType);

		Integer output = sc.parallelize(parallelizeSections)
			.map( new SparkDownsampleFunction(readGroupName, readDatasetName, downsampleFactor, outputGroupName, outputDatasetName))
			.reduce( (i,j) -> i+j);

		System.out.println("Across " + parallelizeSections.size() + " parallelized sections, " + output + " cells were downscaled");
	}
}