package bdv.bigcat.spark;

import java.nio.ByteBuffer;
import java.util.Arrays;

import org.apache.spark.api.java.function.Function;
import org.janelia.saalfeldlab.n5.ByteArrayDataBlock;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.N5FSReader;
import org.janelia.saalfeldlab.n5.N5FSWriter;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;

import bdv.labels.labelset.ByteUtils;
import bdv.labels.labelset.Label;
import bdv.labels.labelset.LabelMultisetEntry;
import bdv.labels.labelset.LabelMultisetEntryList;
import bdv.labels.labelset.LabelMultisetType;
import bdv.labels.labelset.LongMappedAccessData;
import bdv.labels.labelset.Multiset.Entry;
import bdv.labels.labelset.VolatileLabelMultisetArray;
import net.imglib2.RandomAccessible;
import net.imglib2.cache.CacheLoader;
import net.imglib2.cache.img.CachedCellImg;
import net.imglib2.cache.ref.BoundedSoftRefLoaderCache;
import net.imglib2.cache.util.LoaderCacheAsCacheAdapter;
import net.imglib2.img.cell.Cell;
import net.imglib2.img.cell.CellGrid;
import net.imglib2.view.Views;

public class SparkDownsampleFunction implements Function< DownsampleBlock, Integer >
{
	private static final long serialVersionUID = 1384028449836651390L;

	private final String inputGroupName;

	private final String inputDatasetName;

	private final long[] factor;

	private final String outputGroupName;

	private final String outputDatasetName;

	public static LabelMultisetType getOutOfBounds( int count )
	{
		final LongMappedAccessData listData = LongMappedAccessData.factory.createStorage( 32 );

		final LabelMultisetEntryList list = new LabelMultisetEntryList( listData, 0 );
		final LabelMultisetEntry entry = new LabelMultisetEntry( 0, 1 );

		list.createListAt( listData, 0 );
		entry.setId( Label.OUTSIDE );
		entry.setCount( count );
		list.add( entry );

		int[] data = new int[] { 0 };

		return new LabelMultisetType( new VolatileLabelMultisetArray( data, listData, true ) );
	}

	public SparkDownsampleFunction( String inputGroupName, String inputDatasetName, long[] factor, String outputGroupName, String outputDatasetName )
	{
		this.inputGroupName = inputGroupName;
		this.inputDatasetName = inputDatasetName;
		this.factor = factor;
		this.outputGroupName = outputGroupName;
		this.outputDatasetName = outputDatasetName;
	}

	@Override
	public Integer call( DownsampleBlock targetRegion ) throws Exception
	{
		final N5Reader reader = new N5FSReader( inputGroupName );
		final DatasetAttributes attr = reader.getDatasetAttributes( inputDatasetName );

		final long[] dimensions = attr.getDimensions();
		final int[] blocksize = attr.getBlockSize();

		final int nDim = dimensions.length;
		final long[] offset = new long[ nDim ];

		final int[] targetSize = targetRegion.getSize();
		final long[] targetMin = targetRegion.getMin();

		long[] actualLocation = new long[ nDim ];
		long[] actualSize = new long[ nDim ];
		long[] actualMax = new long[ nDim ];

		final CacheLoader< Long, Cell< VolatileLabelMultisetArray > > cacheLoader = new N5CacheLoader( reader, inputDatasetName );

		final BoundedSoftRefLoaderCache< Long, Cell< VolatileLabelMultisetArray > > cache = new BoundedSoftRefLoaderCache<>( 1 );
		final LoaderCacheAsCacheAdapter< Long, Cell< VolatileLabelMultisetArray > > wrappedCache = new LoaderCacheAsCacheAdapter<>( cache, cacheLoader );

		final CachedCellImg< LabelMultisetType, VolatileLabelMultisetArray > inputImg = new CachedCellImg< LabelMultisetType, VolatileLabelMultisetArray >( new CellGrid( dimensions, blocksize ), new LabelMultisetType(), wrappedCache, new VolatileLabelMultisetArray( 0, true ) );

		int eachCount = 0;
		for ( Entry< Label > e : inputImg.firstElement().entrySet() )
			eachCount += e.getCount();

		final RandomAccessible< LabelMultisetType > extendedImg =
				Views.extendValue(
						inputImg,
						getOutOfBounds( eachCount ) );

		VolatileLabelMultisetArray downscaledCell;

		int numCellsDownscaled = 0;

		final N5Writer writer = new N5FSWriter( outputGroupName );
		final DatasetAttributes writerAttributes = writer.getDatasetAttributes( outputDatasetName );

		long[] writeLocation = new long[ nDim ];

		for ( int d = 0; d < nDim; )
		{
			Arrays.setAll( actualLocation, i -> factor[ i ] * ( targetMin[ i ] + offset[ i ] ) );

			// TODO: figure out what part of this is redundant, if any, and
			// clarify it
			Arrays.setAll( actualSize, i -> Math.min( factor[ i ] * ( offset[ i ] + blocksize[ i ] > targetMin[ i ] + targetSize[ i ] ? ( targetMin[ i ] + targetSize[ i ] - offset[ i ] ) : blocksize[ i ] ), factor[ i ] * ( int ) Math.ceil( ( dimensions[ i ] - actualLocation[ i ] ) / ( double ) factor[ i ] ) ) );

			downscaledCell = Downscale.downscale(
					Views.interval( extendedImg, inputImg ),
					factor,
					actualSize,
					actualLocation );

			byte[] bytes = new byte[ getSerializedVolatileLabelMultisetArraySize( downscaledCell ) ];
			serializeVolatileLabelMultisetArray( downscaledCell, bytes );

			for ( int i = 0; i < nDim; i++ )
				writeLocation[ i ] = ( targetMin[ i ] + offset[ i ] ) / blocksize[ i ];

			final ByteArrayDataBlock dataBlock = new ByteArrayDataBlock( blocksize, writeLocation, bytes );
			writer.writeBlock( outputDatasetName, writerAttributes, dataBlock );

			numCellsDownscaled++;

			for ( d = 0; d < nDim; d++ )
			{
				offset[ d ] += blocksize[ d ];
				if ( offset[ d ] < targetSize[ d ] )
					break;
				else
					offset[ d ] = 0;
			}
		}

		return numCellsDownscaled;
	}

	public static int getSerializedVolatileLabelMultisetArraySize( VolatileLabelMultisetArray array )
	{
		return ( int ) ( array.getCurrentStorageArray().length * Integer.BYTES + array.getListDataUsedSizeInBytes() );
	}

	public static void serializeVolatileLabelMultisetArray(VolatileLabelMultisetArray array, byte[] bytes) {

		int[] curStorage = array.getCurrentStorageArray();
		long[] data = ( ( LongMappedAccessData )array.getListData() ).getData();

		ByteBuffer bb = ByteBuffer.wrap(bytes);

		for ( final int d : curStorage )
			bb.putInt( d );

		for( long i = 0; i < array.getListDataUsedSizeInBytes(); i ++)
			bb.put( ByteUtils.getByte( data, i ) );

	}
}
