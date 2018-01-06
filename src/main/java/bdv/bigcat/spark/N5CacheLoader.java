package bdv.bigcat.spark;

import java.io.IOException;
import java.nio.ByteBuffer;

import org.janelia.saalfeldlab.n5.DataBlock;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.N5Reader;

import bdv.labels.labelset.ByteUtils;
import bdv.labels.labelset.LongMappedAccessData;
import bdv.labels.labelset.VolatileLabelMultisetArray;
import net.imglib2.cache.CacheLoader;
import net.imglib2.img.cell.Cell;
import net.imglib2.img.cell.CellGrid;
import net.imglib2.util.Intervals;

public class N5CacheLoader implements CacheLoader< Long, Cell< VolatileLabelMultisetArray > >
{
	private final N5Reader n5;

	private final String dataset;

	private final CellGrid grid;

	@Override
	public Cell<VolatileLabelMultisetArray> get(Long key) {

		int numDimensions = grid.numDimensions();

		long[] cellMin = new long[ numDimensions ];
		int[] cellSize = new int[ numDimensions ];
		long[] gridPosition = new long[ numDimensions ];
		int[] cellDimensions = new int[ numDimensions ];

		grid.cellDimensions(cellDimensions);

		grid.getCellDimensions( key, cellMin, cellSize );

		for ( int i = 0; i < numDimensions; ++i )
		{
			gridPosition[ i ] = cellMin[ i ] / cellDimensions[ i ];
		}

		byte[] bytes = this.getData(gridPosition);

		ByteBuffer bb = ByteBuffer.wrap( bytes );

		final int[] data = new int[( int ) Intervals.numElements( cellSize )];
		final int listDataSize = bytes.length - 4 * data.length;
		final LongMappedAccessData listData = LongMappedAccessData.factory.createStorage( listDataSize );

		for ( int i = 0; i < data.length; ++i )
		{
			data[i] = bb.getInt();
		}

		for ( int i = 0; i < listDataSize; ++i )
			ByteUtils.putByte( bb.get(), listData.getData(), i );

		return new Cell< VolatileLabelMultisetArray >( cellSize, cellMin, new VolatileLabelMultisetArray( data, listData, true ) );
	}

	public N5CacheLoader( final N5Reader n5, final String dataset) throws IOException
	{
		this.grid = generateCellGrid( n5, dataset );
		this.n5 = n5;
		this.dataset = dataset;
	}

	private static CellGrid generateCellGrid(final N5Reader n5, final String dataset) throws IOException {
		final DatasetAttributes attributes = n5.getDatasetAttributes(dataset);

		long[] dimensions = attributes.getDimensions();
		int[] cellDimensions = attributes.getBlockSize();

		return new CellGrid( dimensions, cellDimensions);
	}

	protected byte[] getData( long... gridPosition )
	{
		final DataBlock< ? > block;
		try
		{
			block = n5.readBlock( dataset, n5.getDatasetAttributes( dataset ), gridPosition);
		}
		catch ( final IOException e )
		{
			throw new RuntimeException( e );
		}
		return (byte[]) block.getData();
	}
}