package core.algo.vertical;

import static org.jocl.CL.CL_CONTEXT_PLATFORM;
import static org.jocl.CL.CL_DEVICE_TYPE_ALL;
import static org.jocl.CL.CL_MEM_COPY_HOST_PTR;
import static org.jocl.CL.CL_MEM_READ_ONLY;
import static org.jocl.CL.CL_MEM_READ_WRITE;
import static org.jocl.CL.CL_TRUE;
import static org.jocl.CL.clBuildProgram;
import static org.jocl.CL.clCreateBuffer;
import static org.jocl.CL.clCreateCommandQueue;
import static org.jocl.CL.clCreateContext;
import static org.jocl.CL.clCreateKernel;
import static org.jocl.CL.clCreateProgramWithSource;
import static org.jocl.CL.clEnqueueNDRangeKernel;
import static org.jocl.CL.clEnqueueReadBuffer;
import static org.jocl.CL.clGetDeviceIDs;
import static org.jocl.CL.clGetPlatformIDs;
import static org.jocl.CL.clReleaseCommandQueue;
import static org.jocl.CL.clReleaseContext;
import static org.jocl.CL.clReleaseKernel;
import static org.jocl.CL.clReleaseMemObject;
import static org.jocl.CL.clReleaseProgram;
import static org.jocl.CL.clSetKernelArg;

import java.util.ArrayList;

import core.utils.CollectionUtils;
import core.utils.PartitioningUtils;
import db.schema.BenchmarkTables;
import db.schema.utils.WorkloadUtils;
import experiments.AlgorithmResults;
import experiments.AlgorithmRunner;
import gnu.trove.iterator.TIntIterator;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;

import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_platform_id;
import org.jocl.cl_program;

/**
 * Implementation of the AutoPart vertical partitioning algorithm from S.
 * Papadomanolakis and A. Ailamaki, SSDBM '04.
 * 
 * @author Endre Palatinus
 * 
 */
public class AutoPartCL_Reference extends AbstractPartitionsAlgorithm {

	/**
	 * The amount of storage available for attribute replication expressed as a
	 * factor of increase in storage requirements.
	 */
	private double replicationFactor = 0.5;

	/** The minimal number of queries that should access a candidate fragment. */
	private int queryExtentThreshold = 1;

	public AutoPartCL_Reference(AlgorithmConfig config) {
		super(config);
		type = Algo.AUTOPARTCL;
	}
	
	private HashSet<TIntHashSet> part2InternalLoop (HashSet<TIntHashSet> selectedFragments_prev, HashSet<TIntHashSet> atomicFragments, boolean k) {
		HashSet<TIntHashSet> candidateFragments = new HashSet<>();
		
		/*First step: We create all variables that our kernel will need*/
		Integer smallArraySize = atomicFragments.stream().map(it->it.size()).collect(Collectors.summingInt(Integer::intValue));
		int smallArray[]= new int[smallArraySize];
		int it2Array[]= new int[smallArraySize];
		
		Integer smallArrayNumElements = atomicFragments.size();
		int smallArraySizes[]= new int[smallArrayNumElements];
		
		
		Integer bigArraySize = selectedFragments_prev.stream().map(it->it.size()).collect(Collectors.summingInt(Integer::intValue));
		int bigArray[]= new int[bigArraySize];
		int itArray[]= new int[bigArraySize];
		
		Integer bigArrayNumElements =selectedFragments_prev.size();
		int bigArraySizes[]= new int[bigArrayNumElements];
		
		int currentElement=0;
		int currentPos=0;
		for (TIntHashSet AF : atomicFragments) {
			int previousPos=currentPos;
			TIntIterator it = AF.iterator();
			while (it.hasNext()){
				smallArray[currentPos]=it.next();
				it2Array[currentPos]=currentElement;
				currentPos++;
			}
			smallArraySizes[currentElement]=previousPos-currentPos;
			currentElement++;
		}
		
		currentElement=0;
		currentPos=0;
		for (TIntHashSet AF : selectedFragments_prev) {
			int previousPos=currentPos;
			TIntIterator it = AF.iterator();
			while (it.hasNext()){
				bigArray[currentPos]=it.next();
				itArray[currentPos]=currentElement;
				currentPos++;
			}
			bigArraySizes[currentElement]=previousPos-currentPos;
			currentElement++;
		}
		
		int outputArraySize=0;
		for (int i=0; i<bigArraySizes.length; i++){
			outputArraySize+=(bigArraySizes[i]*smallArrayNumElements)+smallArraySize;
		}
		int outputArray[] = new int[outputArraySize];
		int bitMask[] = new int[smallArrayNumElements*bigArrayNumElements];
		
		/*Second step: We pass the data to the kernel and get for all cases except for k==true*/
		
		//We still should pass the k flag to the kernel
		//TODO: SECOND: PASS TO KERNEL AND GET FROM IT
		///START COPIED CODE
		
	        Pointer src = Pointer.to(is);
	        Pointer srcBitmask = Pointer.to(bitMask);
	        Pointer srcPosList = Pointer.to(posList);
	        Pointer dst = Pointer.to(dstArray);
	        Pointer srcOrderPos = Pointer.to(orderPos);
	        Pointer src2 = Pointer.to(is2Array);
	        
	        int[] src2S = new int [1];
	        src2S[0]= is2ArraySize;
	        Pointer src2Size = Pointer.to(src2S);
	        
	        //int[] count = new int [1];
	        //count[0]= counter;
	        //Pointer countm = Pointer.to(count);/
	        
	        int[] size_n = new int [1];
	        size_n[0]=counter; 
	        Pointer sizen = Pointer.to(size_n);


	        // The platform, device type and device number
	        // that will be used
	        final int platformIndex = 0;
	        final long deviceType = CL_DEVICE_TYPE_ALL;
	        final int deviceIndex = 0;
	        cl_context context;
	        cl_program program;
	        cl_command_queue commandQueue;
	        
	        // Enable exceptions and subsequently omit error checks in this sample
	        CL.setExceptionsEnabled(true);

	        // Obtain the number of platforms
	        int numPlatformsArray[] = new int[1];
	        clGetPlatformIDs(0, null, numPlatformsArray);
	        int numPlatforms = numPlatformsArray[0];

	        // Obtain a platform ID
	        cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
	        clGetPlatformIDs(platforms.length, platforms, null);
	        cl_platform_id platform = platforms[platformIndex];

	        // Initialize the context properties
	        cl_context_properties contextProperties = new cl_context_properties();
	        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);
	        
	        // Obtain the number of devices for the platform
	        int numDevicesArray[] = new int[1];
	        clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
	        int numDevices = numDevicesArray[0];
	        
	        // Obtain a device ID 
	        cl_device_id devices[] = new cl_device_id[numDevices];
	        clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
	        cl_device_id device = devices[deviceIndex];

	        // Create a context for the selected device
	        context = clCreateContext(
	            contextProperties, 1, new cl_device_id[]{device}, 
	            null, null, null);
	        
	        // Create a command-queue for the selected device
	        commandQueue = 
	            clCreateCommandQueue(context, device, 0, null);


	        // Allocate the memory objects for the input- and output data
	        
	        
	        cl_mem memObjects[] = new cl_mem[8];
	        memObjects[0] = clCreateBuffer(context, 
	            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	            Sizeof.cl_int * is.length, src, null); 
	        memObjects[1] = clCreateBuffer(context, 
	            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
	            Sizeof.cl_int * counter, srcBitmask, null);
	        memObjects[2] = clCreateBuffer(context, 
	        	CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  
	            Sizeof.cl_int*counter, srcPosList, null);
	        memObjects[3] = clCreateBuffer(context, 
	        		CL_MEM_READ_WRITE,   
	                Sizeof.cl_int*counter, null, null);
	        memObjects[4] = clCreateBuffer(context, 
	        		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  
	                Sizeof.cl_int, sizen, null);
	        memObjects[5] = clCreateBuffer(context, 
	        		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  
	                Sizeof.cl_int*counter, srcOrderPos, null);
	        memObjects[6] = clCreateBuffer(context, 
	        		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  
	                Sizeof.cl_int*is2ArraySize, src2, null);
	        memObjects[7] = clCreateBuffer(context, 
	        		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  
	                Sizeof.cl_int, src2Size, null);
	        // Create the program from the source code
	        program = clCreateProgramWithSource(context,
	            1, new String[]{ programSource2 }, null, null);
	        
	        // Build the program
	        clBuildProgram(program, 0, null, null, null, null);
	        
	        
	        // Create the kernel
	        cl_kernel kernel = clCreateKernel(program, "merger", null);
	        
	        // Set the arguments for the kernel
	        clSetKernelArg(kernel, 0, 
	            Sizeof.cl_mem, Pointer.to(memObjects[0]));
	        clSetKernelArg(kernel, 1, 
	            Sizeof.cl_mem, Pointer.to(memObjects[1]));
	        clSetKernelArg(kernel, 2, 
	            Sizeof.cl_mem, Pointer.to(memObjects[2]));
	        clSetKernelArg(kernel, 3, 
	                Sizeof.cl_mem, Pointer.to(memObjects[3]));
	        clSetKernelArg(kernel, 4, 
	                Sizeof.cl_mem, Pointer.to(memObjects[4]));
	        clSetKernelArg(kernel, 5, 
	                Sizeof.cl_mem, Pointer.to(memObjects[5]));
	        clSetKernelArg(kernel, 6, 
	                Sizeof.cl_mem, Pointer.to(memObjects[6]));
	        clSetKernelArg(kernel, 7, 
	                Sizeof.cl_mem, Pointer.to(memObjects[7]));
	        
	        // Set the work-item dimensions
	        long global_work_size[] = new long[]{counter};
	        long local_work_size[] = new long[]{1};
	        
	        // Execute the kernel
	        clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
	            global_work_size, local_work_size, 0, null, null);
	        
	        // Read the output data
	        clEnqueueReadBuffer(commandQueue, memObjects[3], CL_TRUE, 0,
	            counter* Sizeof.cl_int, dst, 0, null, null);
	        
	        // Release kernel, program, and memory objects
	        clReleaseMemObject(memObjects[0]);
	        clReleaseMemObject(memObjects[1]);
	        clReleaseMemObject(memObjects[2]);
	        clReleaseMemObject(memObjects[3]);
	        clReleaseMemObject(memObjects[4]);
	        clReleaseMemObject(memObjects[5]);
	        clReleaseMemObject(memObjects[6]);
	        clReleaseMemObject(memObjects[7]);
	        clReleaseKernel(kernel);
	        clReleaseProgram(program);
	        clReleaseCommandQueue(commandQueue);
	        clReleaseContext(context);
	        Runtime r = Runtime.getRuntime();
	        r.gc();
	        return dstArray;
	        
		///END COPIED CODE
		
		//TODO: THIRD: WRITE KERNEL
		//The kernel in turn will have a thread per it, it will create all the combinations with the small array, and write to the output...
		//Next everything that was written to the output will be checked for query extent, and the bitmask will be set based on this.
		//	We still need to figure out how to pass this to the kernel: if (queryExtent(fragment) >= queryExtentThreshold) {//Number of queries that reference all the attributes inside a fragment
		// candidateFragments.add(fragment);//Note, no repetitions of fragments, since it is a set of sets...
	
		/*Third step: We get the data from the kernel into the candidate set, and we return*/
		currentPos=0;
		int posinSmallArray=0;
		int posinBigArray=0;
		TIntHashSet fragment = new TIntHashSet();
		for (int i=0; i<bitMask.length; i++){
			if (bitMask[i]==1){//We should copy the candidate...
				fragment.clear();
				for (int j=0; j<smallArraySizes[posinSmallArray]+bigArraySizes[posinBigArray]; j++){
					if (outputArray[currentPos+j]!=-1){
						fragment.add(outputArray[currentPos+j]);
					}
				}
				candidateFragments.add(fragment);
			}
			currentPos+=smallArraySizes[posinSmallArray]+bigArraySizes[posinBigArray];
			posinSmallArray++;
			if (posinSmallArray>=smallArraySize){
				posinSmallArray=0;
				posinBigArray++;
			}
		}
		
		for (TIntHashSet CF : selectedFragments_prev) { //This part can be parallelized.
			if (k) {
				for (TIntHashSet F : selectedFragments_prev) {
					TIntHashSet fragment = new TIntHashSet(CF);
					fragment.addAll(F);
					if (queryExtent(fragment) >= queryExtentThreshold) {//Number of queries that reference all the attributes inside a fragment
						candidateFragments.add(fragment);
					}
				}
			}
		}
			
		return candidateFragments;
	}
	
	@Override
	public void doPartition()  {

        TIntHashSet unReferenced = WorkloadUtils.getNonReferencedAttributes(w.usageMatrix);
        HashSet<TIntHashSet> unRefHashSet = new HashSet<TIntHashSet>();
        unRefHashSet.add(unReferenced);
        int unReferencedSize = getOverlappingPartitionsSize(unRefHashSet);
        /* Atomic fragment selection.*/ 

		HashSet<TIntHashSet> atomicFragments = new HashSet<TIntHashSet>();
		HashSet<TIntHashSet> newFragments = new HashSet<TIntHashSet>();
		HashSet<TIntHashSet> toBeRemovedFragments = new HashSet<TIntHashSet>();
		/*
		Part 1
		  Here we iterate over all queries and for each we get the attributes that the query uses, as a set called QueryExtent.
		  Next, for each previously considered atomic fragments, we create as a set the intersection of attributes with the current query, and the remainders are also added to a new fragments object.
		  */ 
		long init = System.nanoTime();
		for (int q = 0; q < w.queryCount; q++) {
			TIntHashSet queryExtent = new TIntHashSet(w.attributeCount);

			for (int a = 0; a < w.attributeCount; a++) {
				if (w.usageMatrix[q][a] == 1) {
					queryExtent.add(a);
				}
			}

			newFragments.clear();
			toBeRemovedFragments.clear();

			for (TIntHashSet fragment : atomicFragments) {

				TIntHashSet intersection = new TIntHashSet(queryExtent);
				intersection.retainAll(fragment);

				if (!intersection.isEmpty()) {

					toBeRemovedFragments.add(fragment);
					TIntHashSet remainder = new TIntHashSet(fragment);
					remainder.removeAll(intersection);

					if (!remainder.isEmpty()) {
						newFragments.add(remainder);
					}

					if (!intersection.isEmpty()) {
						newFragments.add(intersection);
					}

					queryExtent.removeAll(intersection);

					if (queryExtent.isEmpty()) {
						break;
					}
				}

			}

			if (!queryExtent.isEmpty()) {
				newFragments.add(queryExtent);
			}

			atomicFragments.removeAll(toBeRemovedFragments);
			atomicFragments.addAll(newFragments);
		}
		long now = System.nanoTime();

		/*
		 Iteration phase 

		 The partitions in the current solution.  */
		HashSet<TIntHashSet> presentSolution = CollectionUtils.deepClone(atomicFragments);
		/*
		 * The fragments selected for inclusion into the solution in the
		 * previous iteration. */
		 
		HashSet<TIntHashSet> selectedFragments_prev = new HashSet<TIntHashSet>();
		/*
		 * The fragments selected for inclusion into the solution in the current
		 * iteration.*/
		 
		HashSet<TIntHashSet> selectedFragments_curr = CollectionUtils.deepClone(atomicFragments);
		/*
		 * The fragments that will be considered for inclusion into the solution
		 * in the current iteration.*/
		 
		HashSet<TIntHashSet> candidateFragments = new HashSet<TIntHashSet>();

		// Iteration count. 
		int k = 0;

		boolean stoppingCondition = false;
/*
		Part 2-
		  We consider all possible combinations of both fragments and  add it to the atomic fragments
		 */
		while (!stoppingCondition) {
			
			k++;

			// composite fragment generation 

			candidateFragments.clear();
			selectedFragments_prev.clear();
			selectedFragments_prev.addAll(selectedFragments_curr);

			
			candidateFragments.addAll(this.part2InternalLoop(selectedFragments_prev, atomicFragments, k>1));
			

		//	 candidate fragment selection 

			selectedFragments_curr.clear();
			boolean solutionFound = true;

			double presentCost = costCalculator
					.findPartitionsCost(PartitioningUtils.getPartitioningMap(presentSolution));
			double bestCost = presentCost;
			HashSet<TIntHashSet> bestSolution = presentSolution;
			TIntHashSet selectedFragment = null;
			init = System.nanoTime();
			while (solutionFound) {

				solutionFound = false;

				for (TIntHashSet candidate : candidateFragments) {

					if (presentSolution.contains(candidate)) {//This could be done in parallel, but might not have a big impact.
						continue;
					}

					HashSet<TIntHashSet> newSolution = CollectionUtils.deepClone(presentSolution);
					newSolution = addFragment(newSolution, candidate);

                    if (getOverlappingPartitionsSize(newSolution) + unReferencedSize <= (1 + replicationFactor) * w.rowSize) {

						presentCost = costCalculator.findPartitionsCost(PartitioningUtils
								.getPartitioningMap(newSolution));

						//System.out.println(newSolution + " - " + presentCost + " / " + bestCost);

						if (presentCost < bestCost) {
							bestCost = presentCost;
							bestSolution = newSolution;
							selectedFragment = candidate;

							solutionFound = true;
						}
					}
				}

				if (solutionFound) {
					presentSolution = bestSolution;
					selectedFragments_curr.add(selectedFragment);
					candidateFragments.remove(selectedFragment);
				}
			}

			// update stoppingCondition
			stoppingCondition = selectedFragments_curr.size() == 0;
		}
		now = System.nanoTime();
		//System.out.println("Part 2 duration..."+(now-init));
		profiler.numberOfIterations = k;

		partitions = PartitioningUtils.getPartitioningMap(presentSolution);

//		 pairwise merge phase 
		
		stoppingCondition = false;

	//	Part 3- Pairwise merge
		double bestCost = costCalculator.findPartitionsCost(PartitioningUtils.getPartitioningMap(presentSolution));
		int bestI = 0, bestJ = 0; // the indexes of the to-be merged fragments

		//	 just a utility representation of the solution 
		TIntObjectHashMap<TIntHashSet> partitionsMap;

		//init = System.nanoTime();
		while (!stoppingCondition) {
			stoppingCondition = true;
			//partitionsMap = CollectionUtils.deepClone(partitions);
            partitionsMap = PartitioningUtils.getPartitioningMap(presentSolution);

			HashSet<TIntHashSet> modifiedSolution = null;

			for (int i = 1; i <= partitionsMap.size(); i++) {
				for (int j = i + 1; j <= partitionsMap.size(); j++) {

					modifiedSolution = new HashSet<TIntHashSet>(presentSolution);
					modifiedSolution.remove(partitionsMap.get(i));
					modifiedSolution.remove(partitionsMap.get(j));
					TIntHashSet mergedIJ = new TIntHashSet(w.attributeCount);
					mergedIJ.addAll(partitionsMap.get(i));
					mergedIJ.addAll(partitionsMap.get(j));
					modifiedSolution.add(mergedIJ);

					double presentCost = costCalculator.findPartitionsCost(PartitioningUtils
							.getPartitioningMap(modifiedSolution));

					if (presentCost < bestCost) {
						bestCost = presentCost;

						bestI = i;
						bestJ = j;

						stoppingCondition = false;
					}
				}
			}

			if (!stoppingCondition) {
				presentSolution.remove(partitionsMap.get(bestI));
				presentSolution.remove(partitionsMap.get(bestJ));
				TIntHashSet mergedIJ = new TIntHashSet(w.attributeCount);
				mergedIJ.addAll(partitionsMap.get(bestI));
				mergedIJ.addAll(partitionsMap.get(bestJ));
				presentSolution.add(mergedIJ);
			}
		}
		//now = System.nanoTime();
		//System.out.println(" Part 3 Duration..."+(now-init));

        if (unReferenced.size() > 0) {
            presentSolution.add(unReferenced);
        }
		partitions = PartitioningUtils.getPartitioningMap(presentSolution);
        costCalculator.findPartitionsCost(partitions);

        bestSolutions = workload.getBestSolutions();

       //  We reduce the partition IDs by 1 and therefore the values in the best solutions as well. 
        TIntObjectHashMap<TIntHashSet> newPartitions = new TIntObjectHashMap<TIntHashSet>();
        TIntObjectHashMap<TIntHashSet> newBestSolutions = new TIntObjectHashMap<TIntHashSet>();
        
        for (int p : partitions.keys()) {
            newPartitions.put(p - 1, partitions.get(p));
        }
        
        for (int q : bestSolutions.keys()) {
            newBestSolutions.put(q, new TIntHashSet());
            for (int p : bestSolutions.get(q).toArray()) {
                newBestSolutions.get(q).add(p - 1);
            }
        }

        partitions = newPartitions;
        bestSolutions = newBestSolutions;
	}

	/**
	 * Method for determining the query extent of a fragment, that is the
	 * cardinality of the set of queries that reference all of the attributes in
	 * a fragment.
	 * 
	 * @param fragment
	 *            The input.
	 * @return The cardinality of the fragment's query extent.
	 */
	private int queryExtent(TIntSet fragment) {
		int size = 0;

		for (int q = 0; q < w.queryCount; q++) {
			boolean referencesAll = true;

			for (TIntIterator it = fragment.iterator(); it.hasNext(); ) {
				if (w.usageMatrix[q][it.next()] == 0) {
					referencesAll = false;
				}
			}

			if (referencesAll) {
				size++;
			}
		}

		return size;
	}

	/**
	 * Method for adding a fragment to a partitioning with removing any of the
	 * subsets of the fragment from the partitioning. Note that this method does
	 * not clone the input partitioning, therefore it returns the modified input
	 * instead of a cloned one.
	 * 
	 * @param partitioning
	 *            The partitioning to be extended.
	 * @param fragment
	 *            The partition to be added.
	 * @return The modified partitioning.
	 */
	private HashSet<TIntHashSet> addFragment(HashSet<TIntHashSet> partitioning, TIntHashSet fragment) {

		HashSet<TIntHashSet> toBeRemoved = new HashSet<TIntHashSet>();

		for (TIntHashSet F1 : partitioning) {
			boolean subset = true;
			for (TIntIterator it = F1.iterator(); it.hasNext(); ) {
				if (!fragment.contains(it.next())) {
					subset = false;
					break;
				}
			}

			if (subset) {
				toBeRemoved.add(F1);
			}
		}

		partitioning.removeAll(toBeRemoved);
		partitioning.add(fragment);

		return partitioning;
	}

	/**
	 * Method for calculating the row size of the partitioned table considering
	 * overlaps, too.
	 * 
	 * @param partitions
	 *            The set of possibly overlapping partitions.
	 * @return The calculated row size.
	 */
	private int getOverlappingPartitionsSize(HashSet<TIntHashSet> partitions) {
		int size = 0;

		for (TIntHashSet partition : partitions) {
			for (TIntIterator it = partition.iterator(); it.hasNext(); ) {
				size += w.attributeSizes[it.next()];
			}
		}

		return size;
	}

    /**
     * Method for calculating the row size of the partitioned table considering
     * overlaps, too.
     *
     * @param partitions
     *            The set of possibly overlapping partitions.
     * @return The calculated row size.
     */
    public int getOverlappingPartitionsSize(TIntObjectHashMap<TIntHashSet> partitions) {
        int size = 0;

        for (TIntHashSet partition : partitions.valueCollection()) {
            for (TIntIterator it = partition.iterator(); it.hasNext(); ) {
                size += w.attributeSizes[it.next()];
            }
        }

        return size;
    }

	public double getReplicationFactor() {
		return replicationFactor;
	}

	public void setReplicationFactor(double replicationFactor) {
		this.replicationFactor = replicationFactor;
	}

	public int getQueryExtentThreshold() {
		return queryExtentThreshold;
	}

	public void setQueryExtentThreshold(int queryExtentThreshold) {
		this.queryExtentThreshold = queryExtentThreshold;
	}
    
	public static void main (String[] args) {
		 String[] queries = {"A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"};
	        Set<AbstractAlgorithm.Algo> algos_sel = new HashSet<AbstractAlgorithm.Algo>();
//	        AbstractAlgorithm.Algo[] ALL_ALGOS_SEL = {AUTOPART, HILLCLIMB, HYRISE};
//			  AbstractAlgorithm.Algo[] ALL_ALGOS_SEL = {TROJAN};
	        AbstractAlgorithm.Algo[] ALL_ALGOS_SEL = {AbstractAlgorithm.Algo.AUTOPARTCL, AbstractAlgorithm.Algo.AUTOPART};
	        for (AbstractAlgorithm.Algo algo : ALL_ALGOS_SEL) {
	            algos_sel.add(algo);
	        }
	        AlgorithmRunner algoRunner = new AlgorithmRunner(algos_sel, 10, queries, new AbstractAlgorithm.HDDAlgorithmConfig(BenchmarkTables.randomTable(1, 1)));
	        algoRunner.runTPC_H_All();
	        String output = AlgorithmResults.exportResults(algoRunner.results);

	        System.out.println(output);
	    }
	}
