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
import java.util.Arrays;

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
import org.apache.commons.lang3.ArrayUtils;

import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

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
public class AutoPartCL extends AbstractPartitionsAlgorithm {

	/**
	 * The amount of storage available for attribute replication expressed as a
	 * factor of increase in storage requirements.
	 */
	private double replicationFactor = 0.5;

	/** The minimal number of queries that should access a candidate fragment. */
	private int queryExtentThreshold = 1;

	public long totalOptTime = 0;
	
	public AutoPartCL(AlgorithmConfig config) {
		super(config);
		type = Algo.AUTOPARTCL;
	}
	
	@Override
	public void doPartition() {


        TIntHashSet unReferenced = WorkloadUtils.getNonReferencedAttributes(w.usageMatrix);
        HashSet<TIntHashSet> unRefHashSet = new HashSet<TIntHashSet>();
        unRefHashSet.add(unReferenced);
        int unReferencedSize = getOverlappingPartitionsSize(unRefHashSet);

		/* Atomic fragment selection. */

		HashSet<TIntHashSet> atomicFragments = new HashSet<TIntHashSet>();

		HashSet<TIntHashSet> newFragments = new HashSet<TIntHashSet>();
		HashSet<TIntHashSet> toBeRemovedFragments = new HashSet<TIntHashSet>();

		
		for (int q = 0; q < w.queryCount; q++) {
			TIntHashSet queryExtent = new TIntHashSet(w.attributeCount);

			for (int a = 0; a < w.attributeCount; a++) {
				if (w.usageMatrix[q][a] == 1) {
					queryExtent.add(a);
				}
			}

			newFragments.clear();
			toBeRemovedFragments.clear();

			List<TIntHashSet> intersections = new ArrayList<>();
			List<TIntHashSet> fragments = new ArrayList<>();
			
			for (TIntHashSet fragment : atomicFragments) {

				TIntHashSet intersection = new TIntHashSet(queryExtent);
				intersection.retainAll(fragment);//Everything in both fragment and query
				if (!intersection.isEmpty()) {
					intersections.add(intersection);
					fragments.add(fragment);
				}
			}
			
			toBeRemovedFragments.addAll(fragments);
			for (int u=0; u<intersections.size(); u++) {
					TIntHashSet remainder = new TIntHashSet(fragments.get(u));
					remainder.removeAll(intersections.get(u));

					if (!remainder.isEmpty()) {
						newFragments.add(remainder);
					}

					if (!intersections.get(u).isEmpty()) {
						newFragments.add(intersections.get(u));
					}

					queryExtent.removeAll(intersections.get(u));

					if (queryExtent.isEmpty()) {
						break;
					}
			}

			if (!queryExtent.isEmpty()) {
				newFragments.add(queryExtent);
			}

			atomicFragments.removeAll(toBeRemovedFragments);
			atomicFragments.addAll(newFragments);
		}
		
		/* Iteration phase */

		/* The partitions in the current solution. */
		HashSet<TIntHashSet> presentSolution = CollectionUtils.deepClone(atomicFragments);
		/*
		 * The fragments selected for inclusion into the solution in the
		 * previous iteration.
		 */
		HashSet<TIntHashSet> selectedFragments_prev = new HashSet<TIntHashSet>();
		/*
		 * The fragments selected for inclusion into the solution in the current
		 * iteration.
		 */
		HashSet<TIntHashSet> selectedFragments_curr = CollectionUtils.deepClone(atomicFragments);
		/*
		 * The fragments that will be considered for inclusion into the solution
		 * in the current iteration.
		 */
		HashSet<TIntHashSet> candidateFragments = new HashSet<TIntHashSet>();

		/* Iteration count. */
		int k = 0;

		boolean stoppingCondition = false;

		//int iteration=0;
		while (!stoppingCondition) {
			//System.out.println("IT: "+iteration+", Start, SF: "+selectedFragments_curr.size());
			k++;

			/* composite fragment generation */

			candidateFragments.clear();
			selectedFragments_prev.clear();
			selectedFragments_prev.addAll(selectedFragments_curr);

			for (TIntHashSet CF : selectedFragments_prev) {

				// with atomic fragments
				for (TIntHashSet AF : atomicFragments) {
					TIntHashSet fragment = new TIntHashSet(CF);
					fragment.addAll(AF);

					if (queryExtent(fragment) >= queryExtentThreshold) {
						candidateFragments.add(fragment);
					}
				}

				// with fragments selected in the previous iteration
				if (k > 1) {
					for (TIntHashSet F : selectedFragments_prev) {
						TIntHashSet fragment = new TIntHashSet(CF);
						fragment.addAll(F);

						if (queryExtent(fragment) >= queryExtentThreshold) {
							candidateFragments.add(fragment);
						}
						
					}
				}
				
			}
			//We start by defining the size of the target array
			int totalSize = candidateFragments.stream().map(frag->frag.size()).mapToInt(it->it.intValue()).sum();
			int candidateArray[] = new int[totalSize];
			int candidatePosArray[] = new int[candidateFragments.size()+1];

			selectedFragments_curr.clear();
			boolean solutionFound = true;

			/* candidate fragment selection */

			double presentCost = costCalculator
					.findPartitionsCost(PartitioningUtils.getPartitioningMap(presentSolution));
			double bestCost = presentCost;
			HashSet<TIntHashSet> bestSolution = presentSolution;
			TIntHashSet selectedFragment = null;
			int startPos = 0;
			candidatePosArray[0]=0;
			int currentCandidate=0;
			for (TIntHashSet candidate : candidateFragments) {
				candidatePosArray[currentCandidate+1]=candidatePosArray[currentCandidate]+candidate.size();
				System.arraycopy( candidate.toArray(), 0, candidateArray, candidatePosArray[currentCandidate], candidate.size());
				currentCandidate++;
			}
			
			while (solutionFound) {

				solutionFound = false;
				
				long init = System.nanoTime();
				//THIS IS WHAT WE WILL CHANGE... Note that we decided to do it all in one kernel, for simplicity.
				
				//Now we need to send to the GPU the following things: candidateArray, candidatePosArray,presentSolution			//Also: an arrayWithTheOutputs that will be newSolutions and newCosts (new Solutions is tricky)
				//Also: replicationFactor, w.rowSize, unReferencedSize
				//We need to implement inside the following functionality: addFragment, getOverlappingPArtitionsSize, and the costCalculation
					//In the same kernel you will take the candidates that pass and...
					//Create a newSolution based on presentSolution, add the candidate fragment to it
					//Evaluate criteria
					//Get cost for those that pass
					//Return the costs and newSolution
				//Outside of the array determine which candidate was the best and use its newSolution and costs as the best.
				
				
				 double replfactorarray[] = new double [1];
				 
				 
				int dstArray[] = new int[];
				Pointer candidatearray = Pointer.to(candidateArray);
		        Pointer candidateposarray = Pointer.to(candidatePosArray);
		        int[] presentSolutionArray = ArrayUtils.toPrimitive(Arrays.copyOf(presentSolution.toArray(), presentSolution.size(), Integer[].class));
		        Pointer presentsol = Pointer.to(presentSolutionArray);
		        Pointer replfactorarra = Pointer.to(replfactorarray);
		        Pointer rowsize = Pointer.to(w.rowSize);
		        
		        Pointer dst = Pointer.to(dstArray);
		       
		        

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
		        
		        
		        cl_mem memObjects[] = new cl_mem[3];
		        memObjects[0] = clCreateBuffer(context, 
		            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		            Sizeof.cl_int * candidateArray.length, candidateArray, null); 
		        
		        memObjects[1] = clCreateBuffer(context, 
		            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
		            Sizeof.cl_int * candidatePosArray.length, candidatePosArray, null);
		        
		        memObjects[2] = clCreateBuffer(context, 
			            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			            Sizeof.cl_int * presentSolutionArray.length, presentSolutionArray, null);
		        		
		        memObjects[3] = clCreateBuffer(context, 
		        		CL_MEM_READ_WRITE,   
		                Sizeof.cl_int*counter, null, null);
		       

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
		        
		        // Set the work-item dimensions
		        long global_work_size[] = new long[]{candidateArray.length};
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
		        clReleaseKernel(kernel);
		        clReleaseProgram(program);
		        clReleaseCommandQueue(commandQueue);
		        clReleaseContext(context);
		        Runtime r = Runtime.getRuntime();
		        r.gc();
		        
		        
				for (TIntHashSet candidate : candidateFragments) {
					
//					System.out.println(" our candidate " + candidate + " / " + "candidateFragments " + candidateFragments);

					if (presentSolution.contains(candidate)) {
						continue;
					}

					HashSet<TIntHashSet> newSolution = CollectionUtils.deepClone(presentSolution);
					newSolution = addFragment(newSolution, candidate);

                    if (getOverlappingPartitionsSize(newSolution) + unReferencedSize <= (1 + replicationFactor) * w.rowSize) {//We call this criteria

						presentCost = costCalculator.findPartitionsCost(PartitioningUtils //cost
								.getPartitioningMap(newSolution));

						//System.out.println(newSolution + " - " + presentCost + " / " + bestCost);

						if (presentCost < bestCost) {
							bestCost = presentCost;
							bestSolution = newSolution;
							selectedFragment = candidate;

							solutionFound = true;
						}
					}
                    //As a result we should have 2 arrays, one with 0s and 1s indicating if a solution was found, the other with the values found.
                    //Then we get the minimum from the values found, and that is our bestSolution, the position of the candidate is the selectedfragment, and then we go to the next step...
				}
				//UNTIL HERE...
				
				long now = System.nanoTime();			
				totalOptTime+=now-init;

				
				if (solutionFound) {
					presentSolution = bestSolution;
					selectedFragments_curr.add(selectedFragment);
					candidateFragments.remove(selectedFragment);
				}
			}


			// update stoppingCondition
			stoppingCondition = selectedFragments_curr.size() == 0;
			//System.out.println("IT: "+iteration+", End, SF: "+selectedFragments_curr.size());
			//iteration++;

		}

		profiler.numberOfIterations = k;

		partitions = PartitioningUtils.getPartitioningMap(presentSolution);

		/* pairwise merge phase */

		stoppingCondition = false;

		double bestCost = costCalculator.findPartitionsCost(PartitioningUtils.getPartitioningMap(presentSolution));
		int bestI = 0, bestJ = 0; // the indexes of the to-be merged fragments

		/* just a utility representation of the solution */
		TIntObjectHashMap<TIntHashSet> partitionsMap;

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

        if (unReferenced.size() > 0) {
            presentSolution.add(unReferenced);
        }
		partitions = PartitioningUtils.getPartitioningMap(presentSolution);
        costCalculator.findPartitionsCost(partitions);

        bestSolutions = workload.getBestSolutions();

        /* We reduce the partition IDs by 1 and therefore the values in the best solutions as well. */
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
       // System.out.println("Total Opt Time:"+totalOptTime);

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
		 /*String[] queries = {"A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"};
	        Set<AbstractAlgorithm.Algo> algos_sel = new HashSet<AbstractAlgorithm.Algo>();
//	        AbstractAlgorithm.Algo[] ALL_ALGOS_SEL = {AUTOPART, HILLCLIMB, HYRISE};
//			  AbstractAlgorithm.Algo[] ALL_ALGOS_SEL = {TROJAN};
	        AbstractAlgorithm.Algo[] ALL_ALGOS_SEL = {AbstractAlgorithm.Algo.AUTOPARTCL};
	        for (AbstractAlgorithm.Algo algo : ALL_ALGOS_SEL) {
	            algos_sel.add(algo);
	        }
	        AlgorithmRunner algoRunner = new AlgorithmRunner(algos_sel, 10, queries, new AbstractAlgorithm.HDDAlgorithmConfig(BenchmarkTables.randomTable(1, 1)));
	        algoRunner.runTPC_H_All();
	        String output = AlgorithmResults.exportResults(algoRunner.results);

	        System.out.println(output);*/
		int [] oldArray= {0, 1, 2, 3};
		int [] newArray= {1,2};
		System.arraycopy( newArray, 0, oldArray, 2, 2);
		for (int i=0; i<oldArray.length; i++) {
			System.out.println(oldArray[i]);
		}
	    }
	}


