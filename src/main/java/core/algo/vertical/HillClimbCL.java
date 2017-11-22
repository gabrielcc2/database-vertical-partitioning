package core.algo.vertical;

/*
 * JOCL - Java bindings for OpenCL
 * 
 * Copyright 2009 Marco Hutter - http://www.jocl.org/
 */

import static org.jocl.CL.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import org.jocl.*;

import core.algo.vertical.AbstractAlgorithm.Algo;
import core.algo.vertical.AbstractAlgorithm.AlgorithmConfig;
import core.utils.ArrayUtils;
import core.utils.PartitioningUtils;
import gnu.trove.set.hash.TIntHashSet;

/**
 * A small JOCL sample.
 */
public class HillClimbCL extends AbstractPartitioningAlgorithm {
    /**
     * The source code of the OpenCL program to execute
     */
	
	private static String programSource =      		
		   
				
			 " kernel void comparer(__global const int* bigArray, __global const int* smallArray, __global int* output,  __global const int* m, __global const int* n)"
		     + "    {"
		     + "        int id= (int)get_global_id(0); "		     
		     +"			if (id<(int)(*m)){output[id] = 1;"
		     + "		for (int k=0; k<(int)(*n); k++){ if (bigArray[id]==smallArray[k]) {output[id] = 0;}}"
		     +"			}"	   
		     + "    }";
	private static String programSource_largeScale =      		
			   
	            			
			 " kernel void comparer_largeScale(__global const int* bigArray, __global const int* smallArray, __global int* output,  __global const int* m, __global const int* fromsize, __global const int* from, __global const int* to,  __global const int* n)"
		     + "    {"
		     + "        int id= (int)get_global_id(0); "		     
		     +"			if (id<(int)(*n)){"
		     + "			output[id] = 1;"
		     + "			int count = (int) floor(id/(*m));"
		     + "            for(int i=from[count]; i<to[count]; i++){"
		    // + "			printf(\"Comparing %d and %d, boolean: %d in %d \\n\", bigArray[id%*m], smallArray[i], bigArray[id%*m]==smallArray[i], id);"
		     + "		    if (bigArray[id%*m]==smallArray[i])output[id] = 0;"
		     +"			}}"	   	
		     + "    }";
	
	private static String programSource2 =      		
			   
					
				 " kernel void merger(__global const int* src, __global const int* srcBitmask, __global const int* srcPoslist,  __global int* output, __global const int* n)"
			     + "    {"
			     + "        int id= (int)get_global_id(0);"
			     +"			if (id<(int)(*n)){"
			     + "		if (srcBitmask[id]==1) {output[srcPoslist[id]] = src[id];}}"
			     +"			"	   
			     + "    }";
	
	
     /*
	 * We do not use a cost table (as in the original algorithm)
	 * because the table becomes too big for large number of 
	 * attributes (~16GB for 46 attributes). Instead, it is not very
	 * expensive to calculate the costs repeatedly.
	 */
	//	private Map<String, Double> costTable;
	
	public HillClimbCL(AlgorithmConfig config) {
		super(config);
		type = Algo.HILLCLIMBCL;

//		costTable = new HashMap<String, Double>();
	}
	
	@Override
	public void doPartition() {
//		int[][] allGroups = getSetOfGroups(usageMatrix);
//		
//		for (int[] group : allGroups) {			
//			costTable.put(Arrays.toString(group), cm.getPartitionsCost(group));
//		}
		
		int[][] cand = new int[w.attributeCount][1];
		for(int i = 0; i < w.attributeCount; i++) {
			cand[i][0] = i;
		}
		
		double candCost = getCandCost(cand);
		//System.out.println("Cost: "+candCost);
		double minCost;
		List<int[][]> candList = new ArrayList<int[][]>();
		int[][] R;
		int[] s;
		
		do {
			R = cand;
			minCost = candCost;
			candList.clear();
			for (int i = 0; i < R.length; i++) {
				//Here we form a List of R.length int[], we store the R[js] in there...
				//Then we change our doMerge so that it works on that and returns a list of candidates...
				List<int[]> tempCandidateList = new ArrayList<>();
				List<int[]> partialResults = new ArrayList<>();
				for (int j = i + 1; j < R.length; j++) { 
					tempCandidateList.add(R[j]);
					//What I need to fill here is R[j]
				}
				partialResults = doMerge(R[i], tempCandidateList);//We already parallelized the candidate generation.
				
				int counter = 0;
				for (int j = i + 1; j < R.length; j++) {//We still need to double-check this, to see if we are missing out some parallelization
					cand = new int[R.length-1][];
					for(int k = 0; k < R.length; k++) {
						if(k == i) {
							cand[k] = partialResults.get(counter);
							counter++;
						} else if(k < j) {
							cand[k] = R[k];
						} else if(k > j) {
							cand[k-1] = R[k];							
						}
					}
					candList.add(cand);
				}
			}
			
			if(!candList.isEmpty()) {
				cand = getLowerCostCand(candList);//This we could parallelize, but we believe that this list will be small.
				candCost = getCandCost(cand);
			}
		} while (candCost < minCost);

		partitioning = PartitioningUtils.getPartitioning(R);
	}
	
	private int[][] getLowerCostCand(List<int[][]> candList) {
		int indexOfLowest = 0;
		int index = 0;
		double lowestCost = Double.MAX_VALUE;
		for (int[][] cand : candList) {
			double cost = getCandCost(cand);
			if (lowestCost > cost) {
				indexOfLowest = index;
				lowestCost = cost;
			}
			index++;
		}
		return candList.get(indexOfLowest);
	}

	private static int [] getResult(int counter, int[] is, int[] bitMask, int [] posList) {

		//Here we create the dstArray
        int dstArray[] = new int[counter];
        
        Pointer src = Pointer.to(is);
        Pointer srcBitmask = Pointer.to(bitMask);
        Pointer srcPosList = Pointer.to(posList);
        Pointer dst = Pointer.to(dstArray);

        int[] count = new int [1];
        count[0]= counter;
        //Pointer countm = Pointer.to(count);/
        
        int[] size_n = new int [1];
        size_n[0]=is.length; 
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
        
        
        cl_mem memObjects[] = new cl_mem[5];
        memObjects[0] = clCreateBuffer(context, 
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            Sizeof.cl_int * is.length, src, null); 
        memObjects[1] = clCreateBuffer(context, 
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
            Sizeof.cl_int * is.length, srcBitmask, null);
        memObjects[2] = clCreateBuffer(context, 
        	CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  
            Sizeof.cl_int*is.length, srcPosList, null);
        memObjects[3] = clCreateBuffer(context, 
        		CL_MEM_READ_WRITE,   
                Sizeof.cl_int*counter, null, null);
        memObjects[4] = clCreateBuffer(context, 
        		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  
                Sizeof.cl_int, sizen, null);

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
        
        // Set the work-item dimensions
        long global_work_size[] = new long[]{is.length};
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
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
        Runtime r = Runtime.getRuntime();
        r.gc();

        return dstArray;
	}
	
	private static int [] doComparison(int[] is, List<int[]> is2) {
		//Here we concatenate them into a single array, in OpenCL
        int dstArray[] = new int[is.length * is2.size()];
        List<Integer> counts = is2.stream().map(it->it.length).collect(Collectors.toList());
        
        Integer size_is2= counts.stream().reduce(0, Integer::sum);
        
        int posArray[] = new int[counts.size()];
        int toArray[] = new int[counts.size()];
        
        toArray[0] = counts.get(0);
        
        int counter = 0;
        posArray[0] = 0;
        for (int i = 1; i<counts.size(); i++) {
        	counter+=counts.get(i-1);
        	posArray[i] = counter;
        	toArray[i] = posArray[i] + counts.get(i); 
        }
        
        //System.out.println("Printing Big Array");
        for (int i=0; i<is.length; i++) {
        	//System.out.println(is[i]);
        }
        
       // System.out.println("Printing Small Array");
        for (int i=0; i<is2.size(); i++) {
        	String temp="";
        	for (int j=0; j<is2.get(i).length; j++) {
            	temp+=" "+is2.get(i)[j];
            }
        	//System.out.println("Pos: "+i+"- "+temp);
        }
        String posString="";
        String toString="";
        for (int j = 0; j<counts.size(); j++) {
        	posString+=" "+posArray[j];
        	toString+=" "+toArray[j];
        }
       // System.out.println("Pos Array- "+posString);
        //System.out.println("To Array- "+toString);
        
        
        
        
        //counts.stream().mapToInt(i->i).toArray();       
        int newis2 [] = new int[size_is2];
        int posToAdd=0;
        for (int[] t: is2) {
        	for (int item: t) {
        		newis2[posToAdd]= item;
        		posToAdd++;
        	}
        }
        
       // System.out.println("Printing newis2");
        for (int i=0; i<newis2.length; i++) {
        	//System.out.println(newis2[i]);
        }
        
        Pointer srcA = Pointer.to(is);
        int[] size_m = new int [1];
        size_m[0]= is.length; 
        Pointer sizem = Pointer.to(size_m);
        
        int[] size_n = new int [1];
        size_n[0]= newis2.length; 
        Pointer sizen = Pointer.to(size_n);
        
        Pointer srcB = Pointer.to(newis2);
        Pointer fromArray = Pointer.to(posArray);
        int[] totalIs2 = new int [1];
        totalIs2[0]= counts.size(); 
        Pointer totaln = Pointer.to(totalIs2);
        Pointer toPositionArray = Pointer.to(toArray);
        
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
        cl_mem memObjects[] = new cl_mem[8];
        memObjects[0] = clCreateBuffer(context, 
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            Sizeof.cl_int * is.length, srcA, null); //Smallest
        
        memObjects[1] = clCreateBuffer(context, 
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, //Biggest
            Sizeof.cl_int * newis2.length, srcB, null);
        memObjects[2] = clCreateBuffer(context, 
            CL_MEM_READ_WRITE, 
            Sizeof.cl_int * is.length * (int)counts.size(), null, null);
        memObjects[3] = clCreateBuffer(context, 
        		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  
                Sizeof.cl_int, sizem, null);
        
        memObjects[4] = clCreateBuffer(context, 
        		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  
                Sizeof.cl_int, totaln, null);

        memObjects[5] = clCreateBuffer(context, 
        		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  
                Sizeof.cl_int*(int)counts.size(), fromArray, null);
        memObjects[6] = clCreateBuffer(context, 
        		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  
                Sizeof.cl_int*(int)counts.size(), toPositionArray, null);
        memObjects[7] = clCreateBuffer(context, 
        		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,  
                Sizeof.cl_int, sizen, null);
        
        // Create the program from the source code
        program = clCreateProgramWithSource(context,
            1, new String[]{ programSource_largeScale }, null, null);
        
        // Build the program
        clBuildProgram(program, 0, null, null, null, null);
        
        
        // Create the kernel
        cl_kernel kernel = clCreateKernel(program, "comparer_largeScale", null);
        
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
        long global_work_size[] = new long[]{counts.size()*is.length};
        long local_work_size[] = new long[]{1};
        
        // Execute the kernel
        clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
            global_work_size, local_work_size, 0, null, null);
        
        // Read the output data
        clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE, 0,
            is.length* (int)counts.size()*Sizeof.cl_int, dst, 0, null, null);
        
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
        
        
        
     //  System.out.println("Printing First Comparison");
       for (int i=0; i<is.length*counts.size(); i++) {
       		//System.out.println(dstArray[i]);
       }
       return dstArray;
	}
	
	@SuppressWarnings("null")
	public static List<int[]> doMerge(int[] is, List<int[]> is2) {
		
		int prefixResults [] = new int [is.length*is2.size()];
		prefixResults = doComparison(is, is2); //Prefix results is an array of 0s and 1s (which is the bitmask of is, repeated is2.size() times). 
	
//		System.out.println("Printing prefix results");
//		for (int i=0; i<prefixResults.length; i++) {
//			System.out.println(prefixResults[i]);
//		}
		
		//working here
		int smallPosList [] = new int [is.length*is2.size()]; //Prefix sum over prefix results (also the position list for using is)
		int counter = 0;
		for (int i=0; i<is.length*is2.size(); i++) {
			smallPosList[i]=counter;
			if (prefixResults[i]==1) {
				counter++;
			}
		}

		int resultSize= smallPosList[smallPosList.length-1]+1+is2.stream().map(it->it.length).reduce(0, Integer::sum);//This is the size of the output
		int posListSize= (is.length*is2.size())+is2.stream().map(it->it.length).reduce(0, Integer::sum);//This is the size of the combined bitmasks
		int posList [] = new int [posListSize];
		int bitmask [] = new int [posListSize];

		int is2Pos= 0;
		int isPos= 0;
		int[] outputarray= new int[resultSize];	
		int outputCounter=0;
		
		for ( int [] item: is2) {	
			for (int i=0; i<item.length; i++) {
				bitmask[outputCounter]=-1;
				posList[outputCounter]=is2Pos;
				is2Pos++;
				outputCounter++;
			}
			for (int i=0; i<is.length; i++) {
				posList[outputCounter]=is2Pos;
				bitmask[outputCounter]=prefixResults[isPos];
				if (prefixResults[isPos]==1) {
						is2Pos++;
				}
				outputCounter++;
				isPos++;
			}
		}
		
		int isid = 0 ;
		
		int is2listPos = 0;
		int is2internalArrayPos=0;
		List<int[]> results = new ArrayList<>();
		int startPos=0;
		for(int i=0; i<bitmask.length; i++ ) {
			if (bitmask[i] == -1) {				
				outputarray[posList[i]] =  is2.get(is2listPos)[is2internalArrayPos];
	           // System.out.println("OUTPUT OF BITMASK -1 IS :"+outputarray[posList[i]]+"\n");
				is2internalArrayPos++;

	         }
			else {
				 if (is2internalArrayPos>0) {
					 is2internalArrayPos= 0;
					 is2listPos++;
				 }
				 if (bitmask[i] == 1 ) {
					 outputarray[posList[i]] = is[isid];		 
					// System.out.println("OUTPUT OF BITMASK 1 IS :"+outputarray[posList[i]]+"\n");
				 }
				 isid++;
				 if (isid>=is.length) {
                    isid=0;
                    int[] newArray = new int[posList[i]-startPos+1];
                    for (int j= 0; j<newArray.length; j++) {
                    	System.out.println("start "+startPos);
                    	System.out.println("j "+j);
                    	newArray[j]=outputarray[startPos+j];
                    }
                    results.add(newArray);
                    startPos=posList[i]+1;
                 }
			}
               
       }//System.out.println("bitmask position:"+i+"----"+"0 ENTERED");
            	
	/*	
		System.out.println("Bitmask");
		for (int i=0; i<bitmask.length; i++) {
		System.out.println(bitmask[i]);
		}
		
		System.arraycopy(is, 0,outputarray, 0, is.length);
		System.arraycopy(is2, 0,outputarray, is.length, is2.size());
	*/	
		//System.out.println("Writing back the results");
		//System.out.println("Writing back the results" +getResult(counter, c, bitMask, posList));
		for (int[] res: results){
			String values="";
			for (int l=0; l<res.length; l++) {
				values+=res[l]+" ";
			}
			System.out.println("output :"+values);
			}
		return results; //getResult(counter, outputarray, prefixResults, posList);

	}

	private static void If(boolean b) {
		// TODO Auto-generated method stub
		
	}

	private double getCandCost(int[][] cand) {
		double sum = 0;

        sum = costCalculator.getPartitionsCost(cand);

        /*
		for (int[] item : cand) {
			sum += costCalculator.costForPartition(item);
//			System.out.println(Arrays.toString(item));
//			sum += costTable.get(Arrays.toString(item));
		} */

		return sum;
	}

//	private int[][] getSetOfGroups(int[][] usageMatrix) {
//		Map<Integer, List<Integer>> partitionAttributes = new HashMap<Integer,List<Integer>>();
//		List<Integer> attributes = new ArrayList<Integer>();
//		for(int i = 0; i < usageMatrix[0].length; i++)
//			attributes.add(i);
////		System.out.println("attrSize: "+attributes.size());
//		List<List<Integer>> psetattr = powerSetIter(attributes);
//		Collections.sort(psetattr, new ListComparator());
//		
//		int partitionCount = 0;
//		for (int p = psetattr.size()-1; p >= 0 ; p--) {
//			partitionAttributes.put(partitionCount++, psetattr.get(p));			
//		}
//				
//		int[][] primaryPartitions = new int[partitionAttributes.size()][];
//		int i = 0;
//		for(int p : partitionAttributes.keySet()){
//			List<Integer> attrs = partitionAttributes.get(p);
//			primaryPartitions[i] = new int[attrs.size()];
//			for(int j = 0; j < attrs.size(); j++)
//				primaryPartitions[i][j] = attrs.get(j);
//			i++;
//		}
//		
//		return primaryPartitions;
//	}
//
//	
//	public class ListComparator implements Comparator<List<Integer>> {
//	    @Override
//	    public int compare(List<Integer> o1, List<Integer> o2) {
//	        return o2.size()-o1.size();
//	    }
//	}
//
//	public static <T> List<List<T>> powerSetIter(Collection<T> list) {
//		List<List<T>> ps = new ArrayList<List<T>>();
//		ps.add(new ArrayList<T>()); // add the empty set
//
//		// for every item in the original list
//		for (T item : list) {
//			List<List<T>> newPs = new ArrayList<List<T>>();
//
//			for (List<T> subset : ps) {
//				// copy all of the current powerset's subsets
//				newPs.add(subset);
//
//				// plus the subsets appended with the current item
//				List<T> newSubset = new ArrayList<T>(subset);
//				newSubset.add(item);
//				newPs.add(newSubset);
//			}
//
//			// powerset is now powerset of list.subList(0, list.indexOf(item)+1)
//			ps = newPs;
//		}
//		ps.remove(new ArrayList<T>()); // remove the empty set
//		return ps;
//	}
	/*public static void main(String[] args) {
		int[] a = {0, 1, 3,6};
		int[] b = {0, 3, 4, 5};
		int[] c = {0, 7, 8, 9, 10};
		List<int[]> example = new ArrayList<>();
		example.add(b);
		example.add(c);
		   
	
		for (int k=0; k<1; k++) {
//		try {
//			Thread.sleep(5000);
//		} catch (InterruptedException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
		List<int[]> result = doMerge(a, example);
		for (int[] res: result){
			String values="";
			for (int l=0; l<res.length; l++) {
				values+=res[l]+" ";
			}
			System.out.println(values);
			}
		}
	}*/
}