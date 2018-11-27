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
		     +"			if (id<=(*n)*(*m)){"
		     + "			output[id] = 1;"
		     + "			int count = (int) floor((double)id/(*m));"
		     + "            for(int i=from[count]; i<to[count]; i++){"
		    // + "			printf(\"Comparing %d and %d, boolean: %d in %d \\n\", bigArray[id%*m], smallArray[i], bigArray[id%*m]==smallArray[i], id);"
		     + "		    if (bigArray[id%*m]==smallArray[i])output[id] = 0;"
		     +"			}}"	   	
		     + "    }";
	
	private static String programSource2 =      		
			   					
				 " kernel void merger(__global const int* src, __global const int* srcBitmask, __global const int* srcPoslist,  __global int* output, __global const int* n, __global const int* srcOrderPos, __global const int* src2, __global const int* n2)"
			     + "    {"
			     + "        int id= (int)get_global_id(0);"
			     +"			if (id<(int)(*n)){"
			     + "		if (srcBitmask[id]==1) {"
			     + "        output[srcOrderPos[id]] = src[srcPoslist[id]];"
			     + "        }"
			     + "        else if(srcBitmask[id]==-1){"
			     + "         output[srcOrderPos[id]] = src2[srcPoslist[id]];"
			     + "        } "
			     + "        }"
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
//		System.out.println("Cost: "+candCost);
		
		double minCost;
		List<int[][]> candList = new ArrayList<int[][]>();
		int[][] R;
		int candNum= 0;
		//int[] s;
		
		do {
			R = cand;
			System.out.println("Length of R"+R.length);
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
				if (!tempCandidateList.isEmpty()){
					partialResults = doMerge(R[i], tempCandidateList);//We already parallelized the candidate generation.
					for (int j=0; j<partialResults.size(); j++){
						for (int k=0; k<partialResults.get(j).length; k++){
							System.out.println("PR["+j+"]["+k+"] is "+partialResults.get(j)[k]);
						}
					}
					//partial result is our s in actual code
					int counter = 0;
					for (int j = i + 1; j < R.length; j++) {//We still need to double-check this, to see if we are missing out some parallelization
						cand = new int[R.length-1][];
						for(int k = 0; k < R.length; k++) {
							if(k == i) {
								cand[k] = partialResults.get(counter);
								System.out.println("CANDK");
								for (int y=0; y<cand[k].length; y++){
									System.out.println(cand[k][y]);
								}
								counter++;
							} else if(k < j) {
								cand[k] = R[k];
							} else if(k > j) {
								cand[k-1] = R[k];							
							}
						}
						candList.add(cand);
					}
					System.out.println("PR Size"+partialResults.size());
					System.out.println("Counter"+counter);
					for(int num2=0; num2<R[i].length; num2++) {
			            System.out.println("R["+i+"]["+num2+"] is "+R[i][num2]);
			        }
				}

			}
			if(!candList.isEmpty()) {
				System.out.println("*******");
				for (int f=0; f<candList.size(); f++){
					System.out.println(f);
					System.out.println("-----");
					for(int num=0; num<candList.get(f).length; num++) {
				        for(int num2=0; num2<candList.get(f)[num].length; num2++) {
				            System.out.println("Values at arr["+num+"]["+num2+"] is "+candList.get(f)[num][num2]);
				        }
				    }
				}
				System.out.println("*******");
				cand = getLowerCostCand(candList);//This we could parallelize, but we believe that this list will be small.
				candCost = getCandCost(cand);
			}
		System.out.println("candCost"+candCost);
		System.out.println("minCost"+minCost);
		for(int num=0; num<cand.length; num++) {
	        for(int num2=0; num2<cand[num].length; num2++) {
	            System.out.println("Values at arr["+num+"]["+num2+"] is "+cand[num][num2]);
	        }
	    }
		candNum++;
		} while (candCost < minCost 
				//&& candNum<2
				);
		System.out.println("Length of R"+R.length);
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

	private static int [] getResult(int counter, int[] is, int[] bitMask, int [] posList, int [] orderPos, int outputCounter, int[] is2Array, int is2ArraySize) {

		//Here we create the dstArray
        int dstArray[] = new int[outputCounter];
        
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
        final int deviceIndex = 1;
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
	}
	
	private static int [] doComparison(int[] is, List<int[]> is2) {
		//System.out.println(is[0]);
		//System.out.println(is2.size());
		//Here we concatenate them into a single array, in OpenCL
        List<Integer> counts = is2.stream().map(it->it.length).collect(Collectors.toList());
        int dstArray[] = new int[is.length * is2.size()];
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
        //System.out.println("Pos Array- "+posString);
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
        
        
        
       System.out.println("Printing First Comparison");
       for (int i=0; i<is.length*counts.size(); i++) {
       		System.out.println(dstArray[i]);
       }
       System.out.println("Output"+dstArray.length);
       return dstArray;
	}
	
	@SuppressWarnings("null")
	public static List<int[]> doMerge(int[] is, List<int[]> is2) {
		
//		System.out.println(is[0]);
//		System.out.println(is2.size());
		int prefixResults [] = new int [is.length*is2.size()];
		prefixResults = doComparison(is, is2); //Prefix results is an array of 0s and 1s (which is the bitmask of is, repeated is2.size() times). 
	
/*		System.out.println("Printing prefix results");
		for (int i=0; i<prefixResults.length; i++) {
			System.out.println(prefixResults[i]);
		}*/
		
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
		int outputCounter2=0;
		int[] outputpos= new int[posListSize];
		int is2ArraySize = is2.stream().map(it->it.length).reduce(0, Integer::sum);
		int[] is2Array = new int[is2ArraySize];
		int [] posTracker = new int [is2.size()+1];//TODO: Somehow here we should use posTracker to keep the intervals for each merged partition
		int itemCounter = 0;
		for ( int [] item: is2) {	
			//posTracker[itemCounter]=outputCounter+outputCounter2;
			itemCounter++;
			for (int i=0; i<item.length; i++) {
				is2Array[is2Pos]=item[i];
				bitmask[outputCounter]=-1;
				posList[outputCounter]=is2Pos;
				outputpos[outputCounter]=outputCounter2;
				is2Pos++;
				outputCounter++;
				outputCounter2++;
			}
			for (int i=0; i<is.length; i++) {
				posList[outputCounter]=i;//is2Pos;
				bitmask[outputCounter]=prefixResults[isPos];
				if (prefixResults[isPos]==1) {
						//is2Pos++;
						outputpos[outputCounter]=outputCounter2;
						outputCounter2++;
				}
				else{
					outputpos[outputCounter]=-1;
				}
				outputCounter++;
				isPos++;
			}
			for (int i=0; i<bitmask.length; i++) {
				if( bitmask[i] != -1 && bitmask[i+1] == -1){
					posTracker[itemCounter] = i;
					itemCounter++;
				}
			  }
		}
		//posTracker[is2.size()]=outputCounter;
			
		System.out.println("Pos List");
		for (int k=0; k<posList.length; k++){
			System.out.println(posList[k]);
		}
		System.out.println("End pos List");
		
		List<int[]> results = new ArrayList<>();
		boolean runWithGPU = Boolean.TRUE;
		if (!runWithGPU){
			int lastSeen = -1;
			is2Pos = 0;
			isPos=0;
			int is2InternalPos = 0;
			List<Integer> tempResults = new ArrayList<>();
			for(int i=0; i<bitmask.length; i++ ) {
							
				if (bitmask[i] == -1){
					if (lastSeen!=-1){
						results.add(tempResults.stream().mapToInt(it->it).toArray());
						tempResults.clear();
						is2Pos++;
						is2InternalPos=0;
					}
					tempResults.add(is2.get(is2Pos)[is2InternalPos]);
					is2InternalPos++;
				}
				else {
					if (lastSeen==-1){
						isPos=0;
					}
					if (bitmask[i]==1){
						tempResults.add(is[isPos]);
					}
					isPos++;
				}
				lastSeen=bitmask[i];
			}
			results.add(tempResults.stream().mapToInt(it->it).toArray());
		}
		else{

			
			int[] totalResults = getResult(outputCounter, is, bitmask, posList, outputpos, outputCounter2, is2Array, is2ArraySize);
			System.out.println(totalResults.length);
			//TODO: Here we need to copy each array based on the intervals given in the previous function, and then add it to a list (results), and return this list.
		}
				
		/*for (int[] res: results){
			String values="";
			for (int l=0; l<res.length; l++) {
				values+=res[l]+" ";
			}
//			System.out.println("output :"+values);
			}*/
		return results; 
		//int[] results= getResult(outputCounter, outputarray, prefixResults, posList);
		//System.out.println(results[0]);
		
	}
	
	private double getCandCost(int[][] cand) {
		double sum = 0;
        sum = costCalculator.getPartitionsCost(cand);
        /*
		for (int[] item : cand) {
			sum += costCalculator.costForPartition(item);
			System.out.println(Arrays.toString(item));
			sum += costTable.get(Arrays.toString(item));
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
	
	public static void main(String[] args) {
		
		int[] a = {0, 1, 3, 7};
		int[] b = {4};
		int[] c = {7};//, 8, 9, 10};
		List<int[]> example = new ArrayList<>();
		example.add(c);
		example.add(b);
		//example.add(c);
//		int[] is, List<int[]> is2
		List<int[]> result = doMerge(a, example);
		//for (int[] res: result){
			for (int l=0; l<result.size(); l++) {
				String values="";
				for (int m=0; m<result.get(l).length; m++){
					values+=result.get(l)[m]+" ";
				}
				System.out.println(values);
			}
			}
		}
	
