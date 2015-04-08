#include <wb.h> 

#define BLOCK_SIZE 32

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

	int nbGangs = (inputLength-1)/BLOCK_SIZE + 1;
	
	wbLog(TRACE, "Run ", nbGangs, " gangs of ", BLOCK_SIZE, " workers");
	
	#pragma acc parallel loop copyin(hostInput1[0:inputLength]) copyin(hostInput2[0:inputLength]) copyout(hostOutput[0:inputLength])  num_gangs(nbGangs) num_workers(BLOCK_SIZE)
	for( int cpt = 0; cpt < inputLength; cpt++ )
		hostOutput[cpt] = hostInput1[cpt] + hostInput2[cpt];

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
