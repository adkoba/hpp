#include <wb.h>
#include <amp.h>

using namespace concurrency;

template<typename DataType>
void vectorAddition( DataType *In1, DataType *In2, DataType *Out, int NbElt )
{
	array_view<DataType, 1> d_In1(NbElt, In1);
	array_view<DataType, 1> d_In2(NbElt, In2);
	array_view<DataType, 1> d_Out(NbElt, Out);
	
	d_Out.discard_data();
	
	parallel_for_each( d_Out.get_extent(), [=](index<1> i) 
		restrict(amp) {	
			d_Out[i] = d_In1[i] + d_In2[i];
		}
	);

	d_Out.synchronize();
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;

  args = wbArg_read(argc, argv);

  //wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  //wbTime_stop(Generic, "Importing data and creating memory on host");

 // wbLog(TRACE, "The input length is ", inputLength);


	vectorAddition<float>(hostInput1, hostInput2, hostOutput, inputLength);

  wbSolution(args, hostOutput, inputLength);


  return 0;
}
