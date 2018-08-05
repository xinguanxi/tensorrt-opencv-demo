#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>
#include <time.h>
#include <cuda_runtime_api.h>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "common.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace nvinfer1;
using namespace nvcaffeparser1;

// stuff we know about the network and the caffe input/output blobs
static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int OUTPUT_SIZE = 1000;
static Logger gLogger;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "fc1";

static bool PairCompare(const std::pair<float, int>& lhs, const std::pair<float, int>& rhs)
{
    return lhs.first > rhs.first;
}

void caffeToGIEModel(const std::string& deployFile,				// name for caffe prototxt
					 const std::string& modelFile,				// name for model 
					 const std::vector<std::string>& outputs,   // network outputs
					 unsigned int maxBatchSize,					// batch size - NB must be at least as large as the batch we want to run with)
					 IHostMemory *&gieModelStream)    // output buffer for the GIE model
{
	// create the builder
	IBuilder* builder = createInferBuilder(gLogger);

	// parse the caffe model to populate the network, then set the outputs
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser* parser = createCaffeParser();
	const IBlobNameToTensor* blobNameToTensor = parser->parse(deployFile.c_str(),modelFile.c_str(),*network, DataType::kFLOAT);

	// specify which tensors are outputs
	for (auto& s : outputs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(16 << 20);
        clock_t t1=clock();
	ICudaEngine* engine = builder->buildCudaEngine(*network);
        clock_t t2=clock();
        std::cout<<"t2-t1: "<<t2-t1<<std::endl;
	assert(engine);

	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	parser->destroy();

	// serialize the engine, then close everything down
	gieModelStream = engine->serialize();
	engine->destroy();
	builder->destroy();
	shutdownProtobufLibrary();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
	const ICudaEngine& engine = context.getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	assert(engine.getNbBindings() == 2);
	void* buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME), 
		outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

	// create GPU buffers and a stream
	CHECK(cudaMalloc(&buffers[inputIndex], batchSize*3 * INPUT_H * INPUT_W * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize *3* INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
	context.enqueue(batchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
}


int main(int argc, char** argv)
{
	// create a GIE model from the caffe model and serialize it to a stream
        IHostMemory *gieModelStream{nullptr};
        cudaSetDevice(1);
   	caffeToGIEModel("goolge1.prototxt", "google1.caffemodel", std::vector < std::string > { OUTPUT_BLOB_NAME }, 1, gieModelStream);
        std::string label_file   = "imagenet1000.txt";

    /* Load labels. */
    std::vector<std::string> labels_;
    std::ifstream labels(label_file.c_str());
    //CHECK(labels) << "Unable to open labels file " << label_file;
    std::string line;
    while (std::getline(labels, line))
        labels_.push_back(std::string(line));
        cv::Mat frame;
        cv::VideoCapture cap(0);
        IRuntime* runtime = createInferRuntime(gLogger);
	ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), nullptr);
        if (gieModelStream) gieModelStream->destroy();

	IExecutionContext *context = engine->createExecutionContext();
        while(1)
        {
           float *data = new float[3 * 112 * 112];
           cap >> frame;
           cv::resize(frame,frame,cv::Size(112,112));
           frame.convertTo(frame, CV_32F);
           frame = (frame - 127.5) / 128;
           //imshow("normalization", cropImg);
           for (int i = 0; i < frame.rows; ++i)
           {
               for (int j = 0; j < frame.cols; ++j)
               {
                  data[0*INPUT_H * INPUT_W + i * INPUT_H + j] = static_cast<float>(frame.at<cv::Vec3b>(i,j)[0]);
                  data[1*INPUT_H * INPUT_W + i * INPUT_H + j] = static_cast<float>(frame.at<cv::Vec3b>(i,j)[1]);
                  data[2*INPUT_H * INPUT_W + i * INPUT_H + j] = static_cast<float>(frame.at<cv::Vec3b>(i,j)[2]);
               }
           }
	   // run inference
	   float prob[OUTPUT_SIZE];
	   doInference(*context, data, prob, 1);
    
           delete[] data;
   
           std::vector<float> probs;

           for (int n = 0; n < OUTPUT_SIZE; n++)
                probs.push_back(prob[n]);

           std::vector<std::pair<float, int> > pairs;
           for (size_t i = 0; i < probs.size(); ++i)
               pairs.push_back(std::make_pair(probs[i], i));
           std::partial_sort(pairs.begin(), pairs.begin() + 5, pairs.end(), PairCompare);

           std::vector<int> result;
           for (int i = 0; i < 5; ++i)
           {
               result.push_back(pairs[i].second);
               std::cout << prob[pairs[i].second] << " - " << labels_[pairs[i].second] << std::endl;
           }
           std::cout << std::endl;

          // input_channels.clear();
           probs.clear();
           pairs.clear();
           result.clear();
           cv::imshow("frame",frame);
           cv::waitKey(1);
           
        }
          context->destroy();
	  engine->destroy();
	  runtime->destroy();

	return 0;
}
