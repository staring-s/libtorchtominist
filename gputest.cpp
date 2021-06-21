#undef UNICODE   //这个一定要加，否则会编译错误

#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/highgui/highgui.hpp> 
#include<opencv2\imgproc.hpp>
#include<opencv2\imgcodecs.hpp>
#include<string>
#include<vector>
#include<memory>
#include<torch\script.h>
#include<torch\torch.h>
using namespace cv;
using namespace std;

int main()
{
	
	Mat image;
	image = imread("img.jpg", IMREAD_GRAYSCALE);//读取灰度图

	torch::jit::script::Module module;
	try {
		module = torch::jit::load("model.pt");  //加载模型
	}
	catch (const c10::Error& e) {
		std::cerr << "无法加载model.pt模型\n";
		return -1;
	}

	torch::DeviceType device_type;	//设置Device类型
	device_type = torch::kCPU;	//torch::kCUDA and torch::kCPU

	torch::Device device(device_type);
	//模型转到GPU
	module.to(device);



	std::vector<int64_t> sizes = { 1, 1, image.rows, image.cols };//image.rows行,cols列
	at::TensorOptions options(at::ScalarType::Byte);
	at::Tensor tensor_image = torch::from_blob(image.data, at::IntList(sizes), options);//将opencv的图像数据转为Tensor张量数据
	tensor_image = tensor_image.toType(at::kFloat);//转为浮点型张量数据
	tensor_image = tensor_image.to(device);
	at::Tensor result = module.forward({ tensor_image }).toTensor();//推理

	auto max_result = result.max(1, true);
	auto max_index = std::get<1>(max_result).item<float>();
	std::cerr << "检测结果为：";
	std::cout << max_index << std::endl;

	waitKey(6000);


}