#include "fdeep.hpp"
#include <iostream>
using namespace std;

fdeep::fdeep() {
  cout << "C++ side, constructor" << endl;
}

fdeep::~fdeep() {
  cout << "C++ side, destructor" << endl;
}

int fdeep::fdeep_load_model
    const auto model = fdeep::load_model("example_model.json", true, fdeep::dev_null_logger);
    const auto result = model.predict({fdeep::tensor(fdeep::tensor_shape(static_cast<std::size_t>(4)),fdeep::float_vec{1, 2, 3, 4})});
    const auto result_vec = result.front().to_vector();


//#include "fdeep.hpp"
//
////#include <ostream>
////#include <string>
////#include <fstream>
////#include <iterator>
////#include <vector>
////#include <cstdio>
////#include <cmath>
////#include <cstdlib>
//#include <iostream>
//
//#include <fplus/fplus.hpp>
//
//using namespace std;
//
//int main()
//{
//    const auto model = fdeep::load_model("example_model.json");
//    //const auto model = fdeep::load_model("example_model.json", true, fdeep::dev_null_logger);
//    const auto result = model.predict({fdeep::tensor(fdeep::tensor_shape(static_cast<std::size_t>(4)),fdeep::float_vec{1, 2, 3, 4})});
//    const auto result_vec = result.front().to_vector();
//    std::cout << fdeep::show_tensors(result) << std::endl;
//    std::cout << fplus::show_cont(result_vec) << std::endl;
//    for (auto i = result_vec.begin(); i != result_vec.end(); ++i)
//    std::cout << *i << ' ';
//}
