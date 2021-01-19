#include <fdeep/fdeep.hpp>
int main()
{
    const auto model = fdeep::load_model("fdeep_model.json");

    const auto result = model.predict({
        fdeep::tensor(fdeep::tensor_shape(static_cast<std::size_t>(1)),
            std::vector<float>{5486})});

    std::cout << fdeep::show_tensors(result) << std::endl;
}
