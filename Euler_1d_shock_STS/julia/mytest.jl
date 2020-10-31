module my_module

#include("file1.jl")
#include("file2.jl")

function my_module_test_print()
  println("Ciaooooooo!")
end

function addme(a,b)
  return a+b
end
end
