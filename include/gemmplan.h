#include <vector>
#include <string>

struct EinsumParsed {
  std::vector<std::string> inputs;
  std::string output;
};

struct GemmPlan {
  std::string I;
  std::string J;
  std::string K;
  std::vector<int> permA;
  std::vector<int> permB;
  std::vector<int> permResult;
};

static std::vector<int> indices_of_chars_in_string(const std::string &src, const std::string &chars);

GemmPlan make_gemm_plan(const EinsumParsed &p);

void print_plan(const GemmPlan &p);
