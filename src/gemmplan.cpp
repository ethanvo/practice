#include <gemmplan.h>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>

static std::vector<int> indices_of_chars_in_string(const std::string &src, const std::string &chars) {
  // For each char in chars, find its index in src (must exist exactly once).
  std::vector<int> res;
  res.reserve(chars.size());
  for (char c : chars) {
    size_t pos = src.find(c);
    if (pos == std::string::npos) {
      throw std::invalid_argument(std::string("Index character '") + c + "' not found in source string '" + src + "'");
    }
    // ensure unique occurence
    if (src.find(c, pos + 1) != std::string::npos) {
      throw std::invalid_argument(std::string("Index character '") + c + "' occurs multiple times in source string '" + src + "'");
    }
    res.push_back(int(pos));
  }
return res;
}

GemmPlan make_gemm_plan(const EinsumParsed &p) {
  if(p.inputs.size() != 2) {
    throw std::invalid_argument("GEMM planner only supports exactly two input tensors");
  }
  const std::string &Aidx = p.inputs[0];
  const std::string &Bidx = p.inputs[1];
  const std::string &Out = p.output;

  // Validate single-character indices and uniqueness within each input
  auto validate_unique_chars = [](const std::string &s, const std::string &name) {
    std::unordered_set<char> seen;
    for (char c : s) {
      if (c == ' ' || c == '\t' || c == '\n') continue;
      if (seen.count(c)) {
        throw std::invalid_argument(name + ": repeated index character found: " + std::string(1, c));
      }
      seen.insert(c);
    }
  };
  validate_unique_chars(Aidx, "A indices");
  validate_unique_chars(Bidx, "B indices");

  // Build sets for quick tests
  std::unordered_set<char> setA(Aidx.begin(), Aidx.end());
  std::unordered_set<char> setB(Bidx.begin(), Bidx.end());
  std::unordered_set<char> setOut(Out.begin(), Out.end());
  
  // Contracted indices K: appear in both A and B but NOT in output
  std::string K;
  for (char c : Aidx) {
    if (setB.count(c) && !setOut.count(c)) {
      K.push_back(c);
    }
  }
  // If there are characters present in B and A but not yet included (to preserve ordering from B's perspective)
  // we don't need to add more because K is defined by intersection. The ordering we choose matters for memory layout:
  // we'll keep K ordered as they appear in A followed by any in B not in A (but intersection by definition is in both).
  // So our K above is sufficient (order by A's appearance).

  // I (free indices from A that appear in output)
  std::string I;
  for (char c : Aidx) {
    if (setOut.count(c) && !setB.count(c)) { // appears in output and not contracted with Bidx
      I.push_back(c);
    } else if (setOut.count(c) && setB.count(c)) {
      // could be free on both if also in output; but if in both and in output it's a free index shared by both:
      // in typical einsum this would be ambiguous (it means same index present as free on both sides => broadcast),
      // but for GEMM mapping we prefer to treat index appearing in both inputs AND in output as belonging to both I and J.
      // We'll treat this as appearing in I (A side); the J construction below will detect it too.
      I.push_back(c);
    }
  }

  // J (free indices from B that appear in output but not in A-only I)
  std::string J;
  for (char c : Bidx) {
    if (setOut.count(c) && !setA.count(c)) {
      J.push_back(c);
    } else if (setOut.count(c) && setA.count(c)) {
      // If index is in both and in output, it already appears in I in our ordering; for result we need ordering of I then J
      // We will not duplicate it in J; duplicate free labels across both tensors should be represented once in the output.
      // So skip adding to J here.
      continue;
    }
  }
  // Edge-case: there are indices that appear in both A and B and also appear in output.
  // Those should appear in the output exactly once; we already placed them in I (in A's order).
  // If you prefer ordering based on output string, we will later permute result to match output.

  // Validate: All output indices must be present in at least one input
  for (char c : Out) {
    if (!setA.count(c) && !setB.count(c)) {
      throw std::invalid_argument(std::string("Ouput index '") + c + "' does not appear in any input");
    }
  }

  // Now build permutation vectors for A and B
  // A's desired axis order: first all I ( in the order weput them), then all K (in the order we put them)
  std::string desiredA = I + K;
  // For indices present in A but neither in desiredA (rare - e.g. indices in A that are neither contracted nor in output),
  // they represent implicit reductions (shouldn't happen normally). We'll append them to K to ensure completeness.
  for (char c : Aidx) {
    if (desiredA.find(c) == std::string::npos) {
      desiredA.push_back(c);
    }
  }
  std::vector<int> permA = indices_of_chars_in_string(Aidx, desiredA);

  // B's desired axis order: first all K, then all J
  std::string desiredB = K + J;
  for (char c : Bidx) {
    if (desiredB.find(c) == std::string::npos) {
      desiredB.push_back(c);
    }
  }
  std::vector<int> permB = indices_of_chars_in_string(Bidx, desiredB);

  // The result after GEMM will have axes order [I..., J...]. But the einsum output may have different ordering.
  // permResult maps from [I..., J...] -> output order. We produce the vector of positions such that:
  // result_axes_after_gemm[permResult[pos]] == output[pos]
  std::string result_axes = I + J; // current order after GEMM reshape
  // But note: if there are indices that were in output that were present in both A and B, they are in I already and not J.
  // Now build permResult: for each char in output, find its index in result_axes.
  std::vector<int> permResult;
  permResult.reserve(Out.size());
  for (char c : Out) {
    size_t pos = result_axes.find(c);
    if (pos == std::string::npos) {
      // If it's not in result_axes (maybe because output contains an index that's only in B and we put it in J),
      // try to find it in J (it should be there). But result_axes is I+J so it must be present if logic above was correct.
      throw std::runtime_error(std::string("Internal error building permResult: '") + c + "' not found in concatenated result axes '" + result_axes + "'");
    }
    permResult.push_back(int(pos));
  }

  GemmPlan plan;
  plan.I = I;
  plan.J = J;
  plan.K = K;
  plan.permA = permA;
  plan.permB = permB;
  plan.permResult = permResult;
  return plan;
}

// Example printing helper
void print_plan(const GemmPlan &p) {
  std::cout << "I (A free) = \"" << p.I << "\"\n";
  std::cout << "J (B free) = \"" << p.J << "\"\n";
  std::cout << "K (contract) = \"" << p.K << "\"\n";
  std::cout << "permA (to [I,K]) = [";
  for (size_t i=0;i<p.permA.size();i++) { if(i) std::cout << ", "; std::cout << p.permA[i]; }
  std::cout << "]\n";
  std::cout << "permB (to [K,J]) = [";
  for (size_t i=0;i<p.permB.size();i++) { if(i) std::cout << ", "; std::cout << p.permB[i]; }
  std::cout << "]\n";
  std::cout << "permResult (from [I,J] -> output order) = [";
  for (size_t i=0;i<p.permResult.size();i++) { if(i) std::cout << ", "; std::cout << p.permResult[i]; }
  std::cout << "]\n";
}
