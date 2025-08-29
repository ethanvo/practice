#include <gemmplan.h>
#include <iostream>

static std::vector<int> indices_of_chars_in_string(const std::string& str, const char* chars) {
  // For each char in chars, find its index in src (must exist exactly once).
  std::vector<int> res;
  res.reserve(chars.size());
  for (char c : chars) {
    size_t pos = src.find(c);
    if (pos == std::string::npos) {
      throw std::invalid_argument(std::string("Index character '") + c + "' not found in source string '" + src + "'");
    }
    // ensure unique occurence

  }
}
