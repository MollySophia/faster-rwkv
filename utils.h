#include "tensor.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#ifdef FR_ENABLE_ANDROID_ASSET
#include <android/asset_manager.h>
#endif

#include <check.h>

#define LOG_INFO(msg) std::cout << "[INFO] " << msg << std::endl;

inline bool file_exists(const std::string &path) {
  std::ifstream file(path);
  return file.good();
}

inline std::string read_file(const std::string &_path,
                             void *_asset_manager = nullptr) {
  if (_path.substr(0, 6) == "asset:") {
#ifdef FR_ENABLE_ANDROID_ASSET
    AAssetManager *asset_manager = static_cast<AAssetManager *>(_asset_manager);
    RV_CHECK(asset_manager != nullptr) << "Asset manager is not provided";
    const std::string path = _path.substr(6);
    AAsset *asset =
        AAssetManager_open(asset_manager, path.c_str(), AASSET_MODE_STREAMING);
    RV_CHECK(asset != nullptr) << "Asset \"" << path << "\" not found";
    std::stringstream ss;
    char buf[BUFSIZ];
    int nb_read = 0;
    while ((nb_read = AAsset_read(asset, buf, BUFSIZ)) > 0) {
      ss.write(buf, nb_read);
    }
    AAsset_close(asset);
    return ss.str();
#else
    RV_UNIMPLEMENTED() << "An asset path is specified, but Android asset is "
                          "not enabled. Please check your build settings.";
#endif
  } else {
    const std::string &path = _path;
    RV_CHECK(file_exists(path)) << "File \"" << path << "\" does not exist";
    std::ifstream file(path);
    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
  }
}

#ifdef _WIN32
inline std::vector<uint8_t> read_file_to_vector(const std::string &_path,
                             void *_asset_manager = nullptr) {
  const std::string &path = _path;
  RV_CHECK(file_exists(path)) << "File \"" << path << "\" does not exist";
  std::basic_ifstream<uint8_t> stream(path, std::ios::in | std::ios::binary);
  auto eos = std::istreambuf_iterator<uint8_t>();
  auto buffer = std::vector<uint8_t>(std::istreambuf_iterator<uint8_t>(stream), eos);
  return buffer;
}
#endif

namespace rwkv {
namespace utils {
LengthType indices_to_offset(const Shape &shape,
                             const std::vector<LengthType> &indices);

void offset_to_indices(LengthType offset, const Shape &shape,
                       std::vector<LengthType> &indices);
} // namespace utils
} // namespace rwkv
