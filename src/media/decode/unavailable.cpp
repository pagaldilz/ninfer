#include "media/decode/decode.h"

#include <stdexcept>

namespace ninfer::media::decode {

namespace {
[[noreturn]] void unavailable() {
    throw std::runtime_error(
        "media decode is unavailable in this text-only Windows benchmark build");
}
} // namespace

Image decode_image(std::span<const std::uint8_t>, const Policy&) { unavailable(); }

Video decode_video(std::span<const std::uint8_t>, const Policy&, double, int, int) {
    unavailable();
}

} // namespace ninfer::media::decode
