#include "ninfer/engine.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: ninfer_generate_check <artifact.ninfer> [prompt]\n";
        return 2;
    }
    try {
        ninfer::EngineOptions engine_options;
        engine_options.artifact_path = argv[1];
        engine_options.max_context = 2048;
        engine_options.prefill_chunk = 128;
        engine_options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
        engine_options.enable_vision = false;
        ninfer::Engine engine(std::move(engine_options));

        ninfer::PromptInput input;
        input.options.enable_thinking = false;
        ninfer::ChatMessage message;
        message.role = "user";
        message.parts.push_back(
            {ninfer::MessagePartKind::Text,
             argc == 3 ? argv[2]
                       : "In three concise sentences, explain why the daytime sky appears blue."});
        input.messages.push_back(std::move(message));

        ninfer::RequestOptions request;
        request.execution.requested_output_tokens = 128;
        request.execution.allow_prefix_reuse = false;
        auto result = engine.generate(engine.prepare(std::move(input)), request);
        std::cout << "prompt_tokens=" << result.prompt.prompt_tokens << "\ncontent:\n"
                  << result.content << "\nreasoning:\n" << result.reasoning << "\ntoken_ids:";
        for (const auto token : result.generated_token_ids) { std::cout << ' ' << token; }
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ninfer_generate_check: " << error.what() << '\n';
        return 1;
    }
}
