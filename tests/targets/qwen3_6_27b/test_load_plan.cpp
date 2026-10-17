// Modified for the RTX 5080 Laptop fork; see CHANGELOG.md.
#include "artifact/binder.h"
#include "artifact/reader.h"
#include "artifact_fixture.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"
#include "targets/qwen3_6_27b/impl/variant.h"

#include <ninfer/targets/qwen3_6_27b/package.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace {

using ninfer::artifact::NumericFormat;
using ninfer::targets::qwen3_6_27b::Package;
using namespace ninfer::targets::qwen3_6_27b::detail;
using ninfer::targets::qwen3_6::ProposalHeadStorage;

std::filesystem::path artifact_path(const char* environment, const char* filename) {
    if (const char* value = std::getenv(environment); value != nullptr && *value != '\0') {
        return value;
    }
    return std::filesystem::path(NINFER_SOURCE_DIR) / "out" / filename;
}

ninfer::targets::qwen3_6::StartupFeatures all_features() {
    return {
        .vision        = true,
        .speculative   = ninfer::SpeculativeBackend::Mtp,
        .proposal_head = ninfer::ProposalHead::Optimized,
    };
}

ninfer::targets::qwen3_6::StartupFeatures
features(ninfer::SpeculativeBackend backend,
         ninfer::ProposalHead proposal_head = ninfer::ProposalHead::Full) {
    return {
        .vision        = false,
        .speculative   = backend,
        .proposal_head = proposal_head,
    };
}

bool is_device_object(const ninfer::artifact::MaterializationPlan& plan,
                      ninfer::artifact::ObjectHandle handle) {
    return std::ranges::any_of(plan.device_objects, [handle](const auto& object) {
        return object.object.index == handle.index;
    });
}

bool is_host_mapped_object(const ninfer::artifact::MaterializationPlan& plan,
                           ninfer::artifact::ObjectHandle handle) {
    return std::ranges::any_of(plan.host_mapped_objects, [handle](const auto& object) {
        return object.object.index == handle.index;
    });
}

std::uint64_t host_mapped_bytes(const ninfer::artifact::MaterializationPlan& plan,
                                ninfer::artifact::ObjectHandle handle) {
    for (const auto& object : plan.host_mapped_objects) {
        if (object.object.index == handle.index) { return object.bytes; }
    }
    return 0;
}

std::size_t dflash2_device_objects(const ninfer::artifact::Reader& reader,
                                   const ninfer::artifact::MaterializationPlan& plan) {
    return static_cast<std::size_t>(
        std::ranges::count_if(plan.device_objects, [&](const auto& item) {
            return ninfer::artifact::object_name(reader.objects().at(item.object.index))
                .starts_with("dflash2/");
        }));
}

bool valid_divisors(const WeightPlan& weight) {
    if (weight.format != NumericFormat::NVFP4) { return false; }
    const float weight_divisor = std::bit_cast<float>(weight.weight_scale_divisor_bits);
    const float input_divisor  = std::bit_cast<float>(weight.input_scale_divisor_bits);
    return std::isfinite(weight_divisor) && weight_divisor > 0.0F && std::isfinite(input_divisor) &&
           input_divisor > 0.0F;
}

int verify_indexed_proposal_exact_proof() {
    using namespace ninfer::test::artifact_fixture;
    constexpr std::array<std::uint64_t, 2> full_shape{7, 130};
    constexpr std::array<std::uint64_t, 2> draft_shape{3, 130};
    constexpr std::array<std::uint32_t, 3> ids{6, 0, 4};
    const auto full = ninfer::artifact::row_split_geometry(NumericFormat::Q4G64_F16S, full_shape);
    const auto draft = ninfer::artifact::row_split_geometry(NumericFormat::Q4G64_F16S, draft_shape);
    const std::uint64_t code_row_bytes = full.groups_per_row * full.low_bytes_per_group;
    const std::uint64_t scale_row_bytes = full.scale_plane_bytes / full.rows;
    std::vector<std::byte> full_bytes(full.encoded_bytes, std::byte{0});
    std::vector<std::byte> draft_bytes(draft.encoded_bytes, std::byte{0});
    std::array<std::byte, ids.size() * sizeof(std::uint32_t)> id_bytes{};
    for (std::uint64_t row = 0; row < full.rows; ++row) {
        for (std::uint64_t byte = 0; byte < code_row_bytes; ++byte) {
            full_bytes[row * code_row_bytes + byte] = std::byte((17 * row + byte) & 0xff);
        }
        for (std::uint64_t byte = 0; byte < scale_row_bytes; ++byte) {
            full_bytes[full.scale_plane_offset + row * scale_row_bytes + byte] =
                std::byte((7 * row + byte) & 0xff);
        }
    }
    for (std::size_t row = 0; row < ids.size(); ++row) {
        std::memcpy(draft_bytes.data() + row * code_row_bytes,
                    full_bytes.data() + ids[row] * code_row_bytes, code_row_bytes);
        std::memcpy(draft_bytes.data() + draft.scale_plane_offset + row * scale_row_bytes,
                    full_bytes.data() + full.scale_plane_offset + ids[row] * scale_row_bytes,
                    scale_row_bytes);
        for (unsigned byte = 0; byte < 4; ++byte) {
            id_bytes[row * 4 + byte] = std::byte((ids[row] >> (8 * byte)) & 0xff);
        }
    }
    // Q4 plane padding is not a row and need not match the full head's padding.
    draft_bytes[draft.low_plane_bytes] = std::byte{0xff};
    const std::uint64_t draft_offset = align_up(full.encoded_bytes, 256);
    const std::uint64_t ids_offset = align_up(draft_offset + draft.encoded_bytes, 256);
    const Json directory = {
        {"identity", {{"model_id", "qwen3.8-27b"}, {"weights_id", "groupwise-int-5080"}}},
        {"objects", Json::array({
                        {{"name", "text/output_head"}, {"kind", "tensor"},
                         {"shape", full_shape}, {"format", "Q4G64_F16S"},
                         {"layout", "row-split-k128-v1"}, {"offset", 0},
                         {"bytes", full.encoded_bytes}},
                        {{"name", "text/draft_head"}, {"kind", "tensor"},
                         {"shape", draft_shape}, {"format", "Q4G64_F16S"},
                         {"layout", "row-split-k128-v1"}, {"offset", draft_offset},
                         {"bytes", draft.encoded_bytes}},
                        {{"name", "text/draft_head_token_ids"}, {"kind", "tensor"},
                         {"shape", {ids.size()}}, {"format", "I32"},
                         {"layout", "contiguous-le-v1"}, {"offset", ids_offset},
                         {"bytes", id_bytes.size()}},
                    })},
    };
    const std::uint64_t payload_offset = align_up(16 + directory.dump().size(), 4096);
    for (unsigned mismatch = 0; mismatch < 3; ++mismatch) {
        auto bytes = draft_bytes;
        if (mismatch == 1) {
            bytes[draft.rows * code_row_bytes - 1] ^= std::byte{1};
        } else if (mismatch == 2) {
            bytes[draft.scale_plane_offset + draft.scale_plane_bytes - 1] ^= std::byte{1};
        }
        const auto fixture = write_fixture(directory, "indexed_proposal_" + std::to_string(mismatch));
        {
            std::fstream output(fixture.path, std::ios::binary | std::ios::in | std::ios::out);
            output.seekp(static_cast<std::streamoff>(payload_offset));
            output.write(reinterpret_cast<const char*>(full_bytes.data()), full_bytes.size());
            output.seekp(static_cast<std::streamoff>(payload_offset + draft_offset));
            output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            output.seekp(static_cast<std::streamoff>(payload_offset + ids_offset));
            output.write(reinterpret_cast<const char*>(id_bytes.data()), id_bytes.size());
            if (!output) { throw std::runtime_error("failed to write indexed proposal fixture"); }
        }
        ninfer::artifact::Reader reader(fixture.path);
        ninfer::artifact::Binder binder(reader);
        const auto output_head = binder.require_tensor(
            "text/output_head", NumericFormat::Q4G64_F16S,
            ninfer::artifact::StorageLayout::RowSplitK128V1, full_shape);
        const auto draft_head = binder.require_tensor(
            "text/draft_head", NumericFormat::Q4G64_F16S,
            ninfer::artifact::StorageLayout::RowSplitK128V1, draft_shape);
        const std::array<std::uint64_t, 1> id_shape{ids.size()};
        const auto token_ids = binder.require_tensor(
            "text/draft_head_token_ids", NumericFormat::I32,
            ninfer::artifact::StorageLayout::ContiguousLeV1, id_shape);
        try {
            validate_indexed_proposal_head(binder, output_head, draft_head, token_ids);
        } catch (const ninfer::artifact::ArtifactError& error) {
            if (mismatch != 0 &&
                std::string(error.what()).find("row 2 is not byte-identical") != std::string::npos) {
                continue;
            }
            std::cerr << "indexed proposal exact proof failed: " << error.what() << '\n';
            return 1;
        }
        if (mismatch != 0) {
            std::cerr << "indexed proposal accepted a different last-row "
                      << (mismatch == 1 ? "code" : "scale") << " byte\n";
            return 1;
        }
    }
    return 0;
}

int verify_indexed_proposal_plan(const std::filesystem::path& path) {
    ninfer::artifact::Reader reader(path);
    if (Package::resolve_weights(reader.identity()) != WeightsProfile::Qwen38GroupwiseInt5080) {
        std::cerr << "5080 artifact resolved to the wrong weights profile\n";
        return 1;
    }
    ninfer::artifact::Binder full_binder(reader);
    const ArtifactLoadPlan full = bind_artifact(
        full_binder, WeightsProfile::Qwen38GroupwiseInt5080, features(ninfer::SpeculativeBackend::Mtp));
    ninfer::artifact::Binder optimized_binder(reader);
    const ArtifactLoadPlan optimized = bind_artifact(
        optimized_binder, WeightsProfile::Qwen38GroupwiseInt5080,
        features(ninfer::SpeculativeBackend::Mtp, ninfer::ProposalHead::Optimized));
    if (full.bindings.proposal_head_storage != ProposalHeadStorage::Compact ||
        optimized.bindings.proposal_head_storage != ProposalHeadStorage::IndexedOutputRows ||
        is_device_object(optimized.materialization, optimized.bindings.draft_head) ||
        is_host_mapped_object(optimized.materialization, optimized.bindings.draft_head) ||
        !is_device_object(optimized.materialization, optimized.bindings.draft_head_token_ids) ||
        !is_device_object(optimized.materialization, optimized.bindings.output_head.object) ||
        optimized.materialization.device_objects.size() !=
            full.materialization.device_objects.size() + 1 ||
        optimized.materialization.device_capacity_bytes !=
            full.materialization.device_capacity_bytes + 131072 * sizeof(std::int32_t)) {
        std::cerr << "indexed proposal materialization must add only the 0.5 MiB token ids\n";
        return 1;
    }
    const auto& output = std::get<ninfer::artifact::TensorDescriptor>(
        reader.objects().at(optimized.bindings.output_head.object.index));
    if (output.shape != std::vector<std::uint64_t>{248320, 5120}) {
        std::cerr << "indexed proposal changed the physical output-head shape\n";
        return 1;
    }
    return 0;
}

int verify_groupwise(const std::filesystem::path& path) {
    ninfer::artifact::Reader reader(path);
    if (Package::resolve_weights(reader.identity()) != WeightsProfile::Qwen36GroupwiseInt) {
        std::cerr << "groupwise identity resolved to the wrong profile\n";
        return 1;
    }
    ninfer::artifact::Binder binder(reader);
    const ArtifactLoadPlan plan =
        bind_artifact(binder, WeightsProfile::Qwen36GroupwiseInt, all_features());
    if (plan.materialization.object_count != 1124 ||
        plan.materialization.device_objects.size() != 1118 ||
        plan.materialization.host_objects.size() != 6 ||
        plan.materialization.device_capacity_bytes == 0) {
        std::cerr << "groupwise materialization plan is incomplete\n";
        return 1;
    }
    if (plan.bindings.proposal_head_storage != ProposalHeadStorage::Compact ||
        plan.bindings.token_embedding.format != NumericFormat::Q6G64_F16S ||
        plan.bindings.output_head.format != NumericFormat::Q6G64_F16S) {
        std::cerr << "groupwise vocabulary endpoints have the wrong storage profile\n";
        return 1;
    }
    for (const TextLayerPlan& layer : plan.bindings.text_layers) {
        if (layer.is_full_attention) {
            if (!std::holds_alternative<SplitAttentionProjectionPlan>(layer.attention.projection)) {
                std::cerr << "groupwise attention parent boundary changed\n";
                return 1;
            }
        } else if (!std::holds_alternative<SplitGdnInputProjectionPlan>(
                       layer.gdn.input_projection)) {
            std::cerr << "groupwise GDN parent boundary changed\n";
            return 1;
        }
        if (layer.mlp.gate_up.format != NumericFormat::Q4G64_F16S ||
            layer.mlp.down.format != NumericFormat::Q5G64_F16S) {
            std::cerr << "groupwise MLP storage profile changed\n";
            return 1;
        }
    }
    return 0;
}

int verify_nvfp4(const std::filesystem::path& path) {
    ninfer::artifact::Reader reader(path);
    if (Package::resolve_weights(reader.identity()) != WeightsProfile::Qwen36Nvfp4) {
        std::cerr << "NVFP4 identity resolved to the wrong profile\n";
        return 1;
    }
    ninfer::artifact::Binder binder(reader);
    const ArtifactLoadPlan plan =
        bind_artifact(binder, WeightsProfile::Qwen36Nvfp4, all_features());
    if (plan.materialization.object_count != 1307 ||
        plan.materialization.device_objects.size() != 1054 ||
        plan.materialization.host_objects.size() != 6 ||
        plan.materialization.object_count - plan.materialization.device_objects.size() -
                plan.materialization.host_objects.size() !=
            247 ||
        plan.materialization.device_capacity_bytes == 0) {
        std::cerr << "NVFP4 materialization plan is incomplete: objects="
                  << plan.materialization.object_count
                  << " device=" << plan.materialization.device_objects.size()
                  << " host=" << plan.materialization.host_objects.size() << '\n';
        return 1;
    }
    if (plan.bindings.proposal_head_storage != ProposalHeadStorage::Compact ||
        plan.bindings.token_embedding.format != NumericFormat::W8G32_F16S ||
        plan.bindings.output_head.format != NumericFormat::W8G32_F16S) {
        std::cerr << "NVFP4 vocabulary endpoints have the wrong storage profile\n";
        return 1;
    }

    std::size_t nvfp4_weights          = 0;
    std::size_t bf16_attention_inputs  = 0;
    std::size_t bf16_attention_outputs = 0;
    std::size_t bf16_gdn_outputs       = 0;
    const auto count_weight            = [&](const WeightPlan& weight) {
        if (weight.format == NumericFormat::NVFP4) {
            ++nvfp4_weights;
            return valid_divisors(weight);
        }
        return true;
    };
    for (const TextLayerPlan& layer : plan.bindings.text_layers) {
        if (!count_weight(layer.mlp.gate_up) || !count_weight(layer.mlp.down)) {
            std::cerr << "NVFP4 MLP divisor is invalid\n";
            return 1;
        }
        if (layer.is_full_attention) {
            const auto* fused =
                std::get_if<FusedAttentionProjectionPlan>(&layer.attention.projection);
            if (fused == nullptr || !count_weight(fused->query_key_gate_value) ||
                !count_weight(layer.attention.output)) {
                std::cerr << "NVFP4 attention binding is invalid\n";
                return 1;
            }
            bf16_attention_inputs +=
                fused->query_key_gate_value.format == NumericFormat::BF16 ? 1 : 0;
            bf16_attention_outputs += layer.attention.output.format == NumericFormat::BF16 ? 1 : 0;
        } else {
            const auto* fused =
                std::get_if<FusedGdnInputProjectionPlan>(&layer.gdn.input_projection);
            if (fused == nullptr || !count_weight(fused->query_key_value_z) ||
                !count_weight(layer.gdn.output)) {
                std::cerr << "NVFP4 GDN binding is invalid\n";
                return 1;
            }
            bf16_gdn_outputs += layer.gdn.output.format == NumericFormat::BF16 ? 1 : 0;
        }
    }
    if (nvfp4_weights != 247 || bf16_attention_inputs != 6 || bf16_attention_outputs != 2 ||
        bf16_gdn_outputs != 1) {
        std::cerr << "NVFP4 Text inventory has the wrong storage profile: nvfp4=" << nvfp4_weights
                  << " bf16_attention_input=" << bf16_attention_inputs
                  << " bf16_attention_output=" << bf16_attention_outputs
                  << " bf16_gdn_output=" << bf16_gdn_outputs << '\n';
        return 1;
    }
    return 0;
}

int verify_embedding_host(const std::filesystem::path& path, WeightsProfile profile) {
    // Default: the token embedding table stays device-resident.
    {
        ninfer::artifact::Reader reader(path);
        ninfer::artifact::Binder binder(reader);
        const ArtifactLoadPlan plan = bind_artifact(binder, profile, all_features());
        const auto& handle = plan.bindings.token_embedding.object;
        if (!is_device_object(plan.materialization, handle) ||
            is_host_mapped_object(plan.materialization, handle)) {
            std::cerr << "default embedding placement is not device-resident: " << path << '\n';
            return 1;
        }
    }
    // --embedding-host: the table moves to pinned host memory and is read over UVA/PCIe.
    {
        ninfer::artifact::Reader reader(path);
        ninfer::artifact::Binder binder(reader);
        auto features = all_features();
        features.embedding_host = true;
        const ArtifactLoadPlan plan = bind_artifact(binder, profile, features);
        const auto& handle = plan.bindings.token_embedding.object;
        if (!is_host_mapped_object(plan.materialization, handle) ||
            is_device_object(plan.materialization, handle)) {
            std::cerr << "embedding-host placement is not host-mapped: " << path << '\n';
            return 1;
        }
        // The reported host-resident bytes must equal the embedding's encoded payload size.
        const auto* descriptor = reader.find("text/token_embedding");
        if (descriptor == nullptr ||
            !std::holds_alternative<ninfer::artifact::TensorDescriptor>(*descriptor)) {
            std::cerr << "embedding descriptor missing: " << path << '\n';
            return 1;
        }
        const auto& tensor = std::get<ninfer::artifact::TensorDescriptor>(*descriptor);
        const std::uint64_t payload =
            ninfer::artifact::tensor_encoded_size(tensor.layout, tensor.format, tensor.shape);
        const std::uint64_t reported = host_mapped_bytes(plan.materialization, handle);
        if (reported == 0 || reported != payload) {
            std::cerr << "embedding-host byte report does not match the payload: reported="
                      << reported << " payload=" << payload << " artifact=" << path << '\n';
            return 1;
        }
    }
    return 0;
}

int verify_legacy_dflash2_compatibility(const std::filesystem::path& path, WeightsProfile profile) {
    {
        ninfer::artifact::Reader reader(path);
        ninfer::artifact::Binder binder(reader);
        const ArtifactLoadPlan plan =
            bind_artifact(binder, profile, features(ninfer::SpeculativeBackend::None));
        if (plan.bindings.dflash2 || dflash2_device_objects(reader, plan.materialization) != 0) {
            std::cerr << "legacy artifact unexpectedly bound DFlash2: " << path << '\n';
            return 1;
        }
    }
    {
        ninfer::artifact::Reader reader(path);
        ninfer::artifact::Binder binder(reader);
        const ArtifactLoadPlan plan =
            bind_artifact(binder, profile, features(ninfer::SpeculativeBackend::Mtp));
        if (plan.bindings.dflash2 ||
            !is_device_object(plan.materialization, plan.bindings.mtp.input_projection.object)) {
            std::cerr << "legacy artifact did not preserve MTP-only binding: " << path << '\n';
            return 1;
        }
    }
    try {
        ninfer::artifact::Reader reader(path);
        ninfer::artifact::Binder binder(reader);
        (void)bind_artifact(binder, profile, features(ninfer::SpeculativeBackend::DFlash2));
    } catch (const ninfer::artifact::ArtifactError& error) {
        if (std::string(error.what()).find("no DFlash2 weight bundle") != std::string::npos) {
            return 0;
        }
    }
    std::cerr << "legacy artifact did not reject selected DFlash2: " << path << '\n';
    return 1;
}

int verify_dflash2_bundle(const std::filesystem::path& path, WeightsProfile profile) {
    for (const ninfer::SpeculativeBackend backend :
         {ninfer::SpeculativeBackend::None, ninfer::SpeculativeBackend::Mtp}) {
        ninfer::artifact::Reader reader(path);
        ninfer::artifact::Binder binder(reader);
        const ArtifactLoadPlan plan = bind_artifact(binder, profile, features(backend));
        if (!plan.bindings.dflash2 || dflash2_device_objects(reader, plan.materialization) != 0) {
            std::cerr << "inactive DFlash2 bundle was not validate-only: " << path << '\n';
            return 1;
        }
        const bool mtp_is_device =
            is_device_object(plan.materialization, plan.bindings.mtp.input_projection.object);
        if (mtp_is_device != (backend == ninfer::SpeculativeBackend::Mtp)) {
            std::cerr << "MTP placement does not match backend selection: " << path << '\n';
            return 1;
        }
    }

    ninfer::artifact::Reader reader(path);
    ninfer::artifact::Binder binder(reader);
    const ArtifactLoadPlan plan = bind_artifact(
        binder, profile,
        features(ninfer::SpeculativeBackend::DFlash2, ninfer::ProposalHead::Optimized));
    if (plan.bindings.proposal_head_storage != ProposalHeadStorage::Compact ||
        !plan.bindings.dflash2 || dflash2_device_objects(reader, plan.materialization) != 66 ||
        is_device_object(plan.materialization, plan.bindings.mtp.input_projection.object) ||
        !is_device_object(plan.materialization, plan.bindings.draft_head) ||
        !is_device_object(plan.materialization, plan.bindings.draft_head_token_ids)) {
        std::cerr << "selected DFlash2 bundle has the wrong placement: " << path << '\n';
        return 1;
    }
    return 0;
}

int verify_rejection() {
    try {
        (void)Package::resolve_weights({"qwen3.6-27b", "unknown"});
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        if (message.find("qwen3.6-27b/unknown") != std::string::npos) { return 0; }
    }
    std::cerr << "unknown weights identity was not rejected with the full identity\n";
    return 1;
}

int verify_profile_mismatch_rejection() {
    ninfer::DeviceContext device(0);
    ninfer::EngineOptions options;
    options.max_context    = 128;
    options.kv_capacity    = ninfer::KvCapacityPolicy::explicit_capacity(128);
    options.prefill_chunk  = 128;
    options.use_cuda_graph = false;
    auto planner =
        Package::make_sequence_planner(device, options, WeightsProfile::Qwen36GroupwiseInt);
    const std::uint32_t pages = planner.capacity_curve().minimum_main_page_groups;
    auto sequence             = std::move(planner).finalize(pages);
    RuntimeModelView empty_model;
    try {
        (void)ninfer::targets::qwen3_6::create_program<Variant>(
            empty_model, WeightsProfile::Qwen36Nvfp4, std::move(sequence), device);
    } catch (const std::invalid_argument& error) {
        if (std::string(error.what()).find("weights profile") != std::string::npos) { return 0; }
    }
    std::cerr << "mismatched load/sequence weights profiles were not rejected\n";
    return 1;
}

} // namespace

int main() {
    if (const int result = verify_indexed_proposal_exact_proof(); result != 0) { return result; }
    const std::filesystem::path qwen38_5080 =
        artifact_path("NINFER_QWEN3_8_27B_5080_WEIGHTS", "qwen3_8_27b_5080.ninfer");
    const bool has_5080 = std::filesystem::is_regular_file(qwen38_5080);
    if (has_5080) {
        if (const int result = verify_indexed_proposal_plan(qwen38_5080); result != 0) { return result; }
    }
    const std::filesystem::path groupwise =
        artifact_path("NINFER_QWEN3_6_27B_WEIGHTS", "qwen3_6_27b.ninfer");
    const std::filesystem::path nvfp4 =
        artifact_path("NINFER_QWEN3_6_27B_NVFP4_WEIGHTS", "qwen3_6_27b_nvfp4.ninfer");
    const std::filesystem::path qwen38_groupwise =
        artifact_path("NINFER_QWEN3_8_27B_WEIGHTS", "qwen3_8_27b.ninfer");
    const std::filesystem::path qwen38_nvfp4 =
        artifact_path("NINFER_QWEN3_8_27B_NVFP4_WEIGHTS", "qwen3_8_27b_nvfp4.ninfer");
    const std::filesystem::path qwen38_groupwise_dflash2 =
        artifact_path("NINFER_QWEN3_8_27B_DFLASH2_WEIGHTS", "qwen3_8_27b_dflash2.ninfer");
    const std::filesystem::path qwen38_nvfp4_dflash2 = artifact_path(
        "NINFER_QWEN3_8_27B_NVFP4_DFLASH2_WEIGHTS", "qwen3_8_27b_nvfp4_dflash2.ninfer");
    if (!std::filesystem::is_regular_file(groupwise) || !std::filesystem::is_regular_file(nvfp4)) {
        std::cerr << "skip: both real 27B artifacts are required: groupwise=" << groupwise
                  << " nvfp4=" << nvfp4 << '\n';
        return has_5080 ? 0 : 77;
    }
    if (const int result = verify_rejection(); result != 0) { return result; }
    if (const int result = verify_profile_mismatch_rejection(); result != 0) { return result; }
    if (const int result = verify_groupwise(groupwise); result != 0) { return result; }
    if (const int result = verify_nvfp4(nvfp4); result != 0) { return result; }
    if (const int result = verify_embedding_host(groupwise, WeightsProfile::Qwen36GroupwiseInt);
        result != 0) {
        return result;
    }
    if (const int result = verify_embedding_host(nvfp4, WeightsProfile::Qwen36Nvfp4); result != 0) {
        return result;
    }
    if (const int result =
            verify_legacy_dflash2_compatibility(groupwise, WeightsProfile::Qwen36GroupwiseInt);
        result != 0) {
        return result;
    }
    const std::array dflash2_artifacts{qwen38_groupwise, qwen38_nvfp4, qwen38_groupwise_dflash2,
                                       qwen38_nvfp4_dflash2};
    if (!std::ranges::all_of(dflash2_artifacts, [](const auto& path) {
            return std::filesystem::is_regular_file(path);
        })) {
        std::cerr << "skip DFlash2 binding matrix: old and new Qwen3.8 artifacts are required\n";
        return 0;
    }
    if (const int result = verify_legacy_dflash2_compatibility(qwen38_groupwise,
                                                               WeightsProfile::Qwen38GroupwiseInt);
        result != 0) {
        return result;
    }
    if (const int result =
            verify_legacy_dflash2_compatibility(qwen38_nvfp4, WeightsProfile::Qwen38Nvfp4);
        result != 0) {
        return result;
    }
    if (const int result =
            verify_dflash2_bundle(qwen38_groupwise_dflash2, WeightsProfile::Qwen38GroupwiseInt);
        result != 0) {
        return result;
    }
    if (const int result = verify_dflash2_bundle(qwen38_nvfp4_dflash2, WeightsProfile::Qwen38Nvfp4);
        result != 0) {
        return result;
    }
    return 0;
}
