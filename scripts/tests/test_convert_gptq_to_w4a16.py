"""Regression test for GPTQ v1 to ARLE W4A16 conversion."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import convert_gptq_to_w4a16 as converter  # noqa: E402


def pack_gptq_weights(signed):
    unsigned = (signed + 8).to(torch.int32)
    packed = torch.zeros((signed.shape[1] // 8, signed.shape[0]), dtype=torch.int32)
    for nibble in range(8):
        packed |= unsigned[:, nibble::8].T << (nibble * 4)
    return packed


def pack_gptq_v1_zeros(zero_points):
    stored = zero_points - 1
    packed = torch.zeros((stored.shape[0], stored.shape[1] // 8), dtype=torch.int32)
    for channel in range(stored.shape[1]):
        packed[:, channel // 8] |= stored[:, channel] << ((channel % 8) * 4)
    return packed


def file_hashes(directory):
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.iterdir()
        if path.is_file()
    }


def write_fixture(root, *, two_groups=False):
    source = root / "source"
    source.mkdir()
    group_size = 128
    config = {
        "quantization_config": {
            "quant_method": "gptq",
            "gptq_version": 1,
            "bits": 4,
            "group_size": group_size,
            "sym": True,
            "desc_act": False,
        }
    }
    (source / "config.json").write_text(json.dumps(config))
    weight_map = {}
    count = 2 if two_groups else 1
    for index in range(count):
        prefix = f"model.layers.{index}.proj"
        shard = f"model-{index + 1:05d}-of-{count:05d}.safetensors"
        signed = torch.zeros((8, 128), dtype=torch.int8)
        tensors = {
            prefix + ".qweight": pack_gptq_weights(signed),
            prefix + ".qzeros": pack_gptq_v1_zeros(
                torch.full((1, 8), 8, dtype=torch.int32)
            ),
            prefix + ".scales": torch.ones((1, 8), dtype=torch.float16),
        }
        save_file(tensors, source / shard)
        weight_map.update({key: shard for key in tensors})
    (source / converter.INDEX_NAME).write_text(json.dumps({"weight_map": weight_map}))
    return source, weight_map


class GptqToW4A16Conversion(unittest.TestCase):
    def test_converts_cross_shard_group128_layout(self):
        output_channels, k, group_size = 8, 256, 128
        prefix = "model.layers.0.linear_attn.in_proj_qkv"
        signed = (
            (torch.arange(output_channels * k).reshape(output_channels, k) * 7 + 1) % 16
            - 8
        ).to(torch.int8)
        qweight = pack_gptq_weights(signed)
        qzeros = pack_gptq_v1_zeros(
            torch.full((k // group_size, output_channels), 8, dtype=torch.int32)
        )
        scales = (
            torch.arange((k // group_size) * output_channels, dtype=torch.float16)
            .reshape(k // group_size, output_channels)
            .add(0.5)
        )
        g_idx = torch.arange(k, dtype=torch.int32) // group_size
        expected_weight = (signed[:, 0::2] + 8).to(torch.uint8) | (
            (signed[:, 1::2] + 8).to(torch.uint8) << 4
        )
        old_raw = (
            qweight.contiguous().view(torch.uint8).reshape(k // 2, output_channels).T
        )
        self.assertFalse(torch.equal(old_raw, expected_weight))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            config = {
                "model_type": "qwen3_5",
                "quantization_config": {
                    "quant_method": "gptq",
                    "gptq_version": 1,
                    "bits": 4,
                    "group_size": group_size,
                    "sym": True,
                    "desc_act": False,
                },
            }
            (source / "config.json").write_text(json.dumps(config))
            (source / "tokenizer.json").write_text("{}")
            shard1 = "model-00001-of-00002.safetensors"
            shard2 = "model-00002-of-00002.safetensors"
            save_file(
                {
                    prefix + ".qweight": qweight,
                    "model.embed_tokens.weight": torch.ones(2),
                },
                source / shard1,
            )
            save_file(
                {
                    prefix + ".qzeros": qzeros,
                    prefix + ".scales": scales,
                    prefix + ".g_idx": g_idx,
                    "aux.scales": torch.tensor([2.0]),
                },
                source / shard2,
            )
            weight_map = {
                prefix + ".qweight": shard1,
                "model.embed_tokens.weight": shard1,
                prefix + ".qzeros": shard2,
                prefix + ".scales": shard2,
                prefix + ".g_idx": shard2,
                "aux.scales": shard2,
            }
            (source / converter.INDEX_NAME).write_text(
                json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map})
            )

            converter.convert_model_dir(source, output)

            index = json.loads((output / converter.INDEX_NAME).read_text())
            actual = {}
            tensors = {}
            for shard in output.glob("*.safetensors"):
                with safe_open(shard, framework="pt", device="cpu") as handle:
                    for key in handle.keys():
                        actual[key] = shard.name
                        tensors[key] = handle.get_tensor(key)
            self.assertEqual(index["weight_map"], actual)
            self.assertTrue(torch.equal(tensors[prefix + ".weight"], expected_weight))
            self.assertTrue(
                torch.equal(
                    tensors[prefix + ".weight_scale"], scales.T.to(torch.bfloat16)
                )
            )
            self.assertEqual(actual["aux.scales"], shard2)
            for suffix in converter.SIDECAR_SUFFIXES:
                self.assertNotIn(prefix + suffix, actual)
            output_config = json.loads((output / "config.json").read_text())
            self.assertEqual(
                output_config["quantization_config"],
                {
                    "quant_method": "w4a16",
                    "bits": 4,
                    "group_size": group_size,
                    "sym": True,
                    "desc_act": False,
                },
            )
            self.assertTrue((output / "tokenizer.json").exists())

    def test_rejects_escaping_shard_before_io(self):
        for malicious in ("../outside.safetensors", "/outside.safetensors"):
            with (
                self.subTest(malicious=malicious),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                source, weight_map = write_fixture(root)
                outside = root / "outside.safetensors"
                outside.write_bytes(b"unchanged")
                key = next(iter(weight_map))
                weight_map[key] = malicious
                (source / converter.INDEX_NAME).write_text(
                    json.dumps({"weight_map": weight_map})
                )
                before = outside.read_bytes()

                with mock.patch.object(
                    converter, "safe_open", side_effect=AssertionError("unexpected IO")
                ):
                    with self.assertRaisesRegex(
                        ValueError, "invalid safetensors shard"
                    ):
                        converter.convert_model_dir(source, root / "output")

                self.assertEqual(outside.read_bytes(), before)
                self.assertFalse((root / "output").exists())

    def test_rejects_indexed_tensor_missing_from_shard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, weight_map = write_fixture(root)
            weight_map["missing.tensor"] = next(iter(weight_map.values()))
            (source / converter.INDEX_NAME).write_text(
                json.dumps({"weight_map": weight_map})
            )

            with self.assertRaisesRegex(ValueError, "index tensor map mismatch"):
                converter.convert_model_dir(source, root / "output")

            self.assertFalse((root / "output").exists())

    def test_rejects_generated_name_collisions_before_staging(self):
        for cross_shard in (False, True):
            with (
                self.subTest(cross_shard=cross_shard),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                source, weight_map = write_fixture(root, two_groups=cross_shard)
                prefix = "model.layers.0.proj"
                shard = sorted(set(weight_map.values()))[int(cross_shard)]
                with safe_open(source / shard, framework="pt", device="cpu") as handle:
                    tensors = {key: handle.get_tensor(key) for key in handle.keys()}
                tensors[prefix + ".weight"] = torch.ones(1)
                save_file(tensors, source / shard)
                weight_map[prefix + ".weight"] = shard
                (source / converter.INDEX_NAME).write_text(
                    json.dumps({"weight_map": weight_map})
                )

                with mock.patch.object(
                    converter.tempfile,
                    "TemporaryDirectory",
                    side_effect=AssertionError("unexpected staging"),
                ):
                    with self.assertRaisesRegex(
                        ValueError, "generated tensor name collision"
                    ):
                        converter.convert_model_dir(source, root / "output")

    def test_rejects_broken_output_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _ = write_fixture(root)
            output = root / "output"
            output.symlink_to(root / "missing")

            with self.assertRaisesRegex(ValueError, "already exists"):
                converter.convert_model_dir(source, output)

            self.assertTrue(output.is_symlink())

    def test_atomic_rename_does_not_replace_existing_empty_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "staging"
            destination = root / "output"
            source.mkdir()
            destination.mkdir()

            with self.assertRaises(FileExistsError):
                converter._rename_noreplace(source, destination)

            self.assertTrue(source.is_dir())
            self.assertTrue(destination.is_dir())

    def test_rejects_output_nested_in_input_before_staging(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _ = write_fixture(root)
            output = source / "converted"

            with mock.patch.object(
                converter.tempfile,
                "TemporaryDirectory",
                side_effect=AssertionError("unexpected staging"),
            ):
                with self.assertRaisesRegex(ValueError, "inside input"):
                    converter.convert_model_dir(source, output)

            self.assertFalse(output.exists())

    def test_failed_second_shard_leaves_no_output_or_staging(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _ = write_fixture(root, two_groups=True)
            output = root / "output"
            before = file_hashes(source)
            real_convert = converter.convert_gptq_tensors
            calls = 0

            def fail_second(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise ValueError("second shard failed")
                return real_convert(*args, **kwargs)

            with mock.patch.object(
                converter, "convert_gptq_tensors", side_effect=fail_second
            ):
                with self.assertRaisesRegex(ValueError, "second shard failed"):
                    converter.convert_model_dir(source, output)

            self.assertFalse(output.exists())
            self.assertEqual(file_hashes(source), before)
            self.assertEqual(list(root.glob(".output.staging-*")), [])


if __name__ == "__main__":
    unittest.main()
