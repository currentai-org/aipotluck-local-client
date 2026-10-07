"""aipotluck.installer.model_presets -- the INI read/write half of CUR-1965's router-mode preset
generation. Real file I/O throughout (no mocks): this is exactly the kind of thing where "looks
right" and "llama.cpp's own PEG parser actually accepts it" can silently diverge.
"""

from __future__ import annotations

from pathlib import Path

from aipotluck.installer import model_presets


class TestHasPreset:
    def test_false_when_file_does_not_exist(self, tmp_path: Path):
        assert model_presets.has_preset(tmp_path / "missing.ini", "org/model:Q4_K_M") is False

    def test_false_when_file_exists_but_section_does_not(self, tmp_path: Path):
        ini_path = tmp_path / "presets.ini"
        ini_path.write_text("[other/model:Q8_0]\nctx-size = 4096\n", encoding="utf-8")
        assert model_presets.has_preset(ini_path, "org/model:Q4_K_M") is False

    def test_true_once_written(self, tmp_path: Path):
        ini_path = tmp_path / "presets.ini"
        model_presets.write_preset(ini_path, "org/model:Q4_K_M", {"ctx-size": "8192"})
        assert model_presets.has_preset(ini_path, "org/model:Q4_K_M") is True


class TestWritePreset:
    def test_creates_the_file_and_parent_directories(self, tmp_path: Path):
        ini_path = tmp_path / "nested" / "presets.ini"
        model_presets.write_preset(ini_path, "org/model:Q4_K_M", {"ctx-size": "8192"})
        assert ini_path.exists()

    def test_written_values_read_back_correctly(self, tmp_path: Path):
        ini_path = tmp_path / "presets.ini"
        model_presets.write_preset(
            ini_path, "org/model:Q4_K_M",
            {"ctx-size": "16384", "parallel": "1", "cache-type-k": "q8_0", "cache-type-v": "q8_0"},
        )
        text = ini_path.read_text(encoding="utf-8")
        assert "[org/model:Q4_K_M]" in text
        assert "ctx-size = 16384" in text
        assert "cache-type-k = q8_0" in text

    def test_a_repo_colon_quant_section_name_round_trips_through_configparser(self, tmp_path: Path):
        # llama.cpp's own preset section names are exactly this shape ("org/repo:Q4_K_M") --
        # confirm configparser (which we use for reading/writing) doesn't choke on the "/" or ":".
        ini_path = tmp_path / "presets.ini"
        model_id = "bartowski/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M"
        model_presets.write_preset(ini_path, model_id, {"ctx-size": "4096"})
        assert model_presets.has_preset(ini_path, model_id) is True

    def test_updating_an_existing_model_preserves_other_sections(self, tmp_path: Path):
        ini_path = tmp_path / "presets.ini"
        ini_path.write_text(
            "[*]\nn-gpu-layers = 8\n\n[hand/written:Q4_0]\nchat-template = chatml\n",
            encoding="utf-8",
        )
        model_presets.write_preset(ini_path, "org/model:Q4_K_M", {"ctx-size": "8192"})
        text = ini_path.read_text(encoding="utf-8")
        assert "[*]" in text and "n-gpu-layers = 8" in text
        assert "[hand/written:Q4_0]" in text and "chat-template = chatml" in text
        assert "[org/model:Q4_K_M]" in text and "ctx-size = 8192" in text

    def test_re_writing_the_same_model_updates_in_place_rather_than_duplicating(self, tmp_path: Path):
        ini_path = tmp_path / "presets.ini"
        model_presets.write_preset(ini_path, "org/model:Q4_K_M", {"ctx-size": "4096"})
        model_presets.write_preset(ini_path, "org/model:Q4_K_M", {"ctx-size": "16384"})
        text = ini_path.read_text(encoding="utf-8")
        assert text.count("[org/model:Q4_K_M]") == 1
        assert "ctx-size = 16384" in text
        assert "ctx-size = 4096" not in text

    def test_none_value_removes_the_key_rather_than_writing_a_placeholder(self, tmp_path: Path):
        ini_path = tmp_path / "presets.ini"
        model_presets.write_preset(ini_path, "org/model:Q4_K_M", {"cache-type-k": "q8_0"})
        model_presets.write_preset(ini_path, "org/model:Q4_K_M", {"cache-type-k": None})
        text = ini_path.read_text(encoding="utf-8")
        assert "cache-type-k" not in text

    def test_none_value_on_a_key_that_was_never_set_is_a_no_op(self, tmp_path: Path):
        ini_path = tmp_path / "presets.ini"
        # Must not raise even though "cache-type-k" was never written for this section.
        model_presets.write_preset(ini_path, "org/model:Q4_K_M", {"ctx-size": "4096", "cache-type-k": None})
        assert model_presets.has_preset(ini_path, "org/model:Q4_K_M")


class TestKnownModelIds:
    def test_empty_set_when_file_does_not_exist(self, tmp_path: Path):
        assert model_presets.known_model_ids(tmp_path / "missing.ini") == set()

    def test_returns_every_section_name(self, tmp_path: Path):
        ini_path = tmp_path / "presets.ini"
        model_presets.write_preset(ini_path, "org/a:Q4_K_M", {"ctx-size": "4096"})
        model_presets.write_preset(ini_path, "org/b:Q8_0", {"ctx-size": "8192"})
        assert model_presets.known_model_ids(ini_path) == {"org/a:Q4_K_M", "org/b:Q8_0"}

    def test_excludes_the_global_section(self, tmp_path: Path):
        ini_path = tmp_path / "presets.ini"
        ini_path.write_text("[*]\nn-gpu-layers = 8\n\n[org/a:Q4_K_M]\nctx-size = 4096\n", encoding="utf-8")
        assert model_presets.known_model_ids(ini_path) == {"org/a:Q4_K_M"}


class TestReadAll:
    def test_empty_dict_when_file_does_not_exist(self, tmp_path: Path):
        assert model_presets.read_all(tmp_path / "missing.ini") == {}

    def test_returns_every_managed_model_with_its_raw_key_values(self, tmp_path: Path):
        ini_path = tmp_path / "presets.ini"
        model_presets.write_preset(
            ini_path, "org/a:Q4_K_M", {"ctx-size": "16384", "parallel": "1", "cache-type-k": "q8_0"}
        )
        result = model_presets.read_all(ini_path)
        assert result == {"org/a:Q4_K_M": {"ctx-size": "16384", "parallel": "1", "cache-type-k": "q8_0"}}

    def test_excludes_the_global_section(self, tmp_path: Path):
        ini_path = tmp_path / "presets.ini"
        ini_path.write_text("[*]\nn-gpu-layers = 8\n\n[org/a:Q4_K_M]\nctx-size = 4096\n", encoding="utf-8")
        result = model_presets.read_all(ini_path)
        assert "*" not in result
        assert result == {"org/a:Q4_K_M": {"ctx-size": "4096"}}


class TestTuning:
    def test_read_tuning_is_empty_dict_when_file_does_not_exist(self, tmp_path: Path):
        ini_path = tmp_path / "presets.ini"
        assert model_presets.read_tuning(ini_path) == {}

    def test_write_then_read_round_trips(self, tmp_path: Path):
        ini_path = tmp_path / "presets.ini"
        model_presets.write_tuning(ini_path, "org/a:Q4_K_M", {"ctx_size": "largest context that fits"})
        assert model_presets.read_tuning(ini_path) == {"org/a:Q4_K_M": {"ctx_size": "largest context that fits"}}

    def test_writing_one_models_tuning_preserves_another_models(self, tmp_path: Path):
        ini_path = tmp_path / "presets.ini"
        model_presets.write_tuning(ini_path, "org/a:Q4_K_M", {"ctx_size": "reason a"})
        model_presets.write_tuning(ini_path, "org/b:Q8_0", {"ctx_size": "reason b"})
        result = model_presets.read_tuning(ini_path)
        assert result["org/a:Q4_K_M"] == {"ctx_size": "reason a"}
        assert result["org/b:Q8_0"] == {"ctx_size": "reason b"}

    def test_re_writing_the_same_model_replaces_its_tuning_rather_than_merging(self, tmp_path: Path):
        ini_path = tmp_path / "presets.ini"
        model_presets.write_tuning(ini_path, "org/a:Q4_K_M", {"ctx_size": "first reason", "parallel": "p"})
        model_presets.write_tuning(ini_path, "org/a:Q4_K_M", {"ctx_size": "second reason"})
        assert model_presets.read_tuning(ini_path)["org/a:Q4_K_M"] == {"ctx_size": "second reason"}

    def test_corrupt_tuning_file_is_treated_as_empty_not_a_crash(self, tmp_path: Path):
        ini_path = tmp_path / "presets.ini"
        ini_path.with_suffix(".tuning.json").write_text("{not valid json", encoding="utf-8")
        assert model_presets.read_tuning(ini_path) == {}

    def test_tuning_file_lives_alongside_the_ini_with_a_distinct_name(self, tmp_path: Path):
        ini_path = tmp_path / "presets.ini"
        model_presets.write_tuning(ini_path, "org/a:Q4_K_M", {"ctx_size": "reason"})
        assert (tmp_path / "presets.tuning.json").exists()
        assert not ini_path.exists()  # writing tuning must not create/touch the .ini itself


class TestRemovePreset:
    def test_drops_the_section_and_its_tuning_and_nothing_else(self, tmp_path: Path):
        ini = tmp_path / "presets.ini"
        model_presets.write_preset(ini, "org/a:Q4_K_M", {"ctx-size": "4096"})
        model_presets.write_preset(ini, "org/b:Q4_K_M", {"ctx-size": "8192"})
        model_presets.write_tuning(ini, "org/a:Q4_K_M", {"ctx_size": "why a"})
        model_presets.write_tuning(ini, "org/b:Q4_K_M", {"ctx_size": "why b"})

        assert model_presets.remove_preset(ini, "org/a:Q4_K_M") is True

        assert model_presets.read_all(ini) == {"org/b:Q4_K_M": {"ctx-size": "8192"}}
        assert model_presets.read_tuning(ini) == {"org/b:Q4_K_M": {"ctx_size": "why b"}}

    def test_returns_false_and_writes_nothing_when_there_was_no_section(self, tmp_path: Path):
        ini = tmp_path / "presets.ini"
        assert model_presets.remove_preset(ini, "org/a:Q4_K_M") is False
        assert not ini.exists()
