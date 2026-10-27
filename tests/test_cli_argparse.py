"""aipotluck.installer.cli's argument parser.

The regression this file exists to guard: --install-dir/--system/-v must not be declared on both
the top-level parser and each subparser (via a shared `parents=[...]`): argparse's subparsers
action silently discards the outer parser's value in that setup -- `--install-dir X login` would
parse without error but `args.install_dir` would come back None, not X. See cli.py's
`_common_args_parser` docstring for the full mechanism. These tests pin the fixed behavior (flags
only recognized after the subcommand) so nobody "fixes" it back by re-adding the shared parents.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aipotluck.installer.cli import build_arg_parser


class TestInstallDirPlacement:
    def test_after_subcommand_is_honored(self):
        parser = build_arg_parser()
        args = parser.parse_args(["status", "--install-dir", "/tmp/somewhere"])
        assert args.install_dir == Path("/tmp/somewhere")

    def test_before_subcommand_is_rejected_not_silently_ignored(self):
        # The critical regression case: this must fail loudly (SystemExit from argparse's usage
        # error), never succeed with install_dir silently reset to the default.
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--install-dir", "/tmp/somewhere", "status"])

    @pytest.mark.parametrize("command", ["login", "logout", "status"])
    def test_every_subcommand_accepts_install_dir_after_itself(self, command):
        parser = build_arg_parser()
        args = parser.parse_args([command, "--install-dir", "/tmp/x"])
        assert args.install_dir == Path("/tmp/x")


class TestSystemAndVerboseFlags:
    def test_system_flag_defaults_false(self):
        parser = build_arg_parser()
        args = parser.parse_args(["status"])
        assert args.system is False
        assert args.verbose is False

    def test_system_and_verbose_after_subcommand(self):
        parser = build_arg_parser()
        args = parser.parse_args(["logout", "--system", "-v"])
        assert args.system is True
        assert args.verbose is True


class TestCommandRequired:
    def test_no_subcommand_is_an_error(self):
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_unknown_subcommand_is_an_error(self):
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["reboot"])


class TestLoginTunnelFlags:
    def test_all_three_tunnel_flags_parse_together(self):
        parser = build_arg_parser()
        args = parser.parse_args(
            ["login", "--tunnel-id", "abc", "--tunnel-secret", "def", "--tunnel-endpoint", "http://x"]
        )
        assert (args.tunnel_id, args.tunnel_secret, args.tunnel_endpoint) == ("abc", "def", "http://x")

    def test_tunnel_flags_default_to_none_for_one_click_pairing(self):
        parser = build_arg_parser()
        args = parser.parse_args(["login"])
        assert args.tunnel_id is None
        assert args.tunnel_secret is None
        assert args.tunnel_endpoint is None

    def test_server_url_parses_and_defaults_to_none(self):
        parser = build_arg_parser()
        assert parser.parse_args(["login"]).server_url is None
        args = parser.parse_args(["login", "--server-url", "http://localhost:5199"])
        assert args.server_url == "http://localhost:5199"

    def test_logout_and_status_have_no_tunnel_flags(self):
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["logout", "--tunnel-id", "abc"])
        with pytest.raises(SystemExit):
            parser.parse_args(["status", "--tunnel-id", "abc"])

    def test_credentials_file_defaults_to_none(self):
        parser = build_arg_parser()
        args = parser.parse_args(["login"])
        assert args.credentials_file is None

    def test_credentials_file_parses_to_a_path(self):
        parser = build_arg_parser()
        args = parser.parse_args(["login", "--credentials-file", "/tmp/creds.json"])
        assert args.credentials_file == Path("/tmp/creds.json")

    def test_logout_and_status_have_no_credentials_file_flag(self):
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["logout", "--credentials-file", "/tmp/x.json"])
        with pytest.raises(SystemExit):
            parser.parse_args(["status", "--credentials-file", "/tmp/x.json"])


class TestPullArgs:
    def test_model_is_required(self):
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["pull"])

    def test_model_and_default_timeout(self):
        parser = build_arg_parser()
        args = parser.parse_args(["pull", "bartowski/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M"])
        assert args.model == "bartowski/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M"
        assert args.timeout is None  # run_pull_model fills in the real default

    def test_custom_timeout(self):
        parser = build_arg_parser()
        args = parser.parse_args(["pull", "org/repo", "--timeout", "120"])
        assert args.timeout == 120.0

    def test_install_dir_after_the_model_and_subcommand(self):
        # Same placement rule as every other subcommand -- see TestInstallDirPlacement above.
        parser = build_arg_parser()
        args = parser.parse_args(["pull", "org/repo", "--install-dir", "/tmp/x"])
        assert args.install_dir == Path("/tmp/x")

    def test_install_dir_before_subcommand_is_rejected(self):
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--install-dir", "/tmp/x", "pull", "org/repo"])


class TestListArgs:
    def test_no_positional_arguments_needed(self):
        parser = build_arg_parser()
        args = parser.parse_args(["list"])
        assert args.command == "list"

    def test_install_dir_after_subcommand(self):
        parser = build_arg_parser()
        args = parser.parse_args(["list", "--install-dir", "/tmp/x"])
        assert args.install_dir == Path("/tmp/x")
