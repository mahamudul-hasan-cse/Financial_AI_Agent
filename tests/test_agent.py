"""Tests for the financial agent definition (financial_agent.py).

Covers:
- Module-level singleton configuration
- ``create_agent()`` factory (defaults, overrides, missing key)
- ``_build_tools()`` registration & graceful degradation
- CLI entry-point (``main``, ``_interactive_loop``)
- Response-packaging via AGENT_INSTRUCTIONS content
"""

from __future__ import annotations

import logging
from unittest.mock import patch, MagicMock, call

import pytest

from agno.agent import Agent
from agno.tools import Toolkit


# ── Module-level singleton ────────────────────────────────────────────

class TestAgentConfiguration:
    """Tests for the agent's initialization and tool configuration."""

    def test_agent_is_importable(self):
        from financial_agent import agent
        assert agent is not None

    def test_agent_is_agent_instance(self):
        from financial_agent import agent
        assert isinstance(agent, Agent)

    def test_agent_has_tools(self):
        from financial_agent import agent
        assert agent.tools is not None
        assert len(agent.tools) > 0

    def test_agent_has_yfinance_tools(self):
        from financial_agent import agent
        tool_names = [type(t).__name__ for t in agent.tools]
        assert "YFinanceTools" in tool_names

    def test_agent_has_duckduckgo_tools(self):
        from financial_agent import agent
        tool_names = [type(t).__name__ for t in agent.tools]
        assert "DuckDuckGoTools" in tool_names

    def test_agent_has_chart_tools(self):
        from financial_agent import agent
        tool_names = [type(t).__name__ for t in agent.tools]
        assert "ChartTools" in tool_names

    def test_agent_has_excel_tools(self):
        from financial_agent import agent
        tool_names = [type(t).__name__ for t in agent.tools]
        assert "ExcelTools" in tool_names

    def test_agent_has_rag_tools(self):
        from financial_agent import agent
        tool_names = [type(t).__name__ for t in agent.tools]
        assert "RAGTools" in tool_names

    def test_agent_uses_groq_model(self):
        from financial_agent import agent
        model_class = type(agent.model).__name__
        assert model_class == "Groq"

    def test_agent_has_instructions(self):
        from financial_agent import agent
        assert agent.instructions is not None
        assert len(agent.instructions) > 0

    def test_agent_markdown_enabled(self):
        from financial_agent import agent
        assert agent.markdown is True

    def test_agent_show_tool_calls_enabled(self):
        from financial_agent import agent
        assert agent.show_tool_calls is True

    def test_agent_history_responses_set(self):
        from financial_agent import agent
        assert agent.num_history_responses == 3


# ── create_agent() factory ───────────────────────────────────────────

class TestCreateAgent:
    """Tests for the ``create_agent()`` factory function."""

    def test_returns_agent_instance(self):
        from financial_agent import create_agent
        a = create_agent()
        assert isinstance(a, Agent)

    def test_uses_default_model_id(self):
        from financial_agent import create_agent, DEFAULT_MODEL_ID
        a = create_agent()
        assert a.model.id == DEFAULT_MODEL_ID

    def test_custom_model_id(self):
        from financial_agent import create_agent
        custom_id = "llama-3.3-70b-versatile"
        a = create_agent(model_id=custom_id)
        assert a.model.id == custom_id

    def test_explicit_api_key_used(self):
        """Factory should accept an explicit key without needing the env var."""
        from financial_agent import create_agent
        a = create_agent(api_key="test-key-123")
        assert a.model.api_key == "test-key-123"

    def test_no_api_key_raises_runtime_error(self):
        from financial_agent import create_agent
        with patch.dict("os.environ", {}, clear=True), \
             patch("backend.app.agent.financial_agent.os.getenv", return_value=None):
            with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
                create_agent(api_key=None)

    def test_custom_tools_override(self):
        from financial_agent import create_agent
        mock_tool = MagicMock(spec=Toolkit)
        a = create_agent(tools=[mock_tool])
        assert len(a.tools) == 1
        assert a.tools[0] is mock_tool

    def test_empty_tools_override(self):
        from financial_agent import create_agent
        a = create_agent(tools=[])
        assert a.tools == []

    def test_custom_instructions_override(self):
        from financial_agent import create_agent
        custom = ["You are a test agent."]
        a = create_agent(instructions=custom)
        assert a.instructions == custom

    def test_default_instructions_match_constant(self):
        from financial_agent import create_agent, AGENT_INSTRUCTIONS
        a = create_agent()
        assert a.instructions == AGENT_INSTRUCTIONS

    def test_markdown_always_enabled(self):
        from financial_agent import create_agent
        a = create_agent()
        assert a.markdown is True

    def test_show_tool_calls_always_enabled(self):
        from financial_agent import create_agent
        a = create_agent()
        assert a.show_tool_calls is True

    def test_env_var_api_key_used_when_no_explicit(self):
        """Factory falls back to GROQ_API_KEY env var."""
        from financial_agent import create_agent
        with patch("backend.app.agent.financial_agent.os.getenv", return_value="env-key-456"):
            a = create_agent(api_key=None)
            assert a.model.api_key == "env-key-456"

    def test_explicit_key_takes_precedence_over_env(self):
        from financial_agent import create_agent
        with patch("backend.app.agent.financial_agent.os.getenv", return_value="env-key"):
            a = create_agent(api_key="explicit-key")
            assert a.model.api_key == "explicit-key"

    def test_factory_delegates_to_build_tools(self):
        """create_agent() without explicit tools should call build_tools()."""
        from financial_agent import create_agent
        with patch("backend.app.agent.financial_agent.build_tools") as mock_bt:
            mock_bt.return_value = [MagicMock(spec=Toolkit)]
            a = create_agent()
            mock_bt.assert_called_once()
            assert len(a.tools) == 1

    def test_num_history_responses_set_by_factory(self):
        from financial_agent import create_agent
        a = create_agent()
        assert a.num_history_responses == 3


# ── FinancialAgentService ────────────────────────────────────────────

class TestFinancialAgentService:
    """Tests for the ``FinancialAgentService`` wrapper class."""

    def test_service_importable(self):
        from financial_agent import FinancialAgentService
        assert FinancialAgentService is not None

    def test_service_singleton_exists(self):
        from financial_agent import agent_service, FinancialAgentService
        assert isinstance(agent_service, FinancialAgentService)

    def test_service_agent_property(self):
        from financial_agent import agent_service
        assert agent_service.agent is not None
        assert isinstance(agent_service.agent, Agent)

    def test_service_agent_matches_module_agent(self):
        from financial_agent import agent_service, agent
        assert agent_service.agent is agent

    def test_service_custom_agent(self):
        """Service should accept a pre-built agent."""
        from financial_agent import FinancialAgentService, create_agent
        custom = create_agent(model_id="llama-3.3-70b-versatile")
        svc = FinancialAgentService(agent=custom)
        assert svc.agent is custom

    def test_service_custom_max_retries(self):
        from financial_agent import FinancialAgentService, create_agent
        svc = FinancialAgentService(max_retries=5)
        assert svc._max_retries == 5

    def test_run_returns_string(self):
        from financial_agent import FinancialAgentService
        mock_agent = MagicMock()
        result = MagicMock()
        result.content = "Hello from agent"
        mock_agent.run.return_value = result

        svc = FinancialAgentService(agent=mock_agent)
        response = svc.run("test prompt")
        assert response == "Hello from agent"
        mock_agent.run.assert_called_once_with("test prompt", stream=False)

    def test_run_retries_on_failure(self):
        from financial_agent import FinancialAgentService
        mock_agent = MagicMock()
        good = MagicMock()
        good.content = "recovered"
        mock_agent.run.side_effect = [RuntimeError("fail"), good]

        svc = FinancialAgentService(agent=mock_agent, max_retries=3)
        response = svc.run("retry prompt")
        assert response == "recovered"
        assert mock_agent.run.call_count == 2

    def test_run_exhausts_retries(self):
        from financial_agent import FinancialAgentService
        mock_agent = MagicMock()
        mock_agent.run.side_effect = RuntimeError("persistent")

        svc = FinancialAgentService(agent=mock_agent, max_retries=2)
        with pytest.raises(RuntimeError, match="Agent failed after 2 attempts"):
            svc.run("fail prompt")
        assert mock_agent.run.call_count == 2

    def test_run_falls_back_to_str_if_no_content(self):
        from financial_agent import FinancialAgentService
        mock_agent = MagicMock()
        mock_agent.run.return_value = 42  # no .content attribute

        svc = FinancialAgentService(agent=mock_agent)
        response = svc.run("prompt")
        assert response == "42"

    def test_stream_yields_strings(self):
        from financial_agent import FinancialAgentService
        mock_agent = MagicMock()
        c1, c2 = MagicMock(), MagicMock()
        c1.content = "Hello "
        c2.content = "World"
        mock_agent.run.return_value = [c1, c2]

        svc = FinancialAgentService(agent=mock_agent)
        chunks = list(svc.stream("test"))
        assert chunks == ["Hello ", "World"]

    def test_stream_error_yields_sentinel(self):
        from financial_agent import FinancialAgentService
        mock_agent = MagicMock()
        mock_agent.run.side_effect = RuntimeError("boom")

        svc = FinancialAgentService(agent=mock_agent)
        chunks = list(svc.stream("fail"))
        assert len(chunks) == 1
        assert chunks[0].startswith("[ERROR]")

    def test_stream_skips_empty_chunks(self):
        from financial_agent import FinancialAgentService
        mock_agent = MagicMock()
        c1 = MagicMock()
        c1.content = ""
        c2 = MagicMock()
        c2.content = "data"
        mock_agent.run.return_value = [c1, c2]

        svc = FinancialAgentService(agent=mock_agent)
        chunks = list(svc.stream("test"))
        assert chunks == ["data"]

    # ── Query execution & retry behaviour ─────────────────────────

    def test_service_default_max_retries(self):
        from financial_agent import FinancialAgentService
        from backend.app.agent.financial_agent import _DEFAULT_MAX_RETRIES
        mock_agent = MagicMock()
        svc = FinancialAgentService(agent=mock_agent)
        assert svc._max_retries == _DEFAULT_MAX_RETRIES

    def test_run_calls_agent_with_stream_false(self):
        from financial_agent import FinancialAgentService
        mock_agent = MagicMock()
        result = MagicMock(); result.content = "response"
        mock_agent.run.return_value = result
        svc = FinancialAgentService(agent=mock_agent)
        svc.run("hello")
        mock_agent.run.assert_called_once_with("hello", stream=False)

    def test_run_succeeds_first_attempt_count(self):
        """Successful first attempt should NOT trigger additional retries."""
        from financial_agent import FinancialAgentService
        mock_agent = MagicMock()
        result = MagicMock(); result.content = "ok"
        mock_agent.run.return_value = result
        svc = FinancialAgentService(agent=mock_agent, max_retries=3)
        svc.run("prompt")
        assert mock_agent.run.call_count == 1

    def test_run_multiple_sequential_calls(self):
        """Service should handle successive calls on the same instance."""
        from financial_agent import FinancialAgentService
        mock_agent = MagicMock()
        r1, r2 = MagicMock(), MagicMock()
        r1.content = "first"
        r2.content = "second"
        mock_agent.run.side_effect = [r1, r2]
        svc = FinancialAgentService(agent=mock_agent)
        assert svc.run("one") == "first"
        assert svc.run("two") == "second"

    def test_stream_calls_agent_with_stream_true(self):
        from financial_agent import FinancialAgentService
        mock_agent = MagicMock()
        mock_agent.run.return_value = []
        svc = FinancialAgentService(agent=mock_agent)
        list(svc.stream("hello"))
        mock_agent.run.assert_called_once_with("hello", stream=True)

    def test_stream_handles_plain_string_chunks(self):
        """Chunks that are plain strings (not objects with .content) should work."""
        from financial_agent import FinancialAgentService
        mock_agent = MagicMock()
        mock_agent.run.return_value = ["chunk1", "chunk2"]
        svc = FinancialAgentService(agent=mock_agent)
        chunks = list(svc.stream("test"))
        assert chunks == ["chunk1", "chunk2"]

    def test_stream_does_not_retry(self):
        """Streaming makes exactly one call to the agent, even on failure."""
        from financial_agent import FinancialAgentService
        mock_agent = MagicMock()
        mock_agent.run.side_effect = RuntimeError("stream fail")
        svc = FinancialAgentService(agent=mock_agent, max_retries=5)
        list(svc.stream("prompt"))
        assert mock_agent.run.call_count == 1

    def test_run_retries_on_connection_error(self):
        from financial_agent import FinancialAgentService
        mock_agent = MagicMock()
        good = MagicMock(); good.content = "ok"
        mock_agent.run.side_effect = [ConnectionError("conn"), good]
        svc = FinancialAgentService(agent=mock_agent, max_retries=3)
        assert svc.run("test") == "ok"

    def test_run_retries_on_timeout_error(self):
        from financial_agent import FinancialAgentService
        mock_agent = MagicMock()
        good = MagicMock(); good.content = "recovered"
        mock_agent.run.side_effect = [TimeoutError("timeout"), good]
        svc = FinancialAgentService(agent=mock_agent, max_retries=3)
        assert svc.run("test") == "recovered"

    def test_run_exhausted_chains_original_cause(self):
        """RuntimeError from exhausted retries should chain the original exception."""
        from financial_agent import FinancialAgentService
        mock_agent = MagicMock()
        original = ValueError("original cause")
        mock_agent.run.side_effect = original
        svc = FinancialAgentService(agent=mock_agent, max_retries=1)
        with pytest.raises(RuntimeError) as exc_info:
            svc.run("fail")
        assert exc_info.value.__cause__ is original

    def test_run_max_retries_one_makes_single_attempt(self):
        from financial_agent import FinancialAgentService
        mock_agent = MagicMock()
        mock_agent.run.side_effect = RuntimeError("fail")
        svc = FinancialAgentService(agent=mock_agent, max_retries=1)
        with pytest.raises(RuntimeError):
            svc.run("prompt")
        assert mock_agent.run.call_count == 1

    def test_stream_mid_stream_error(self):
        """If error occurs mid-stream, sentinel appears after partial chunks."""
        from financial_agent import FinancialAgentService
        mock_agent = MagicMock()

        def failing_iter(*args, **kwargs):
            c = MagicMock(); c.content = "partial"
            yield c
            raise RuntimeError("mid-stream failure")

        mock_agent.run.return_value = failing_iter()
        svc = FinancialAgentService(agent=mock_agent)
        chunks = list(svc.stream("test"))
        assert chunks[0] == "partial"
        assert chunks[-1].startswith("[ERROR]")

    def test_run_logs_success_on_first_attempt(self, caplog):
        from financial_agent import FinancialAgentService
        mock_agent = MagicMock()
        result = MagicMock(); result.content = "done"
        mock_agent.run.return_value = result
        svc = FinancialAgentService(agent=mock_agent)
        with caplog.at_level(logging.INFO, logger="financial_agent"):
            svc.run("test")
        assert any("succeeded on attempt 1" in m for m in caplog.messages)

    def test_run_logs_warning_on_retry(self, caplog):
        from financial_agent import FinancialAgentService
        mock_agent = MagicMock()
        good = MagicMock(); good.content = "ok"
        mock_agent.run.side_effect = [RuntimeError("fail"), good]
        svc = FinancialAgentService(agent=mock_agent, max_retries=3)
        with caplog.at_level(logging.WARNING, logger="financial_agent"):
            svc.run("test")
        assert any("attempt 1/3 failed" in m for m in caplog.messages)

    def test_stream_logs_exception_on_error(self, caplog):
        from financial_agent import FinancialAgentService
        mock_agent = MagicMock()
        mock_agent.run.side_effect = RuntimeError("boom")
        svc = FinancialAgentService(agent=mock_agent)
        with caplog.at_level(logging.ERROR, logger="financial_agent"):
            list(svc.stream("test"))
        assert any("Stream error" in m for m in caplog.messages)


# ── _build_tools() registration ──────────────────────────────────────

class TestBuildTools:
    """Tests for ``_build_tools()`` toolkit construction."""

    def test_returns_list(self):
        from financial_agent import _build_tools
        result = _build_tools()
        assert isinstance(result, list)

    def test_all_five_toolkits_loaded(self):
        from financial_agent import _build_tools
        tools = _build_tools()
        assert len(tools) == 5

    def test_toolkit_types(self):
        from financial_agent import _build_tools
        tools = _build_tools()
        type_names = {type(t).__name__ for t in tools}
        expected = {"DuckDuckGoTools", "YFinanceTools", "ChartTools", "ExcelTools", "RAGTools"}
        assert type_names == expected

    def test_each_element_is_toolkit_subclass(self):
        from financial_agent import _build_tools
        for tool in _build_tools():
            assert isinstance(tool, Toolkit)

    def test_custom_factories_parameter(self):
        """build_tools() accepts an explicit factories list."""
        from backend.app.agent.tool_registry import build_tools
        mock_factory = lambda: MagicMock(spec=Toolkit)
        tools = build_tools(factories=[("Test", mock_factory)])
        assert len(tools) == 1

    def test_returns_fresh_list_each_call(self):
        """Successive calls should return independent lists (no shared state)."""
        from financial_agent import _build_tools
        list1 = _build_tools()
        list2 = _build_tools()
        assert list1 is not list2

    def test_empty_factories_returns_empty(self):
        from backend.app.agent.tool_registry import build_tools
        tools = build_tools(factories=[])
        assert tools == []


# ── TOOL_FACTORIES registry structure ────────────────────────────────

class TestToolFactories:
    """Validate the TOOL_FACTORIES registry structure and contents."""

    def test_factories_is_list(self):
        from backend.app.agent.tool_registry import TOOL_FACTORIES
        assert isinstance(TOOL_FACTORIES, list)

    def test_expected_factory_count(self):
        from backend.app.agent.tool_registry import TOOL_FACTORIES
        assert len(TOOL_FACTORIES) == 5

    def test_each_entry_is_name_callable_pair(self):
        from backend.app.agent.tool_registry import TOOL_FACTORIES
        for entry in TOOL_FACTORIES:
            assert isinstance(entry, tuple) and len(entry) == 2
            name, factory = entry
            assert isinstance(name, str)
            assert callable(factory)

    def test_factory_names_are_unique(self):
        from backend.app.agent.tool_registry import TOOL_FACTORIES
        names = [name for name, _ in TOOL_FACTORIES]
        assert len(names) == len(set(names))

    def test_factory_names_match_expected(self):
        from backend.app.agent.tool_registry import TOOL_FACTORIES
        names = {name for name, _ in TOOL_FACTORIES}
        assert names == {"DuckDuckGo", "YFinance", "Charts", "Excel", "RAG"}

    def test_each_factory_produces_toolkit(self):
        """Every registered factory should produce a valid Toolkit instance."""
        from backend.app.agent.tool_registry import TOOL_FACTORIES
        for name, factory in TOOL_FACTORIES:
            tool = factory()
            assert isinstance(tool, Toolkit), f"{name} factory did not produce a Toolkit"


# ── Tool failure handling (graceful degradation) ─────────────────────

class TestToolFailureHandling:
    """Tests for _build_tools() when individual toolkits fail to load."""

    def test_skips_single_failing_toolkit(self):
        """If one toolkit factory raises, the others should still load."""
        from financial_agent import _build_tools
        with patch("backend.app.agent.tool_registry.ChartTools", side_effect=ImportError("no matplotlib")):
            tools = _build_tools()
            type_names = {type(t).__name__ for t in tools}
            assert "ChartTools" not in type_names
            assert len(tools) == 4

    def test_skips_multiple_failing_toolkits(self):
        from financial_agent import _build_tools
        with patch("backend.app.agent.tool_registry.ChartTools", side_effect=ImportError), \
             patch("backend.app.agent.tool_registry.ExcelTools", side_effect=OSError):
            tools = _build_tools()
            assert len(tools) == 3
            type_names = {type(t).__name__ for t in tools}
            assert "ChartTools" not in type_names
            assert "ExcelTools" not in type_names

    def test_returns_empty_when_all_fail(self):
        from financial_agent import _build_tools
        with patch("backend.app.agent.tool_registry.DuckDuckGoTools", side_effect=Exception), \
             patch("backend.app.agent.tool_registry.YFinanceTools", side_effect=Exception), \
             patch("backend.app.agent.tool_registry.ChartTools", side_effect=Exception), \
             patch("backend.app.agent.tool_registry.ExcelTools", side_effect=Exception), \
             patch("backend.app.agent.tool_registry.RAGTools", side_effect=Exception):
            tools = _build_tools()
            assert tools == []

    def test_logs_warning_on_failure(self, caplog):
        from financial_agent import _build_tools
        with caplog.at_level(logging.WARNING, logger="financial_agent"), \
             patch("backend.app.agent.tool_registry.RAGTools", side_effect=RuntimeError("boom")):
            _build_tools()
        assert any("Failed to load toolkit 'RAG'" in msg for msg in caplog.messages)

    def test_logs_debug_on_success(self, caplog):
        from financial_agent import _build_tools
        with caplog.at_level(logging.DEBUG, logger="financial_agent"):
            _build_tools()
        loaded = [m for m in caplog.messages if m.startswith("Loaded toolkit")]
        assert len(loaded) == 5

    def test_remaining_tools_still_functional_after_failure(self):
        """Ensure the loaded tools are real, working instances (not broken proxies)."""
        from financial_agent import _build_tools
        with patch("backend.app.agent.tool_registry.ChartTools", side_effect=ImportError):
            tools = _build_tools()
            for t in tools:
                # Every Toolkit subclass exposes a name attribute
                assert hasattr(t, "name")

    def test_order_preserved_after_middle_failure(self):
        """If the 3rd toolkit fails, tools 1-2 and 4-5 keep their original order."""
        from backend.app.agent.tool_registry import build_tools
        with patch("backend.app.agent.tool_registry.ChartTools", side_effect=ImportError):
            tools = build_tools()
        names = [type(t).__name__ for t in tools]
        assert names == ["DuckDuckGoTools", "YFinanceTools", "ExcelTools", "RAGTools"]


# ── Response packaging (AGENT_INSTRUCTIONS content) ──────────────────

class TestResponsePackaging:
    """Verify AGENT_INSTRUCTIONS instructs the agent to format output properly."""

    def test_instructions_is_list_of_strings(self):
        from financial_agent import AGENT_INSTRUCTIONS
        assert isinstance(AGENT_INSTRUCTIONS, list)
        assert all(isinstance(s, str) for s in AGENT_INSTRUCTIONS)

    def test_instructions_mention_tool_categories(self):
        from financial_agent import AGENT_INSTRUCTIONS
        joined = "\n".join(AGENT_INSTRUCTIONS)
        for keyword in ["Stock data", "Search", "Charts", "Excel", "Knowledge retrieval"]:
            assert keyword in joined, f"Missing tool category: {keyword}"

    def test_instructions_contain_chart_directive(self):
        from financial_agent import AGENT_INSTRUCTIONS
        joined = "\n".join(AGENT_INSTRUCTIONS)
        assert "create_stock_chart" in joined
        assert "Do NOT use markdown image syntax" in joined

    def test_instructions_contain_excel_directive(self):
        from financial_agent import AGENT_INSTRUCTIONS
        joined = "\n".join(AGENT_INSTRUCTIONS)
        assert "create_stock_excel_report" in joined

    def test_instructions_contain_disclaimer(self):
        from financial_agent import AGENT_INSTRUCTIONS
        joined = "\n".join(AGENT_INSTRUCTIONS)
        assert "informational purposes only" in joined

    def test_instructions_never_say_cannot(self):
        """Agent must be instructed to never claim inability to create charts/sheets."""
        from financial_agent import AGENT_INSTRUCTIONS
        joined = "\n".join(AGENT_INSTRUCTIONS)
        assert "NEVER say you cannot create" in joined

    def test_instructions_require_dollar_sign_prices(self):
        from financial_agent import AGENT_INSTRUCTIONS
        joined = "\n".join(AGENT_INSTRUCTIONS)
        assert "$ symbols" in joined

    def test_instructions_require_tables(self):
        from financial_agent import AGENT_INSTRUCTIONS
        joined = "\n".join(AGENT_INSTRUCTIONS)
        assert "tables" in joined.lower()


# ── Prompts data module ──────────────────────────────────────────────

class TestPromptsModule:
    """Tests for the prompts.py pure-data module."""

    def test_tool_catalogue_is_non_empty_string(self):
        from backend.app.agent.prompts import TOOL_CATALOGUE
        assert isinstance(TOOL_CATALOGUE, str)
        assert len(TOOL_CATALOGUE) > 0

    def test_behavioural_rules_is_non_empty_string(self):
        from backend.app.agent.prompts import BEHAVIOURAL_RULES
        assert isinstance(BEHAVIOURAL_RULES, str)
        assert len(BEHAVIOURAL_RULES) > 0

    def test_agent_instructions_contains_tool_catalogue(self):
        from backend.app.agent.prompts import AGENT_INSTRUCTIONS, TOOL_CATALOGUE
        assert TOOL_CATALOGUE in AGENT_INSTRUCTIONS

    def test_agent_instructions_contains_behavioural_rules(self):
        from backend.app.agent.prompts import AGENT_INSTRUCTIONS, BEHAVIOURAL_RULES
        assert BEHAVIOURAL_RULES in AGENT_INSTRUCTIONS

    def test_prompts_module_all_exports(self):
        from backend.app.agent import prompts
        expected = {"TOOL_CATALOGUE", "BEHAVIOURAL_RULES", "AGENT_INSTRUCTIONS"}
        assert expected == set(prompts.__all__)

    def test_tool_catalogue_lists_all_five_categories(self):
        from backend.app.agent.prompts import TOOL_CATALOGUE
        for category in ["Stock data", "Search", "Charts", "Excel", "Knowledge retrieval"]:
            assert category in TOOL_CATALOGUE, f"Missing category: {category}"

    def test_behavioural_rules_mention_chart_and_spreadsheet(self):
        from backend.app.agent.prompts import BEHAVIOURAL_RULES
        assert "chart" in BEHAVIOURAL_RULES.lower()
        assert "spreadsheet" in BEHAVIOURAL_RULES.lower() or "excel" in BEHAVIOURAL_RULES.lower()

    def test_instructions_separator_between_fragments(self):
        """AGENT_INSTRUCTIONS should have a blank separator between fragments."""
        from backend.app.agent.prompts import AGENT_INSTRUCTIONS
        assert "" in AGENT_INSTRUCTIONS


# ── Tool-registry module ─────────────────────────────────────────────

class TestToolRegistryModule:
    """Tests for tool_registry.py module-level exports."""

    def test_module_all_exports(self):
        from backend.app.agent import tool_registry
        assert set(tool_registry.__all__) == {"build_tools", "TOOL_FACTORIES"}

    def test_build_tools_is_callable(self):
        from backend.app.agent.tool_registry import build_tools
        assert callable(build_tools)


# ── Invalid input handling ───────────────────────────────────────────

class TestInvalidInputs:
    """Tests for edge cases and bad inputs to the public API."""

    def test_create_agent_empty_model_id_accepted(self):
        """Agent factory doesn't validate model IDs—that's Groq's job."""
        from financial_agent import create_agent
        a = create_agent(model_id="")
        assert a.model.id == ""

    def test_create_agent_none_tools_builds_defaults(self):
        from financial_agent import create_agent
        a = create_agent(tools=None)
        assert len(a.tools) == 5

    def test_create_agent_none_instructions_uses_defaults(self):
        from financial_agent import create_agent, AGENT_INSTRUCTIONS
        a = create_agent(instructions=None)
        assert a.instructions == AGENT_INSTRUCTIONS


# ── CLI ──────────────────────────────────────────────────────────────

class TestCLI:
    """Tests for the CLI entry point."""

    def test_main_with_query_arg(self):
        from financial_agent import main
        with patch("backend.app.agent.financial_agent.agent") as mock_agent, \
             patch("sys.argv", ["financial_agent.py", "What is AAPL?"]):
            main()
            mock_agent.print_response.assert_called_once_with("What is AAPL?", stream=False)

    def test_main_with_stream_flag(self):
        from financial_agent import main
        with patch("backend.app.agent.financial_agent.agent") as mock_agent, \
             patch("sys.argv", ["financial_agent.py", "--stream", "Hello"]):
            main()
            mock_agent.print_response.assert_called_once_with("Hello", stream=True)

    def test_main_no_query_enters_interactive(self):
        from financial_agent import main
        with patch("backend.app.agent.financial_agent.agent"), \
             patch("backend.app.agent.financial_agent._interactive_loop") as mock_loop, \
             patch("sys.argv", ["financial_agent.py"]):
            main()
            mock_loop.assert_called_once_with(stream=False)

    def test_main_stream_flag_passed_to_interactive(self):
        from financial_agent import main
        with patch("backend.app.agent.financial_agent.agent"), \
             patch("backend.app.agent.financial_agent._interactive_loop") as mock_loop, \
             patch("sys.argv", ["financial_agent.py", "--stream"]):
            main()
            mock_loop.assert_called_once_with(stream=True)


class TestInteractiveLoop:
    """Tests for the ``_interactive_loop`` REPL."""

    def test_exit_command_breaks_loop(self):
        from financial_agent import _interactive_loop
        with patch("builtins.input", return_value="exit"), \
             patch("builtins.print") as mock_print:
            _interactive_loop()
            mock_print.assert_any_call("Goodbye!")

    def test_quit_command_breaks_loop(self):
        from financial_agent import _interactive_loop
        with patch("builtins.input", return_value="quit"), \
             patch("builtins.print"):
            _interactive_loop()

    def test_empty_input_breaks_loop(self):
        from financial_agent import _interactive_loop
        with patch("builtins.input", return_value=""), \
             patch("builtins.print"):
            _interactive_loop()

    def test_keyboard_interrupt_handled(self):
        from financial_agent import _interactive_loop
        with patch("builtins.input", side_effect=KeyboardInterrupt), \
             patch("builtins.print") as mock_print:
            _interactive_loop()
            mock_print.assert_any_call("\nGoodbye!")

    def test_eof_error_handled(self):
        from financial_agent import _interactive_loop
        with patch("builtins.input", side_effect=EOFError), \
             patch("builtins.print") as mock_print:
            _interactive_loop()
            mock_print.assert_any_call("\nGoodbye!")

    def test_valid_query_calls_print_response(self):
        from financial_agent import _interactive_loop
        # First call returns a query, second returns "exit"
        with patch("builtins.input", side_effect=["What is AAPL?", "exit"]), \
             patch("builtins.print"), \
             patch("backend.app.agent.financial_agent.agent") as mock_agent:
            _interactive_loop(stream=False)
            mock_agent.print_response.assert_called_once_with("What is AAPL?", stream=False)

    def test_valid_query_streams_when_flag_set(self):
        from financial_agent import _interactive_loop
        with patch("builtins.input", side_effect=["Show me TSLA", "exit"]), \
             patch("builtins.print"), \
             patch("backend.app.agent.financial_agent.agent") as mock_agent:
            _interactive_loop(stream=True)
            mock_agent.print_response.assert_called_once_with("Show me TSLA", stream=True)

    def test_agent_error_does_not_crash_loop(self):
        """Runtime errors from the agent are caught; the loop continues."""
        from financial_agent import _interactive_loop
        with patch("builtins.input", side_effect=["bad query", "exit"]), \
             patch("builtins.print") as mock_print, \
             patch("backend.app.agent.financial_agent.agent") as mock_agent:
            mock_agent.print_response.side_effect = RuntimeError("LLM timeout")
            _interactive_loop()
            # Loop should still exit normally after the error
            mock_print.assert_any_call("Goodbye!")
