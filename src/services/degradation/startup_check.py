#!/usr/bin/env python3
"""
Startup Self-Check Script
=========================

Validates all critical endpoints and services on startup.
Prints actionable fixes if something is broken.

Usage:
    python3 -m src.services.degradation.startup_check

Or run automatically at startup via:
    python3 -c "from src.services.degradation import run_startup_check; run_startup_check()"
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.logging import get_logger

logger = get_logger("StartupCheck")


class StartupCheck:
    """Runs all startup checks and reports status."""

    def __init__(self):
        self.checks: list[dict] = []
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def add_check(self, name: str, category: str, check_func, fix_message: str) -> None:
        """Add a check to run."""
        self.checks.append(
            {
                "name": name,
                "category": category,
                "check": check_func,
                "fix": fix_message,
            }
        )

    async def run_all(self) -> dict[str, Any]:
        """Run all checks and return results."""
        print("\n" + "=" * 60)
        print("🔍 DeepTutor Startup Self-Check")
        print("=" * 60 + "\n")

        # Environment checks
        await self._check_env()

        # Directory checks
        await self._check_directories()

        # Config checks
        await self._check_config()

        # Service checks
        await self._check_services()

        # Print summary
        self._print_summary()

        return {
            "passed": self.passed,
            "failed": self.failed,
            "warnings": self.warnings,
            "timestamp": datetime.now().isoformat(),
        }

    async def _check_env(self) -> None:
        """Check required environment variables."""
        print("📋 Checking Environment...")

        checks = [
            ("DEEPTUTOR_DATA", "Data directory path"),
            ("VOICE_ASR_URL", "ASR service URL (default: http://localhost:7001)"),
            ("OPENROUTER_API_KEY", "OpenRouter API key for cloud LLM"),
            ("RAG_PROVIDER", "RAG provider (raganything, llamaindex, etc.)"),
        ]

        for var, desc in checks:
            value = os.getenv(var)
            if value:
                print(f"  ✅ {var}: Set")
            else:
                print(f"  ⚠️  {var}: Not set ({desc})")
                self.warnings += 1

    async def _check_directories(self) -> None:
        """Check required directories exist."""
        print("\n📁 Checking Directories...")

        base = _project_root

        dirs = [
            (base / "data" / "user", "User data"),
            (base / "data" / "knowledge_bases", "Knowledge bases"),
            (base / "data" / "logs", "Application logs"),
            (base / "src", "Source code"),
        ]

        for path, desc in dirs:
            if path.exists():
                print(f"  ✅ {path.name}: exists")
                self.passed += 1
            else:
                print(f"  ❌ {path.name}: missing ({desc})")
                print(f"     Fix: mkdir -p {path}")
                self.failed += 1

    async def _check_config(self) -> None:
        """Check configuration files."""
        print("\n⚙️  Checking Configuration...")

        config_file = _project_root / "solve_config.yaml"
        if config_file.exists():
            print(f"  ✅ solve_config.yaml: exists")
            self.passed += 1

            # Check key configs
            try:
                from src.services.config import load_config_with_main

                config = load_config_with_main("solve_config.yaml", _project_root)

                llm_config = config.get("llm", {})
                if llm_config.get("api_key") or os.getenv("OPENROUTER_API_KEY"):
                    print(f"  ✅ LLM API key: configured")
                else:
                    print(f"  ❌ LLM API key: missing")
                    print(f"     Fix: Set OPENROUTER_API_KEY env var or add to config")
                    self.failed += 1

                rag_config = config.get("rag", {})
                if rag_config.get("provider"):
                    print(f"  ✅ RAG provider: {rag_config['provider']}")
                else:
                    print(f"  ⚠️  RAG provider: default (raganything)")
                    self.warnings += 1

            except Exception as e:
                print(f"  ⚠️  Config parse warning: {e}")
                self.warnings += 1
        else:
            print(f"  ❌ solve_config.yaml: missing")
            print(f"     Fix: Copy solve_config.example.yaml to solve_config.yaml")
            self.failed += 1

    async def _check_services(self) -> None:
        """Check external services."""
        print("\n🌐 Checking External Services...")

        await self._check_voice_gateway()
        await self._check_llm()
        await self._check_rag()

    async def _check_voice_gateway(self) -> None:
        """Check if voice gateway is available."""
        import aiohttp

        url = os.getenv("VOICE_GATEWAY_URL", "http://localhost:7001")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{url}/health", timeout=aiohttp.ClientTimeout(total=3)
                ) as resp:
                    if resp.status == 200:
                        print(f"  ✅ Voice Gateway: healthy ({url})")
                        self.passed += 1
                    else:
                        print(f"  ⚠️  Voice Gateway: returned {resp.status}")
                        self.warnings += 1
        except Exception as e:
            print(f"  ⚠️  Voice Gateway: unavailable ({url})")
            print(f"     Impact: Voice chat and TTS will use text-only mode")
            print(f"     Fix: Start voice gateway: python3 -m extensions.voice.server")
            self.warnings += 1

    async def _check_llm(self) -> None:
        """Check if LLM provider is available."""
        try:
            from src.services.llm.config import get_llm_config

            config = get_llm_config()

            if not config.api_key and not config.base_url:
                print(f"  ❌ LLM: No provider configured")
                print(f"     Fix: Set OPENROUTER_API_KEY or configure local provider")
                self.failed += 1
                return

            async with asyncio.timeout(10):
                import aiohttp

                headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
                base_url = config.base_url or "https://api.openai.com/v1"

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{base_url}/chat/completions",
                        json={
                            "model": config.model,
                            "messages": [{"role": "user", "content": "test"}],
                        },
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        if resp.status in (200, 400):
                            print(f"  ✅ LLM: {config.model} accessible")
                            self.passed += 1
                        else:
                            print(f"  ❌ LLM: returned {resp.status}")
                            self.failed += 1
        except asyncio.TimeoutError:
            print(f"  ⚠️  LLM: timeout - service may be slow")
            self.warnings += 1
        except Exception as e:
            print(f"  ❌ LLM: {str(e)[:50]}")
            print(f"     Fix: Check API key and provider URL")
            self.failed += 1

    async def _check_rag(self) -> None:
        """Check RAG service and knowledge bases."""
        kb_dir = _project_root / "data" / "knowledge_bases"

        if not kb_dir.exists():
            print(f"  ⚠️  RAG: knowledge_bases directory missing")
            print(f"     Fix: mkdir -p {kb_dir}")
            self.warnings += 1
            return

        kbs = [d for d in kb_dir.iterdir() if d.is_dir() and (d / "_indexed").exists()]

        if kbs:
            print(f"  ✅ RAG: {len(kbs)} knowledge base(s) indexed")
            for kb in kbs[:3]:
                print(f"     - {kb.name}")
            if len(kbs) > 3:
                print(f"     ... and {len(kbs) - 3} more")
            self.passed += 1
        else:
            print(f"  ⚠️  RAG: no indexed knowledge bases")
            print(f"     Impact: Responses will use general knowledge only")
            print(f"     Fix: Add documents via /api/v1/knowledge/add")
            self.warnings += 1

    def _print_summary(self) -> None:
        """Print check summary."""
        print("\n" + "=" * 60)
        print("📊 Summary")
        print("=" * 60)

        total = self.passed + self.failed + self.warnings
        status = (
            "✅ HEALTHY"
            if self.failed == 0
            else ("⚠️  DEGRADED" if self.warnings > 0 else "❌ UNHEALTHY")
        )

        print(f"Status: {status}")
        print(f"Passed: {self.passed}/{total}")
        print(f"Failed: {self.failed}")
        print(f"Warnings: {self.warnings}")
        print()

        if self.failed > 0:
            print("❌ Critical Issues:")
            print("   Fix the issues above before proceeding.")
            print()
        if self.warnings > 0:
            print("⚠️  Warnings:")
            print("   The system will run in degraded mode.")
            print("   Some features may be unavailable.")
            print()

        print("🚀 To start the server:")
        print("   python3 -m src.api.main")
        print()


async def run_startup_check() -> dict[str, Any]:
    """Run the startup check and return results."""
    checker = StartupCheck()
    return await checker.run_all()


def main():
    """Main entry point."""
    try:
        result = asyncio.run(run_startup_check())
        sys.exit(0 if result["failed"] == 0 else 1)
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Startup check failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
