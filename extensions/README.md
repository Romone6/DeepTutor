# Extensions Layer

This directory provides a clean separation layer for adding new functionality to DeepTutor without modifying core modules in `src/`.

## Purpose

The `extensions/` layer allows contributors to:
- Add new features (voice, knowledge, policies, agents, routers)
- Experiment with alternative implementations
- Maintain backwards compatibility

All while keeping `src/` modules untouched and stable.

## Boundaries

1. **DO NOT rewrite `src/` modules** - Extensions should only add, never modify existing core functionality
2. **DO import from `src/`** - Extensions can depend on core modules (`from src.core.config import ...`)
3. **DO NOT make `src/` depend on `extensions/`** - This keeps core independent of extensions

## Directory Structure

```
extensions/
├── README.md              # This file
├── utils/                 # Shared utilities (schemas, config)
├── router/                # Router decision handlers
├── voice/                 # Voice/TTS extensions
├── knowledge/             # Knowledge base extensions
├── policies/              # Policy/routing policies
├── agents/                # Agent extensions
└── utils/                 # (shared utilities)
```

## Usage

### Importing shared schemas

```python
from extensions.utils import RouteDecision

decision = RouteDecision(
    target="voice",
    handler="elevenlabs_tts",
    confidence=0.95
)
```

### Importing shared config

```python
from extensions.utils import get_extension_config

config = get_extension_config()
if config.enabled:
    # Extension logic
```

## Adding a New Extension

1. Create a new subfolder under `extensions/` (e.g., `extensions/voice/`)
2. Add an `__init__.py` with module docstring
3. Import shared schemas/config from `extensions.utils`
4. Add your extension code
5. Document usage in this README

## Guidelines

- Follow existing code style (see `pyproject.toml`)
- Use Pydantic for schemas (match patterns in `src/`)
- Use `pydantic-settings` for configuration
- Prefix environment variables with `EXTENSION_`
- Write tests in `tests/` if adding test coverage
