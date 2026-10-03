# Agent Instructions for this Repository

Welcome, Agent. You are operating within the **Modular 5-Axis CAM Framework**.

## Strict Rules
1. **Always Verify Architecture:** Read `ARCHITECTURE.md` before attempting to modify contracts or core components.
2. **Living Documentation:** If you add a new Stage 1 Slicer Plugin, Mathematical Strategy, Kinematic Solver, or Data Schema property, you **MUST** update `ARCHITECTURE.md` to reflect the new state of the system in the same Pull Request/Commit.
3. **No C-Extensions:** Do not introduce dependencies that require compiled C-extensions (e.g., `scipy`, `cython`) unless strictly necessary, as they break deployment on certain constrained Windows targets via DLL restrictions. Use pure `numpy` for mathematical operations.
4. **Intermediate JSON Artifacts:** Ensure any testing outputs generated in `tests/output/` or `*.html` visualization files are strictly added to `.gitignore` and never committed.
5. **No Blind Eval:** Mathematical expressions passed through configurations must be sanitized. Do not expose `__builtins__` in any `eval()` environment to prevent RCE.

By adhering to these principles, we maintain a strictly decoupled, verifiable, and secure pipeline.