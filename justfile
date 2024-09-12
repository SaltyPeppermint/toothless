alias r := run

run:
    uv run toothless/util.py

alias i := reinstall

reinstall:
    uv remove eggshell && uv add ../eggshell/target/wheels/eggshell-0.0.1-cp312-cp312-macosx_11_0_arm64.whl

alias ri := reinstall_run

reinstall_run: reinstall run