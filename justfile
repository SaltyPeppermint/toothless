alias r := run

run:
    uv run toothless/main.py

alias i := reinstall

reinstall: remove-eggshell install-eggshell

reinstall-macos: remove-eggshell install-eggshell-macos

remove-eggshell:
    uv remove eggshell

install-eggshell:
    cp ../eggshell/target/wheels/eggshell-0.0.1-cp313-cp313-manylinux_2_34_x86_64.whl cache/
    uv add cache/eggshell-0.0.1-cp313-cp313-manylinux_2_34_x86_64.whl

install-eggshell-macos:
    cp ../eggshell/target/wheels/eggshell-0.0.1-cp313-cp313-macosx_11_0_arm64.whl cache/
    uv add cache/eggshell-0.0.1-cp313-cp313-macosx_11_0_arm64.whl

alias ri := reinstall-run

reinstall-run: reinstall run
