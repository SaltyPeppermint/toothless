alias r := run

run:
    uv run toothless/main.py

alias i := reinstall

reinstall: remove_eggshell install_eggshell

remove_eggshell:
    uv remove eggshell

install_eggshell:
    uv add ../eggshell/target/wheels/eggshell-0.0.1-cp312-cp312-manylinux_2_34_x86_64.whl

alias ri := reinstall_run

reinstall_run: reinstall run