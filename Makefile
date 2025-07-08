# Isaac Lab Command Shortcuts
# Usage: make p ARGS="your_command_here"
# Example: make p ARGS="python scripts/skrl/train.py"

ISAAC_LAB_PATH = ..\IsaacLab
ISAAC_LAB_CMD = $(ISAAC_LAB_PATH)\isaaclab.bat -p

# Default target to show usage
help:
	@echo Available targets:
	@echo   p ARGS="command"  - Run command with IsaacLab prefix
	@echo   Example: make p ARGS="python scripts/skrl/train.py"

# Run command with IsaacLab prefix
p:
	$(ISAAC_LAB_CMD) $(ARGS)

.PHONY: help p