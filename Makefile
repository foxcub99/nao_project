# Isaac Lab Command Shortcuts

ISAAC_LAB_PATH = ..\..\IsaacLab
ISAAC_LAB_CMD = $(ISAAC_LAB_PATH)\isaaclab.bat -p

DEFAULT_TASK = Nao-Direct

# Default target to show usage
help:
	@echo Available targets:
	@echo   isaac ARGS="command"  - Run command with IsaacLab prefix
	@echo   Example: make isaac ARGS="scripts/skrl/train.py"
	@echo   debug                 - Run debug training script
	@echo   train                 - Run training script
	@echo   play                  - Run play script

# Run command with IsaacLab prefix
isaac:
	$(ISAAC_LAB_CMD) $(ARGS)

.PHONY: help isaac debug train play view-logs view-recent-log
	
debug:
	$(ISAAC_LAB_CMD) scripts/skrl/train.py --task $(if $(word 2,$(MAKECMDGOALS)),$(word 2,$(MAKECMDGOALS)),$(DEFAULT_TASK)) --num_envs 1

train:
	$(ISAAC_LAB_CMD) scripts/skrl/train.py --task $(if $(word 2,$(MAKECMDGOALS)),$(word 2,$(MAKECMDGOALS)),$(DEFAULT_TASK)) --headless

play:
	$(ISAAC_LAB_CMD) scripts/skrl/play.py --task $(if $(word 2,$(MAKECMDGOALS)),$(word 2,$(MAKECMDGOALS)),$(DEFAULT_TASK)) --num_envs 6

# Dummy target to prevent make from trying to build files named after tasks
%:
	@:

view-logs:
	@echo [INFO]if you run into error make sure to 'conda deactivate' so that it uses isaacs python
	$(ISAAC_LAB_CMD) -m tensorboard.main --logdir logs/skrl/nao_direct

view-recent-log:
	@echo [INFO] If you run into errors, make sure to 'conda deactivate' so that it uses Isaac\'s Python
	@for /f %%i in ('dir /b /ad /o-d logs\skrl\nao_project') do ( \
		echo [INFO] Launching TensorBoard with logdir=logs\skrl\nao_project\%%i && \
		$(ISAAC_LAB_CMD) -m tensorboard.main --logdir logs/skrl/nao_project/%%i \
		& goto end \
	)
	@:end