

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import yaml


import pathlib
# Custom loader for !!python/tuple
def construct_python_tuple(loader, node):
    return tuple(loader.construct_sequence(node))
yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple', construct_python_tuple)

# Custom loader for !!python/object/apply:builtins.slice
def construct_python_slice(loader, node):
    args = loader.construct_sequence(node)
    while len(args) < 3:
        args.append(None)
    return slice(*args)
yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/object/apply:builtins.slice', construct_python_slice)

# Custom loader for !!python/object/apply:pathlib.PosixPath
def construct_posix_path(loader, node):
    args = loader.construct_sequence(node)
    # Just join args as a string path, do not instantiate PosixPath
    return os.path.join(*[str(a) for a in args])
yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/object/apply:pathlib.PosixPath', construct_posix_path)

# Custom Unpickler to ignore missing modules/classes
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Only allow safe built-in types
        if module == "builtins":
            return super().find_class(module, name)
        # For anything else, return dict (or None)
        return dict

def extract_tables(env_data):
    # Table 1: Reward weights and command weights
    rewards = env_data.get('rewards', {}) or {}
    reward_df = pd.DataFrame({
        'Reward Title': list(rewards.keys()),
        'Reward Weight': [v.get('weight', None) if isinstance(v, dict) else None for v in rewards.values()]
    })
    commands = env_data.get('commands', {}) or {}
    command_titles = []
    command_weights = []
    x_weights = []
    y_weights = []
    z_weights = []
    for k, v in commands.items():
        command_titles.append(k)
        if isinstance(v, dict):
            command_weights.append(v.get('weight', None))
            ranges = v.get('ranges', {})
            x_weights.append(ranges.get('lin_vel_x', None))
            y_weights.append(ranges.get('lin_vel_y', None))
            z_weights.append(ranges.get('ang_vel_z', None))
        else:
            command_weights.append(None)
            x_weights.append(None)
            y_weights.append(None)
            z_weights.append(None)
    command_df = pd.DataFrame({
        'Command Title': command_titles,
        'Command Weight': command_weights,
        'X Weight': x_weights,
        'Y Weight': y_weights,
        'Z Weight': z_weights
    })

    # Curriculum as text
    curriculum = env_data.get('curriculum', {})
    curriculum_text = ''
    if isinstance(curriculum, dict) and curriculum:
        for k, v in curriculum.items():
            if v is not None:
                curriculum_str = f"{k}: {v}\n"
                curriculum_text += curriculum_str

    # Event differences as text
    events = env_data.get('events', {}) or {}
    default_events = {
        'physics_material': {
            'func': 'isaaclab.envs.mdp.events:randomize_rigid_body_material',
            'params': {
                'asset_cfg': {
                    'name': 'robot',
                    'joint_names': None,
                    'joint_ids': (None, None, None),
                    'fixed_tendon_names': None,
                    'fixed_tendon_ids': (None, None, None),
                    'body_names': '.*',
                    'body_ids': (None, None, None),
                    'object_collection_names': None,
                    'object_collection_ids': (None, None, None),
                    'preserve_order': False
                },
                'static_friction_range': (0.8, 0.8),
                'dynamic_friction_range': (0.6, 0.6),
                'restitution_range': (0.0, 0.0),
                'num_buckets': 64
            },
            'mode': 'startup',
            'interval_range_s': None,
            'is_global_time': False,
            'min_step_count_between_reset': 0
        },
        'add_base_mass': None,
        'base_external_force_torque': {
            'func': 'isaaclab.envs.mdp.events:apply_external_force_torque',
            'params': {
                'asset_cfg': {
                    'name': 'robot',
                    'joint_names': None,
                    'joint_ids': (None, None, None),
                    'fixed_tendon_names': None,
                    'fixed_tendon_ids': (None, None, None),
                    'body_names': ['base_link'],
                    'body_ids': (None, None, None),
                    'object_collection_names': None,
                    'object_collection_ids': (None, None, None),
                    'preserve_order': False
                },
                'force_range': (0.0, 0.0),
                'torque_range': (-0.0, 0.0)
            },
            'mode': 'reset',
            'interval_range_s': None,
            'is_global_time': False,
            'min_step_count_between_reset': 0
        },
        'reset_base': {
            'func': 'isaaclab.envs.mdp.events:reset_root_state_uniform',
            'params': {
                'pose_range': {
                    'x': (-0.5, 0.5),
                    'y': (-0.5, 0.5),
                    'yaw': (-3.14, 3.14)
                },
                'velocity_range': {
                    'x': (0.0, 0.0),
                    'y': (0.0, 0.0),
                    'z': (0.0, 0.0),
                    'roll': (0.0, 0.0),
                    'pitch': (0.0, 0.0),
                    'yaw': (0.0, 0.0)
                }
            },
            'mode': 'reset',
            'interval_range_s': None,
            'is_global_time': False,
            'min_step_count_between_reset': 0
        },
        'reset_robot_joints': {
            'func': 'isaaclab.envs.mdp.events:reset_joints_by_scale',
            'params': {
                'position_range': (1.0, 1.0),
                'velocity_range': (0.0, 0.0)
            },
            'mode': 'reset',
            'interval_range_s': None,
            'is_global_time': False,
            'min_step_count_between_reset': 0
        },
        'push_robot': None
    }
    def normalize(obj):
        if isinstance(obj, dict):
            return {k: normalize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return tuple(normalize(v) for v in obj)
        return obj
    events_norm = normalize(events)
    default_events_norm = normalize(default_events)
    show_event_text = events_norm != default_events_norm
    event_text = ''
    if show_event_text:
        event_text = 'Differences from default event settings:\n\n'
        for k, v in events.items():
            default_cfg = default_events.get(k, None)
            if v != default_cfg:
                event_text += f"Event: {k}\n{v}\n\n"
    return reward_df, command_df, curriculum_text, event_text

def save_reward_command_tables_as_png(reward_df, command_df, save_path):
    if reward_df.empty and command_df.empty:
        print(f"Warning: Both reward and command tables are empty. Skipping save for {save_path}.")
        return
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    for ax, df, title in zip(axs, [reward_df, command_df], ["Reward Weights", "Command Weights"]):
        ax.axis('off')
        if not df.empty:
            tbl = ax.table(cellText=df.values,
                           colLabels=df.columns,
                           loc='center',
                           cellLoc='center')
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def save_table_as_png(df, save_path, title):
    # For curriculum and events, render as formatted text for readability
    if isinstance(df, str):
        text = df if df else f"No {title.lower()} found."
        # Estimate figure size based on text length
        num_lines = len(text.split('\n'))
        num_chars = len(text)
        # Heuristic: 80 chars per line, 0.4 height per line, 16 width for long lines
        fig_width = max(12, min(24, num_chars // 80 + 12))
        fig_height = max(2, min(40, num_lines * 0.4 + num_chars // 400))
        # Reduce font size for very long text
        font_size = 16 if num_chars < 2000 else 12
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis('off')
        ax.text(0, 1, text, fontsize=font_size, va='top', ha='left', wrap=True, family='monospace')
        plt.title(title, fontsize=18)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
    else:
        if df.empty:
            print(f"Warning: Table '{title}' is empty. Skipping save for {save_path}.")
            return
        fig, ax = plt.subplots(figsize=(max(6, df.shape[1]*2), max(2, df.shape[0]*0.5 + df.shape[0])))
        ax.axis('off')
        tbl = ax.table(cellText=df.values,
                       colLabels=df.columns,
                       loc='center',
                       cellLoc='center')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

def main():
    base_dir = os.path.join(os.path.dirname(__file__), 'logs-batches')
    for root, dirs, files in os.walk(base_dir):
        for d in dirs:
            folder_path = os.path.join(root, d)
            env_yaml_path = os.path.join(folder_path, 'params', 'env.yaml')
            if os.path.isfile(env_yaml_path):
                try:
                    with open(env_yaml_path, 'r') as f:
                        env_data = yaml.safe_load(f)
                except Exception as e:
                    print(f"Failed to load {env_yaml_path}: {e}")
                    continue
                reward_df, command_df, curriculum_text, event_text = extract_tables(env_data)
                save_reward_command_tables_as_png(reward_df, command_df, os.path.join(folder_path, 'reward_command_weights.png'))
                if curriculum_text:
                    save_table_as_png(curriculum_text, os.path.join(folder_path, 'curriculum.png'), 'Curriculum')
                if event_text:
                    save_table_as_png(event_text, os.path.join(folder_path, 'event_params_weights.png'), 'Event Params & Weights')

if __name__ == "__main__":
    main()