"""Inspect fitted tree structure independently of requested hyperparameters."""
import json
import re


def model_details(model, backend):
    if backend == 'lightgbm':
        dump = model.dump_model()
        trees = [item['tree_structure'] for item in dump['tree_info']]
        objective = dump.get('objective')
        rounds = model.current_iteration()
        config = dict(model.params)
        # Loaded LightGBM files retain the parameter block even when params is empty.
        for name, value in re.findall(r'^\[([^:\n]+): (.*)\]$', model.model_to_string(), re.MULTILINE):
            config.setdefault(name, value)
    else:
        trees = [json.loads(tree) for tree in model.get_dump(dump_format='json')]
        config = json.loads(model.save_config())
        objective = config['learner']['objective']['name']
        rounds = model.num_boosted_rounds()
    depths, leaves = [], []
    for tree in trees:
        stack, depth_max, leaf_count = [(tree, 0)], 0, 0
        while stack:
            node, depth = stack.pop()
            children = ([node['left_child'], node['right_child']] if 'left_child' in node
                        else node.get('children', []))
            if children:
                stack.extend((child, depth + 1) for child in children)
            else:
                leaf_count += 1
                depth_max = max(depth_max, depth)
        depths.append(depth_max)
        leaves.append(leaf_count)
    return {
        'backend': backend, 'objective': objective, 'number_of_trees': len(trees),
        'boosting_rounds': rounds, 'number_of_features': model.num_feature() if backend == 'lightgbm' else model.num_features(),
        'tree_depth': {'min': min(depths) if depths else None, 'max': max(depths) if depths else None,
                       'mean': sum(depths) / len(depths) if depths else None, 'per_tree': depths},
        'leaves_per_tree': leaves, 'loaded_model_configuration': config,
        'configuration_note': 'Loaded model/runtime configuration is not necessarily the original training configuration. Verified training parameters are reported separately.',
        'depth_definition': 'Number of edges from root to deepest leaf; a constant tree has depth zero.',
    }
