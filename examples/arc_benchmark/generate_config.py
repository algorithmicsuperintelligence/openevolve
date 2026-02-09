import os
import yaml
import json


def load_task_as_prompt(task_json, task_num):
    with open(task_json, 'r') as f:
        tasks = json.load(f)
    
    task_id = list(tasks.keys())[int(task_num)]
    task = tasks[task_id]
    train_inputs = [inp["input"] for inp in task['train']]
    train_outputs = [gt["output"] for gt in task['train']]
    test_inputs = [inp["input"] for inp in task['test']]
    
    train_pairs = ""
    for i, (inp, out) in enumerate(zip(train_inputs, train_outputs)):
        train_pairs += f"In {i} - {inp}\nOut {i} - {out}\n"
    
    test_input = ""    
    for i, inp in enumerate(test_inputs):
        test_input += f"In Test {i} - {inp}\n"
    
    prompt = f"""You are participating in a puzzle solving competition. You are an expert at solving puzzles.
Find the common pattern that transforms each input grid into its corresponding output grid...

Your task is to write a python function that transforms each input grid into its corresponding output grid. This function must:
- Apply consistently to ALL training examples
- Be general enough to work on new test cases  
- Be intuitive and easy to understand
- Apply the pattern without referencing specific example numbers

You are provided the following training example grids:
{train_pairs}

You are also provided the test input that you have to succesfully transform into the output using your python code:
{test_input}

Looking carefully at the train input-output pairs, understand the transformation and modify PYTHON functions to generate 2 attempts to solve the task. These python functions will sequentially take each input grid as a numpy array and output the transformed grid as a numpy array. Your solution will then be evaluated against the ground truth output grid.
Remember to only output the modified python functions as your solution."""
    
    return prompt

def generate_config(task_num, task_file, dataset_root="/workspaces/ARC-Evolve/data/arc-prize-2025"):
    task_json = os.path.join(dataset_root, f"arc-agi_{task_file}_challenges.json")
    prompt = load_task_as_prompt(task_json, task_num)
    
    cfg_file = "./base_config.yaml"
    with open(cfg_file, 'r') as file:
        config = yaml.safe_load(file)
    
    config['prompt']['system_message'] = prompt
    
    with open('./config.yaml', 'w') as file:
        yaml.dump(config, file)
        
if __name__ == "__main__":
    TASK_FILE = os.getenv("ARC_TASK_FILE", "training")
    TASK_NUM = os.getenv("TASK_NUM", 0)
    
    generate_config(TASK_NUM, TASK_FILE)
    