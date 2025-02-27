import re
import random
import ast
import operator


def extract_answer(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer

def extract_num(answer_str):
    try:
        return int(answer_str.strip())
    except:
        return None

def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    S = ground_truth['S']
    T = ground_truth['T']
    B = ground_truth['B']
    
    answer_str = extract_answer(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"base target: {B}")
        print(f"Extracted string: {answer_str}")
        print(f"Solution string: {solution_str}")

    if answer_str is None:
        if do_print:
            print(f"No answer found")
        return 0

    answer_num = extract_num(answer_str)
    if answer_num is None:
        if do_print:
            print(f"Invalid answer num")
        return format_score

    if abs(answer_num - B) < 1e-5:  # Account for floating point precision
        if do_print:
            print(f"Correct equation: {answer_num} = {B}")
        return score
    return format_score


# fake_solution = """
# prompt............... ..... . .. . . . .....
# Assistant:
# dsadsd <think>thinkthinkthinkthink think think thinkthink thinkthink </think><answer> 5       </answer>"""
# ground_truth = {'S': '1010110111000', 'T': 32134, 'B': 5}

# print(compute_score(fake_solution, ground_truth))