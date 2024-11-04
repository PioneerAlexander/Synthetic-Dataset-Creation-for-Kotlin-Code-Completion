import torch
import re
from transformers import StoppingCriteria, StoppingCriteriaList


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops, tokenizer):
        (StoppingCriteria.__init__(self),)
        self.stops = rf"{stops}"
        self.tokenizer = tokenizer

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        last_three_tokens = [int(x) for x in input_ids.data[0][-3:]]
        decoded_last_three_tokens = self.tokenizer.decode(last_three_tokens)

        return bool(re.search(self.stops, decoded_last_three_tokens))


def generate(problem, tokenizer, model):
    criterion = StoppingCriteriaSub(stops="\n}\n", tokenizer=tokenizer)
    stopping_criteria = StoppingCriteriaList([criterion])
    
    problem = tokenizer(problem, return_tensors="pt").to('cuda')
    sample = model.generate(
        **problem,
        max_new_tokens=256,
        min_new_tokens=128,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        num_beams=1,
        stopping_criteria=stopping_criteria,
    )
    
    answer = tokenizer.decode(sample[0], skip_special_tokens=True)
    
    return answer


def clean_answer(code):
    # Clean comments
    code_without_line_comments = re.sub(r"//.*", "", code)
    code_without_all_comments = re.sub(
        r"/\*.*?\*/", "", code_without_line_comments, flags=re.DOTALL
    )
    #Clean signatures
    lines = code.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("fun "):
            return "\n".join(lines[i + 1:])
            
    return code
