import json
import os
from src.params import parse_arguments
from src.utils import get_tool_dict, get_s1_score
from src.output_parsing import evaluate_output_finetuned
from tqdm import tqdm 
from src.prompts.agent_prompts import PROMPT_DICT
from src.llm_client import ATTACK_MODEL, LLM_TEMPERATURE, chat_completion_with_retry, extract_chat_content
from pydantic import BaseModel
from typing import Optional, List, Literal

class Function(BaseModel):
    arguments: str
    name: str

class ChatCompletionMessageToolCall(BaseModel):
    id: str
    function: Function
    type: Literal["function"]

class FunctionCall(BaseModel):
    arguments: str
    name: str

class ChatCompletionMessage(BaseModel):
    content: Optional[str]
    role: Literal["assistant"]
    function_call: Optional[FunctionCall] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None


def predict_one_case(system_prompt, user_prompt, params, item, tool_dict, step_2=False):
    available_tool_names = [item['User Tool']] + item['Attacker Tools']
    available_tools = [tool_dict[tool] for tool in available_tool_names]
    
    messages = [
        {"role": "system", "content":system_prompt},
        {"role": "user", "content": item['User Instruction']},
    ]
    
    call_id = 'call_dx6NRJIZOLS2GS7HtIFxVpyG' # random one
    function_name = item['User Tool']
    message = ChatCompletionMessage(
        content = None,
        role = 'assistant',
        tool_calls = [ChatCompletionMessageToolCall(id=call_id, function=Function(arguments = json.dumps(item['Tool Parameters']), name = function_name), type='function')]
    )
    messages.append(message)
    
    messages.append(
        {
            "tool_call_id": call_id,
            "role": "tool",
            "name": function_name,
            "content": json.dumps(item['Tool Response']),
        }
    ) 
    
    if step_2:
        tool_call = item['tool_calls'][0]
        message = ChatCompletionMessage(
            content = item['output'],
            role = 'assistant',
            tool_calls = [ChatCompletionMessageToolCall(id=tool_call['id'], function=Function(arguments = json.dumps(item['Attacker Tool Parameters']), name = tool_call['function']['name']), type='function')]
        )
        messages.append(message)
        messages.append(
            {
                "tool_call_id": tool_call['id'],
                "role": "tool",
                "name": tool_call['function']['name'],
                "content": item['Attacker Tool Response'],
            }
        )   

    _, error, response = chat_completion_with_retry(
        messages=messages,
        model=params.get('model_name') or ATTACK_MODEL,
        tools=available_tools,
        temperature=LLM_TEMPERATURE,
        label=f"finetuned:{params.get('model_name') or ATTACK_MODEL}",
    )
    if error or response is None:
        return None, None
    _, parse_error = extract_chat_content(response)
    if parse_error:
        return None, None
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    if isinstance(tool_calls, List):
        tool_calls = [item.model_dump() for item in response_message.tool_calls]
    return response_message.content, tool_calls


def main(params):
    print(params)    
    system_prompt, user_prompt = PROMPT_DICT[params['prompt_type']]
    output_dir = f"./results/finetuned_{params['model_name']}_{params['prompt_type']}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data_dir = './data'
    
    tool_dict = get_tool_dict(gpt_format=True)
    
    file_name = f"test_cases_ds_{params['setting']}.json"
    test_case_file = os.path.join(data_dir, file_name)
    output_file = os.path.join(output_dir, file_name)

    with open(test_case_file, 'r') as f:
        data = json.load(f)

    with open(output_file, 'w') as f:
        for item in tqdm(data):
            output, tool_calls = predict_one_case(system_prompt, user_prompt, params, item, tool_dict)
            item['output'] = output
            item['tool_calls'] = tool_calls
            parsed_output = evaluate_output_finetuned(output, item['Attacker Tools'][0], item['User Tool'], tool_calls)
            item['eval'] = parsed_output['eval']

            f.write(json.dumps(item)+'\n')

    scores = get_s1_score(output_file)
    print(json.dumps(scores, indent=True))

if __name__ == "__main__":
    params = parse_arguments(agent_type="finetuned")

    main(params)
