# This script uses LLaMA to generate responses to questions of harmless dataset
# Follow https://github.com/facebookresearch/llama to download the corresponding model and install dependencies

from typing import Optional
import fire
from llama import Llama
import json
import time

INPUT_PATH = "prompt_test.jsonl" # Input prompt jsonl file path
OUTPUT_PATH = "prompt_response.jsonl" # Output file path. Prompt + respnose
SLICE = 10 


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.5,
    top_p: float = 0.9,
    max_seq_len: int = 1024,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    start_time = time.time()
    # Initalize the model
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    
    # Load input file and format the prompt
    dialogs = []
    with open(INPUT_PATH, "r", encoding="utf-8") as input_fp:
        for line in input_fp:
            question = json.loads(line)["Human"].strip()
            prompt = [
                {
                    "role": "user",
                    "content": question,
                }
            ]
            dialogs.append(prompt)
            
    # Generate chatbot responses       
    batch_num = len(dialogs) // SLICE
    results = []
    for i in range(0, batch_num):
        start1 = time.time()
        results = generator.chat_completion(
            dialogs[i * SLICE : (i + 1) * SLICE],  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        print(i)
        print("\n")
        print(time.time() - start1)
        print("\n")

        # Write prompt + response to output file
        start2 = time.time()
        with open(OUTPUT_PATH, "a", encoding="utf-8") as output_fp:
            for dialog, result in zip(dialogs[i * SLICE : (i + 1) * SLICE], results):
                response = {
                    "role": "assistant",
                    "content": result["generation"]["content"],
                }
                conversation = {
                    "prompt": dialog[0],
                    "response": response,
                }
                output_fp.write(json.dumps(conversation) + "\n")
        print("Write time:\n")
        print(time.time() - start2)
        print("\n")
    
    print("Total time elapsed:\n")
    print(time.time() - start_time)
if __name__ == "__main__":
    fire.Fire(main)
