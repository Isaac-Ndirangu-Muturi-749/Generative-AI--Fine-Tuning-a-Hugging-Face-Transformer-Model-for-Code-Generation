
# Generative AI: Fine-Tuning a Hugging Face Transformer Model for Code Generation

![Project Image](llm.png)

## Table of Contents
- [Introduction](#introduction)
- [About the Project](#about-the-project)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
- [Usage](#usage)
    - [Training the Model](#training-the-model)
    - [Generating Code](#generating-code)
- [Quantization and Progressive Embedding Fine-Tuning (PEFT)](#quantization-and-progressive-embedding-fine-tuning-peft)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction
This project explores the fine-tuning of Hugging Face's transformer model for code generation applications. Leveraging generative AI, the project aims to understand and generate code-related responses based on user prompts. The model used in this project is fine-tuned on the evol-codealpaca-v1 dataset using quantization and Progressive Embedding Fine-Tuning (PEFT) for optimization.

## About the Project
- Model Architecture: Salesforce's Hugging Face Transformer Model (codegen-350M-mono) 
- Training Data: evol-codealpaca-v1 dataset
- Training Methodology:
  - Quantization Configuration:
    - Quantization Type: 4-bit
    - Compute Dtype: float16
    - Double Quant: Enabled
  - PEFT Configuration:
    - Lora Alpha: 16
    - Lora Dropout: 0.5
    - Bias: None
    - Task Type: CAUSAL_LM
    - Target Modules: q_proj, v_proj, k_proj
- Target Application: Code generation

## Getting Started

### Prerequisites
- Python
- Dependencies:
  - `torch`
  - `transformers`
  - `datasets`
  - `trl`
  - `peft`
  - `bitsandbytes`
  - `accelerate`
  - `huggingface_hub`
  - `wandb`
  - `einops`
  - `scipy`

### Installation
1. Clone the repository.
   ```sh
   git clone https://github.com/your_username/your_repo.git
   ```
2. Install the required libraries.
   ```sh
   pip install -q -U torch
   pip install -q -U transformers
   pip install -q -U datasets
   pip install -q -U trl
   pip install -q -U git+https://github.com/huggingface/peft.git
   pip install -q -U bitsandbytes
   pip install -q -U accelerate
   pip install -q -U huggingface_hub
   pip install -q -U wandb
   pip install -q -U einops
   pip install -q -U scipy
   ```

## Usage

### Training the Model
- Details on how to train the model can be found in [TRAINING.md](TRAINING.md).

### Generating Code
- Use the following code snippet to generate code from the trained model.
   ```python
   # Insert code to generate code here
   device = "cuda:0"
    model.to(device)

    prompts = [
        "Write code to print 'Hello, world!' in Python.",
    ]

    # Initialize an empty list to store the model's responses
    responses = []

    # Maximum token limit for responses
    max_token_limit = 100  # Adjust this limit as needed

    # Loop through the prompts and generate responses
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_length=max_token_limit, num_return_sequences=1, no_repeat_ngram_size=2)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        responses.append(response)

    # Print the responses
    for i, response in enumerate(responses):
        print(f"PROMPT {i + 1}:\n{prompts[i]}\nRESPONSE:\n{response}\n")

   ```

## Quantization and Progressive Embedding Fine-Tuning (PEFT)
- Details on quantization and PEFT configuration can be found in [PEFT.md](PEFT.md).

## Results
- Provide insights into the project's results. Include code samples and example outputs.

PROMPT 1:
Add 5 and 7.
RESPONSE:
Add 5 and 7.
#
print(f"The sum of the numbers is {sum(numbers)}")

n = int(input("Enter the number of elements: "))
for i in range(0, n):
    print("Element: ", end="")

    element = input()


PROMPT 2:
Multiply 3 by 9.
RESPONSE:
Multiply 3 by 9.

# def multiply(a, b):
    
def multiply_3(x, y):

    return x * y


print(multipy(3, 9))


PROMPT 3:
Write code to print 'Hello, world!' in Python.
RESPONSE:
Write code to print 'Hello, world!' in Python.
#
print('Hello', 'world!')

"""
Output:
Hello world!
 """


PROMPT 4:
Calculate the square root of 16.
RESPONSE:
Calculate the square root of 16.

# In[ ]:


import math
print(math.sqrt(16))



PROMPT 5:
Find the result of 12 divided by 4.
RESPONSE:
Find the result of 12 divided by 4.

# In[ ]:


def divide(x, y):
    return x / y
print(divide(12, 4))



PROMPT 6:
Write a Python program to check if a number is even or odd.
RESPONSE:
Write a Python program to check if a number is even or odd.

# def is_even(num):
    
 # if num % 2 == 0:
        # return True
  
   # else: 
       #return False
 


def isOdd(number):

    if number %2 ==0:

        return True

   else:  

       return  False

## Conclusion:

Overall, the model performed decently.
The responses contain both relevant information based on the prompts and unrelated code.
It seems that the model generated code beyond the desired response.
In the next steps, we may want to train for more epochs to get better results.

Follow me on Twitter 🐦, connect with me on LinkedIn 🔗, and check out my GitHub 🐙. You won't be disappointed!

👉 Twitter: https://twitter.com/NdiranguMuturi1  
👉 LinkedIn: https://www.linkedin.com/in/isaac-muturi-3b6b2b237  
👉 GitHub: https://github.com/Isaac-Ndirangu-Muturi-749  
