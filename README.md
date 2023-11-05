# Resume-Based Interview Preparation Tool

## Project Overview

The Resume-Based Interview Preparation Tool is a software application designed to streamline the interview process by helping interviewers generate relevant and meaningful questions based on a candidate's resume or portfolio page. This tool aims to make interviewers' jobs easier and improve the quality of interviews by focusing on deeper and more insightful inquiries.

## Features

- **Resume Analysis**: The tool allows interviewers to upload a candidate's resume or portfolio page, which is then analyzed to extract relevant information.

- **Question Generation**: Based on the resume content, the tool generates a set of interview questions designed to delve into the candidate's qualifications, experience, and skills.

- **Question Relevance**: The tool identifies and organizes questions that are pertinent to the candidate's resume, ensuring that the interview process is more effective and efficient.

- **Customization**: Users can tailor the generated questions to their specific needs or preferences, allowing for personalized interview preparation.

## Supported Models
- [`bert-small-uncased-whole-word-masking-squad-int8-0002`](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/bert-small-uncased-whole-word-masking-squad-int8-0002#bert-small-uncased-whole-word-masking-squad-int8-0002) is supported by this program.
  - This is a small BERT-large like model distilled and quantized to INT8 on SQuAD v1.1 training set from larger BERT-large model (bert-large-uncased-whole-word-masking) provided by the Transformers library) and tuned on SQuAD v1.1 training set. 
  - Tokenization occurs using the BERT tokenizer and the enclosed vocab.txt dictionary file. Input is to be lower-cased before tokenizing.
  - However, the model is very limited and sensitive for the input. Please put appropriate format and amount of input later. Otherwise, the algorithm will not be able to find it. 
  - For more information, refer to the Input section of [BERT model documentation](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/bert-small-uncased-whole-word-masking-squad-int8-0002#input).

## Getting Started

### Prerequisites

- Python
- Install required packages (list them in a requirements.txt file)

### Installation

1. Clone this repository to your local machine:
```bash
git clone https://github.com/serinryu/interviewhelper_openvino.git
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running

```
usage: bert_question_answering_demo.py [-h] -i INPUT

Options:
  -h, --help            Show this help message and exit.
  -i INPUT, --input INPUT  Required. URL to a page with context

```

#### Example CMD Line
```python
python3 resume_interview_tool.py -i https://dynamicfolio.vercel.app
```
You should input the candidate's resume or portfolio page and type the questions. 

```python
python3 resume_interview_tool.py -i https://dynamicfolio.vercel.app

	Write a question (q to exit): Tell me one of your projects bulit with Java.
	Answer: (.....)
	Score: 0.99
	Time: 0.14s
```

Sample questions:
* Which tech stack do you have?
* When you interned at Intel in 2023, what were your primary responsibilities and tasks?
* Did you receive any academic awards or honors in high school for outstanding grades?