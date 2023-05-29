from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import rich
from rich.console import Console
from rich.panel import Panel
from rich import print

########## INITIALIZE RICH CONSOLE  ##################
console = Console()

local_path='Z:/MHossain_OneDrive/OneDrive/ChatGPT/LLM/Models/ggml-gpt4all-j-v1.3-groovy.bin'
console.print(f"[italic black] with {local_path} \n")
callbacks = [StreamingStdOutCallbackHandler()]
console.print("[yellow bold] Inizializing Summarization Chain") 
llm = GPT4All(model=local_path, callbacks=callbacks, backend='gptj', verbose=True)

template = """Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "How does Shakespeare present the love between Romeo and Juliet?"
llm_chain.run(question)