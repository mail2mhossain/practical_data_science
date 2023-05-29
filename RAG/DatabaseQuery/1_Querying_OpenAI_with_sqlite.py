from langchain import SQLDatabase, SQLDatabaseChain, OpenAI


OpenAI_API_KEY = "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"

db = SQLDatabase.from_uri("sqlite:///Chinook_Sqlite.sqlite")

llm = OpenAI(
    openai_api_key=OpenAI_API_KEY, temperature=0, verbose=True
)  # model_name="gpt-3.5-turbo",


# query = "How many employees are there?"
# query = "How many albums by Aerosmith?"
query = "How many employees are also customers?"

db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=False) # use_query_checker=

# Return Intermediate Steps
#db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=False, use_query_checker=True, return_intermediate_steps=True)

result = db_chain.run(query)


print(result)
# result["intermediate_steps"]


