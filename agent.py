import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
import bs4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

from langchain_core.runnables import RunnableSequence

llm = ChatOpenAI(model="gpt-3.5-turbo")

query = """
Preciso de um post do facebook focado na história de São José e nas obras disponibilizadas em seu museu. Trazendo detalhes relevantes sobre a história das obras e das peças. 
"""

def researchAgent(query, llm):
    tools = load_tools(["ddg-search", "wikipedia"], llm=llm)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, prompt=prompt, verbose=True)
    webContext = agent_executor.invoke({"input": query})
    print(f"Web Context: {webContext['output']}")
    return webContext['output']

def loadData():
    loader = WebBaseLoader(
        web_paths=["https://saojose.sc.gov.br/museu-historico-municipal-gilberto-gerlach-2/"],
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("page-template"))),
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    return retriever

def getRelevantDocs(query):
    retriever = loadData()
    relevant_documents = retriever.invoke(query)
    print(f"Relevant Documents: {relevant_documents}")
    return relevant_documents

def supervisorAgent(query, llm, webContext, relevant_documents):
    prompt_template = """
    Você é um gerente de uma agência de marketing. Sua resposta final deverá ser modelo de post do facebook completo e detalhado. 
    Utilize o contexto disponibilizado das informações do museu, o input do usuário e também os documentos relevantes para elaborar o roteiro. Retorne com três modelos com cada modelo contendo ideia do post, ideia da imagem e legenda com hashtags.
    Contexto: {webContext}
    Documento relevante: {relevant_documents}
    Usuário: {query}
    Assistente:
    """

    prompt = PromptTemplate(
        input_variables=['webContext', 'relevant_documents', 'query'],
        template=prompt_template
    )

    sequence = RunnableSequence(prompt | llm)
    response = sequence.invoke({"webContext": webContext, "relevant_documents": relevant_documents, "query": query})
    return response

def getResponse(query, llm):
    webContext = researchAgent(query, llm)
    relevant_documents = getRelevantDocs(query)
    response = supervisorAgent(query, llm, webContext, relevant_documents)
    return response

print(getResponse(query, llm))
