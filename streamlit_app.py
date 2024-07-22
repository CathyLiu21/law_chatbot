__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from langchain_chroma import Chroma #0.1.2
from langchain.chains.query_constructor.base import AttributeInfo #0.2.9, langchain-community==0.2.7
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI, OpenAI #0.1.16
from langchain_core.documents import Document #0.2.20
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter #0.2.2
import pandas as pd
import numpy as np
import os
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import openai #1.35.14
import requests
from bs4 import BeautifulSoup



def load_data():
            url1 = "https://docs.google.com/spreadsheets/d/1ONQo3Ep0m91Xc8r2aHq0hYkrS7MJxIiA/edit?usp=sharing&ouid=101885794421148722340&rtpof=true&sd=true"
            url_for_pandas1 = url1.replace("/edit?usp=sharing", "/export?format=xlsx")
            df_news = pd.read_excel(url_for_pandas1, engine='openpyxl')

            df_news['Report Date'] = pd.to_datetime(df_news['Report Date'])
            df_news['Report Date'] = df_news['Report Date'].dt.strftime('%Y-%m-%d').astype(str)

            url2 = "https://drive.google.com/file/d/1O2Mif1J5BXndCtH-CtY4ZpqBoaozWyOx/view?usp=sharing"
            file_id = url2.split('/d/')[1].split('/')[0]
            url_for_pandas2 = f"https://drive.google.com/uc?export=download&id={file_id}"
            df_clients = pd.read_csv(url_for_pandas2)

            df_news['Content_all'] = df_news.apply(create_news_document, axis=1)
            df_news['Type'] = 'NewsArticle'

            df_clients['Description'] = df_clients['SummitGuard Clients'].apply(create_client_document)
            df_clients['Type'] = 'SummitGuard Client'

            return df_news, df_clients

def create_news_document(row):
            document = f"Title: {row['Title']}\n"
            document += f"Report Date: {row['Report Date']}\n"
            document += f"URL: {row['URL']} \n"
            document += f"Full Article: {row['Full Article']}\n"
            return document

def create_client_document(client):
            return f"Company: {client}. This company is a client of SummitGuard's retirement services."






# Web scraping function
def scrape_article(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to load page {url}")
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    article_text = " ".join([para.text for para in paragraphs])
    return article_text

# Summarization function
def summarize_content(text):
    prompt = f"""
    Summarize the following ERISA litigation news article in a structured format:

    Article Text:
    {text}

    Please provide a summary in the following format:
   1. Case Name: <Case Name>
   2. Parties Involved: <Parties Involved>
   3. Key Issues: <Key Issues>
   4. Outcome/Settlement: <Outcome/Settlement>
   5. Key reasons for this lawsuit: <Key reasons for this lawsuit>

    Summary:
    """
    client = openai.OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
            )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a legal expert specializing in ERISA litigation. Summarize the given article concisely and accurately."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )

    return response.choices[0].message.content

def main():
    # Streamlit app
    st.set_page_config(page_title="ERISA Litigation and Client Assistant", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
    st.title('ERISA Litigation and Client Assistant')
    st.subheader(':blue[Welcome! This app allows you to interact with ERISA lawsuits news and SummitGuard client data.]')





    # Input field for OpenAI API key
    with st.sidebar:
        st.title('OpenAI key')
        openai_key = st.text_input('Enter OpenAI key:', type='password')
        if not openai_key:
            st.warning('Please enter your OpenAI key!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
        # Sidebar for choosing functionality

        st.title('Choose a function')
        option = st.selectbox("Choose from:", ("ERISA Litigation Chat", "Web Article Summarization"))

    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

        # Load the ERISA news data

        if option == "ERISA Litigation Chat":
           st.header("ERISA Litigation Chat")
          # st.markdown('<p class="big-font">Chat with our AI about ERISA litigation topics.</p>', unsafe_allow_html=True)
           st.subheader("Chat with our AI about ERISA litigation topics")
           st.subheader("Example questions:")
           st.markdown(
        """
        - What is ERISA litigation?
        - List a few companies that are current SummitGuard clients in your database.
        - Is XX (e.g., Humana) company a current SummitGuard client?
        - Find a company that appears in news but not a SummitGuard Client.
        - Summarize the news about XX (given the company found in the previous question) ERISA lawsuit.
        - Summarize the reason for the XX ERISA lawsuit briefly.
        - Summarize one piece of news found in May 2024 in the database.
        - Find the URL of XX ('title' or company name) ERISA lawsuit in the metadata.
        - What's the title of the news of the lawsuit on March 18, 2024 in your database?
        """
    )
           st.markdown('''
     <style>
   [data-testid="stMarkdownContainer"] ul{
    list-style-position: inside;
   }
    </style>
   ''', unsafe_allow_html=True)
           st.subheader("Type 'exit' to end the conversation.")



           df_news, df_clients = load_data()

           text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len
            )

           text_column1 = 'Content_all'
           documents = [Document(page_content=text, metadata={'title': title, 'url': url, 'date': date, 'Type': n_type})
                    for text, title, url, date, n_type in zip(df_news[text_column1], df_news['Title'], df_news['URL'], df_news['Report Date'], df_news['Type'])]

           all_splits_news = text_splitter.split_documents(documents)

           text_column2 = 'Description'
           documents2 = [Document(page_content=text, metadata={'ClientName': client, 'Type': c_type})
                    for text, client, c_type in zip(df_clients[text_column2], df_clients['SummitGuard Clients'], df_clients['Type'])]

           all_splits_clients = text_splitter.split_documents(documents2)

           all_chunks = all_splits_news + all_splits_clients

           model_name = 'text-embedding-3-large'
           embeddings = OpenAIEmbeddings(model=model_name)

           vectordb = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            )

           metadata_field_info = [
            AttributeInfo(name="date", description="The date of the NewsArticle of ESRISA lawsuits", type="string"),
            AttributeInfo(name="Type", description="types of the content from ESRISA lawsuits NewsArticle and SummitGuard Client", type="string"),
            AttributeInfo(name="title", description="The title of the NewsArticle of ESRISA lawsuits", type="string"),
            AttributeInfo(name="url", description="The url of the NewsArticle of ESRISA lawsuits", type="string"),
            AttributeInfo(name="ClientName", description="The name of the SummitGuard Client", type="string"),
           ]

           document_content_description = "ERISA lawsuits news articles and clients"
           llm = OpenAI(temperature=0)
           retriever_sq = SelfQueryRetriever.from_llm(
            llm, vectordb, document_content_description, metadata_field_info, verbose=True
            )

           template = """
        You're an assistant answering questions related to ERISA lawsuits news and SummitGuard clients in the database.
        If the questions start with 'How many', follow the steps below to ensure a thorough and clear response:
        1. Find all ERISA lawsuits that occurred in the given time range based on the date in the metadata.
        2. Count the total number of ERISA news found in step 1.
        3. Output the count in step 2. Remember to exclude duplicated ERISA news you found.
        For other types of questions, follow the steps below:
        1. Identify the relevant pieces of context (ERISA news) related to the question by searching both the page_content and metadata sentence by sentence.
        2. Summarize the relevant information concisely.
        3. Formulate a final answer based on the summarized information.
        If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
        {context}
        Question: {question}
        Helpful Answer:"""
           QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
           llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
           qa_chain_p = RetrievalQA.from_chain_type(
            llm, retriever=retriever_sq, return_source_documents=True, chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
            )



           def generate_response(query) -> str:
             context = qa_chain_p.invoke({"query": query})
             context = context["result"]
             client = openai.OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
            )
             messages = [{"role": "system", "content": context}, {"role": "user", "content": query}]

             response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=300
            )

             return response.choices[0].message.content.strip()
           def clear_history():
             st.session_state.messages = []



           if st.button('Clear Conversation History'):
               clear_history()
               st.write("Conversation history cleared.")

           if 'messages' not in st.session_state:
               st.session_state.messages = [{"role": "assistant", "content": "I am your ERISA Litigation and SummitGuard Client assistant. Ask me something!"}]

        # Display the prior chat messages
           for message in st.session_state.messages:
               with st.chat_message(message["role"]):
                   st.write(message["content"])

        # User-provided prompt
           prompt = st.chat_input(disabled=not openai_key)
           if prompt:
              if prompt.lower() == 'exit':
                st.write("Thank you for using the ERISA Litigation and Client Assistant. Goodbye!")
              else:
                 st.session_state.messages.append({"role": "user", "content": prompt})
                 with st.chat_message("user"):
                    st.write(prompt)

                # If last message is not from assistant, generate a new response
                 if st.session_state.messages[-1]["role"] != "assistant":
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response = generate_response(prompt)
                            st.write(response)
                    message = {"role": "assistant", "content": response}
                    st.session_state.messages.append(message)  # Add response to message history
        elif option == "Web Article Summarization":
           st.header("Web Article Summarization")
           st.markdown('<p class="big-font">Enter a URL of the ERISA litigation news article to summarize it.</p>', unsafe_allow_html=True)

           url = st.text_input("Enter the URL of the news article:")
           if st.button("Summarize Article"):
             if url:
                try:
                    with st.spinner('Scraping and summarizing the article...'):
                        article_text = scrape_article(url)
                        st.success("Article text scraped successfully.")
                        summary = summarize_content(article_text)

                        st.subheader("Summary of the article:")

                        st.markdown(f'<div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px;">{summary}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
             else:
                st.warning("Please enter a URL before clicking the Summarize button.")


    else:
        st.write('Please enter your OpenAI API Key to proceed.')

if __name__ == "__main__":
    main()
