import streamlit as st
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

