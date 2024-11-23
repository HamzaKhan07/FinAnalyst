import streamlit as st
from data import companies
import yfinance as yf
import plotly.express as px
import datetime
from data_helper import get_data
import pandas as pd
import altair as alt
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import prompts as pr

# page
st.set_page_config(
    page_title="FinAnalyst",
    page_icon="💸",
)


# load api key
# load_dotenv()

api_key = st.secrets["api_key"]

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


def start():
    # header
    st.title("💸 FinAnalyst")
    st.write(
        "Get core insights and chat with the Annual Reports 📈 of Indian companies and make your investment strategy better 💡 ")

    # select company
    option = st.selectbox(
        "Type a company symbol",
        companies,
        index=None,
        placeholder="BAJAJ-AUTO",
    )

    st.write("")

    # end date data is excluded
    if option:
        # change date
        option_date = st.selectbox(
            "Change duration",
            ["1w", "1m", "6m", "1y", "5y"],
            index=1,
            placeholder="1m",
        )
        if option_date:
            if option_date == '1w':
                days_to_deduct = 7
            elif option_date == '1m':
                days_to_deduct = 30
            elif option_date == '6m':
                days_to_deduct = 30 * 6
            elif option_date == '1y':
                days_to_deduct = 365
            elif option_date == '5y':
                days_to_deduct = 365 * 5

            current_date = datetime.datetime.today()
            previous_date = datetime.datetime.today() - datetime.timedelta(days=days_to_deduct)
            start_date = current_date.date()
            end_date = previous_date.date()

        selected_company = option
        # save selected company
        if 'selected_company' not in st.session_state:
            st.session_state.selected_company = selected_company

        stock_data = yf.download(f'{selected_company}.NS', start=end_date, end=start_date)
        print(stock_data)
        fig = px.line(stock_data, x=stock_data.index, y=stock_data['Adj Close'], title=selected_company)
        st.plotly_chart(fig)

        # get data
        if selected_company!=st.session_state.selected_company:
            # delete previous data
            del st.session_state.content
            del st.session_state.ratios
            del st.session_state.messages
            del st.session_state.selected_company
            # clear session state variables
            if 'vectordb' in st.session_state:
                del st.session_state.vectordb
            if 'chain' in st.session_state:
                del st.session_state.chain
            if 'response' in st.session_state:
                del st.session_state.response
            if 'messages' in st.session_state:
                del st.session_state.messages

            # save new company
            st.session_state.selected_company = selected_company

        with st.spinner("Analyzing Annual Report..."):
            if 'content' not in st.session_state:
                content, ratios = get_data(selected_company)

                # create embeddings
                # divide text into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
                chunks = text_splitter.split_text(text=content)

                # create embeddings
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

                # store embeddings
                vectordb = FAISS.from_texts(chunks, embedding=embeddings)
                # store into current state
                st.session_state.vectordb = vectordb

            else:
                content, ratios = st.session_state.content, st.session_state.ratios

                vectordb = st.session_state.vectordb

            st.success("Annual Report Analyzed")

        # insights
        # st.info(content)

        # key ratios
        with st.sidebar:
            st.subheader(selected_company)
            st.title(f"₹ {ratios['current_price']}")
            st.write(ratios['industry'])
            st.write('Rating: ', ratios['recommendation_key'])

            st.subheader("📊 Key Ratios")
            st.info(f"""
                        1. ROE: {ratios['roe']}
                        2. Forward PE: {ratios['forward_pe']}
                        3. Debt to Equity: {ratios['db_to_eq']}
                        4. P/B Ratio: {ratios['pb']}
                        """)

        return vectordb


def get_insights(vectordb):
    if 'chain' not in st.session_state:
        # prompt template
        prompt_template = """
                    Answer the question as detailed as possible from the provided context, make sure to provide all the details.\n\n
                    Context:\n {context}?\n
                    Question: \n{question}\n

                    Answer:
                    """
        # question answer chain
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)  # models/text-bison-001 gemini-pro
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        st.session_state.chain = chain

    if 'response' not in st.session_state:
        query = pr.prompt_insights
        docs = st.session_state.vectordb.similarity_search(query)

        # response
        response = st.session_state.chain(
            {"input_documents": docs, "question": query}
            , return_only_outputs=True)

        # insights
        # print(docs)

        # insights
        st.balloons()
        st.subheader("💡 Key Insights from Annual Report")
        st.info(response["output_text"])

        st.write('\n\n')
        st.session_state.response = response['output_text']
    else:
        st.subheader("💡 Key Insights from Annual Report")
        st.info(st.session_state.response)

    return st.session_state.chain


def handle_chat(prompt):
    # chat
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # add loading
    with st.spinner('Loading...'):
        query = prompt
        try:
            docs = st.session_state.vectordb.similarity_search(query)

            response = st.session_state.chain(
                {"input_documents": docs, "question": query}
                , return_only_outputs=True)

            print('Response Chat: ', response['output_text'])
            st.session_state.chat_result = response['output_text']
        except:
            st.session_state.chat_result = "Sorry, I'm not able to assist you with that"

    response = st.session_state.chat_result
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == '__main__':
    db = start()
    if db:
        chain = get_insights(db)
        if chain:
            prompt = st.chat_input("Talk with Annual Report")
            if prompt:
                handle_chat(prompt)
