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
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.chains import RetrievalQA
import prompts as pr

# page
st.set_page_config(
    page_title="FinAnalyst",
    page_icon="💸",
)

# hide links
st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# load api key
load_dotenv()

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
            del st.session_state.df_cashflow
            del st.session_state.df_pl
            del st.session_state.df_bs
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
                content, ratios, df_cashflow, df_pl, df_bs = get_data(selected_company)
                st.session_state.content = content
                st.session_state.ratios = ratios
                st.session_state.df_cashflow = df_cashflow
                st.session_state.df_pl = df_pl
                st.session_state.df_bs = df_bs

                # create embeddings
                # divide text into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
                chunks = text_splitter.split_text(text=content)

                # create embeddings
                embeddings = GooglePalmEmbeddings()

                # store embeddings
                vectordb = FAISS.from_texts(chunks, embedding=embeddings)
                # store into current state
                st.session_state.vectordb = vectordb

            else:
                content = st.session_state.content
                ratios = st.session_state.ratios
                df_cashflow = st.session_state.df_cashflow
                df_pl = st.session_state.df_pl
                df_bs = st.session_state.df_bs

                vectordb = st.session_state.vectordb

            st.success("Annual Report Analyzed")

        # insights
        # st.info(content)

        # key ratios
        with st.sidebar:
            st.subheader(selected_company)
            st.title(f"₹ {ratios['price']}")
            st.caption("NSE")

            st.subheader("📊 Key Ratios")
            st.info(f"""
            1. ROE: {ratios['roe']}
            2. P/E Ratio(TTM): {ratios['pe']}
            3. EPS(TTM): {ratios['eps']}
            4. P/B Ratio: {ratios['pb']}
            5. Dividend Yield: {ratios['div_yield']}
            6. Debt to Equity: {ratios['d_to_e']}
            """)

        # Financials
        st.write("")
        st.subheader("💹 Financial Statements")
        # Profit and Loss
        st.subheader("Profit and Loss")
        # df_pl.pop(df_pl.columns[-1])
        st.dataframe(df_pl, use_container_width=True)
        # chart data
        chart = pd.DataFrame({
            "Amount Type": ["Total Revenue"] * 5 + ["Total Expense"] * 5,
            "Amount": list(df_pl.iloc[5, 1:6].astype(float).values) + list(df_pl.iloc[14, 1:6].astype(float).values),
            "Month": list(df_pl.columns[1:]) + list(df_pl.columns[1:])
        })
        bar_chart_pl = alt.Chart(chart).mark_bar().encode(
            x=alt.X("Month", sort=None),
            y="Amount",
            color="Amount Type"
        )
        st.altair_chart(bar_chart_pl, use_container_width=True)

        # profits chart data
        st.subheader("Profit/Loss")
        chart = pd.DataFrame({
            "Profit/Loss": ["Total Profit/Loss"] * 5,
            "Amount": list(df_pl.iloc[25, 1:6].astype(float).values),
            "Month": list(df_pl.columns[1:])
        })
        profit_chart_pl = alt.Chart(chart).mark_bar().encode(
            x=alt.X("Month", sort=None),
            y="Amount",
            color="Profit/Loss"
        )
        st.altair_chart(profit_chart_pl, use_container_width=True)

        # Cash Flow
        st.subheader("Cash Flow")
        # df_cashflow.pop(df_cashflow.columns[-1])
        st.dataframe(df_cashflow, use_container_width=True)
        # chart data
        chart = pd.DataFrame({
            "Cash Type": ["Operating Activities"] * 5 + ["Investing Activities"] * 5 + ["Financing Activities"] * 5,
            "Amount": list(df_cashflow.iloc[1, 1:6].astype(float).values) + list(df_cashflow.iloc[2, 1:6].astype(float).values) + list(df_cashflow.iloc[3, 1:6].astype(float).values),
            "Month": list(df_cashflow.columns[1:]) + list(df_cashflow.columns[1:]) + list(df_cashflow.columns[1:])
        })
        bar_chart_cashflow = alt.Chart(chart).mark_bar().encode(
            x=alt.X("Month", sort=None),
            y="Amount",
            color="Cash Type"
        )
        st.altair_chart(bar_chart_cashflow, use_container_width=True)

        # Balance Sheet
        st.subheader("Balance Sheet")
        # df_bs.pop(df_bs.columns[-1])
        st.dataframe(df_bs, use_container_width=True)
        # Add assets and liabilities
        total_non_current_assets = df_bs.iloc[25, 1:6].astype(float).values
        total_current_assets = df_bs.iloc[32, 1:6].astype(float).values
        total_assets = total_non_current_assets + total_current_assets

        total_non_current_liabilities = df_bs.iloc[10, 1:6].astype(float).values
        total_current_liabilities = df_bs.iloc[15, 1:6].astype(float).values
        total_liabilities = total_non_current_liabilities + total_current_liabilities

        # chart
        chart = pd.DataFrame({
            "Amount Type": ["Total Assets"] * 5 + ["Total Liabilities"] * 5,
            "Amount": list(total_assets) + list(total_liabilities),
            "Month": list(df_bs.columns[1:]) + list(df_bs.columns[1:])
        })
        bar_chart_bs = alt.Chart(chart).mark_bar().encode(
            x=alt.X("Month", sort=None),
            y="Amount",
            color="Amount Type"
        )
        st.altair_chart(bar_chart_bs, use_container_width=True)
        st.write("")

        return vectordb


def get_insights(vectordb):
    if 'chain' not in st.session_state:
        # load embeddings
        retriever = vectordb.as_retriever(score_threshold=0.7)
        # question answer chain
        llm = GooglePalm(temperature=0.3)
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever,
                                            return_source_documents=True)
        st.session_state.chain = chain

    if 'response' not in st.session_state:
        query = pr.prompt_insights

        # response
        response = st.session_state.chain(query)

        # insights
        st.balloons()
        st.subheader("💡 Key Insights from Annual Report")
        st.info(response['result'])

        st.write('\n\n')
        st.session_state.response = response['result']
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
        query = pr.prompt_chat + prompt
        try:
            response = st.session_state.chain(query)
            print('Response Chat: ', response['result'])
            st.session_state.chat_result = response['result']
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
