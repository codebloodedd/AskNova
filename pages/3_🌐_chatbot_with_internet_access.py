import utils
import streamlit as st

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# Tavily integration
from langchain_tavily import TavilySearch, TavilyExtract

from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool

st.set_page_config(page_title="ChatNet", page_icon="🌐")
st.header('Chatbot with Internet Access')
st.write('Equipped with internet access, enables users to ask questions about recent events')
st.write('[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/codebloodedd/AskNova/blob/main/pages/3_%F0%9F%8C%90_chatbot_with_internet_access.py)')

class InternetChatbot:

    def __init__(self):
        utils.sync_st_session()
        self.llm = utils.configure_llm()

    def setup_agent(_self):
        # --- Replace DuckDuckGo with Tavily ---
        # Create Tavily search tool; tweak max_results/topic as needed
        tavily_search = TavilySearch(
            max_results=5,
            topic="general",        # optional: "general", "news", etc.
        )

        # If you want content extraction from urls, also add TavilyExtract:
        tavily_extract = TavilyExtract()

        # Wrap as Tool objects (keeps your existing agent setup style)
        tools = [
            Tool(
                name="TavilySearch",
                func=tavily_search.run,
                description="Useful for when you need to answer questions about current events. Use targeted queries."
            ),
            Tool(
                name="TavilyExtract",
                func=tavily_extract.run,
                description="Extract and return page content for a given URL."
            ),
        ]

        # Get the prompt - can modify this
        prompt = hub.pull("hwchase17/react-chat")

        # Setup LLM and Agent
        memory = ConversationBufferMemory(memory_key="chat_history")
        agent = create_react_agent(_self.llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=False)
        return agent_executor, memory

    @utils.enable_chat_history
    def main(self):
        agent_executor, memory = self.setup_agent()
        user_query = st.chat_input(placeholder="Ask me anything!")
        if user_query:
            utils.display_msg(user_query, 'user')
            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container())
                result = agent_executor.invoke(
                    {"input": user_query, "chat_history": memory.chat_memory.messages},
                    {"callbacks": [st_cb]}
                )
                response = result["output"]
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)
                utils.print_qa(InternetChatbot, user_query, response)


if __name__ == "__main__":
    obj = InternetChatbot()
    obj.main()
