from langchain import hub
from langchain.agents import Tool, create_react_agent
from langchain_fireworks import ChatFireworks
from typing import TypedDict, Annotated, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator
from typing import TypedDict, Annotated
from langchain_core.agents import AgentFinish
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.prebuilt import ToolInvocation
from langgraph.graph import END, StateGraph
from langchain_core.agents import AgentActionMessageLog
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain.schema import OutputParserException


st.set_page_config(page_title="LangGraph Agent", layout="wide")

def main():
    # Streamlit UI elements
    st.title("LangGraph Agent + Mistral AI + Custom Tool + Streamlit")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    l = len(st.session_state.messages)
    if l == 1:
        with st.chat_message(st.session_state.messages[0]["role"]):
            st.markdown(st.session_state.messages[0]["content"])
    elif l>1:
        with st.chat_message(st.session_state.messages[0]["role"]):
            st.markdown(st.session_state.messages[0]["content"])
        for i in range(1,l):
            correct_index = i - 1 if i % 2 == 0 else i + 1
            with st.chat_message(st.session_state.messages[correct_index]["role"]):
                st.markdown(st.session_state.messages[correct_index]["content"])

    # Input from user COMMENTED FOR TESTING
    # st.chat_message("assistant").write("Enter the background information about the candidate or the job role:")
    # input_text = st.chat_input(placeholder="provide the information")
    # if input_text is None:
    ques = "Enter the background information about the candidate or the job role"
    input_text = "The candidate has a strong background in software development with expertise in Python and machine learning."
    chat = ChatFireworks(model="accounts/fireworks/models/mixtral-8x7b-instruct", verbose=True, streaming=True)
    
    def toggle_case(word):
        toggled_word = ""
        for char in word:
            if char.islower():
                toggled_word += char.upper()
            elif char.isupper():
                toggled_word += char.lower()
            else:
                toggled_word += char
        return toggled_word

    def sort_string(string):
        return ''.join(sorted(string))

    tools = [
        Tool(
            name = "Search",
            func= lambda context: chat.invoke(context),
            description= "useful for when you need to generate a question to interview a candidate",
        ),
        Tool(
            name = "Ask",
            func= lambda answer: chat.invoke(answer),
            description= "useful for when you need to generate an interview question based on candidate answers",
        ),
        Tool(
            name = "Toggle_Case",
            func = lambda word: toggle_case(word),
            description = "use when you want covert the letter to uppercase or lowercase",
        ),
        Tool(
            name = "Sort String",
            func = lambda string: sort_string(string),
            description = "use when you want sort a string alphabetically",
        ),

    ]

    # prompt_interviewer = "You're an {}. You need to interview a {}. This is the interview so far:\n{}\n\
    # Ask your next question and dont repeat your questions.\
    # Output just the question and no extra text"
    # prompt_interviewer.format("interviewer","candidate","Nothing")

    #initial_prompt = "You're an interviewer. You need to interview a candidate. This is the interview so far:\nNothing\n\
    #Ask your next question and dont repeat your questions.\
    #Output just the question and no extra text"
    #prompt = PromptTemplate.from_template("Tell me a joke about {topic}?")

    template = """
    Generate an interview question for the candidate based on the following context. Your final output should be just the generated question.
    You have access to the following tools:{tools}
    Use the following format:
    Context: the background information about the candidate or the job role
    Thought: you should always think about the context and the information needed
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat 1 time)
    Thought: I now know the best question to ask
    Generated Question: the question to ask the candidate 

    Begin!
    Context: {input}
    Candidate's Answers: {chat_history} 
    Thought:{agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template)

    llm = ChatFireworks(model="accounts/fireworks/models/mixtral-8x7b-instruct",
                        verbose=True) 
    exceptSearch = [tool for tool in tools if tool.name != "Search"]
    agent_runnable = create_react_agent(llm, exceptSearch, prompt)

    class AgentState(TypedDict):
        input: str 
        chat_history: list[BaseMessage]
        agent_outcome: Union[AgentAction, AgentFinish, None]
        return_direct: bool
        intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

    tool_executor = ToolExecutor(tools)

    def run_agent(state):
        """
        #if you want to better manages intermediate steps
        inputs = state.copy()
        if len(inputs['intermediate_steps']) > 5:
            inputs['intermediate_steps'] = inputs['intermediate_steps'][-5:]
        """

        agent_outcome = agent_runnable.invoke(state)
        return {"agent_outcome": agent_outcome}
        

    def execute_tools(state):
        messages = [state['agent_outcome'] ]
        last_message = messages[-1]
        ######### human in the loop ###########   
        # human input y/n 
        # Get the most recent agent_outcome - this is the key added in the `agent` above
        #state_action = state['agent_outcome']
        #human_key = input(f"[y/n] continue with: {state_action}?")
        #if human_key == "n":
        #    raise ValueError

        tool_name = last_message.tool
        arguments = last_message
        if tool_name in ["Search","Ask","Sort","Toggle_Case"]:
            if "return_direct" in arguments:
                del arguments["return_direct"]
        if tool_name == "Ask":
            action = ToolInvocation(
                tool=tool_name,
                tool_input= {"input":k.tool_input for k in messages}
            )
            #state["answers"] = [k.tool_input for k in messages[1:]]
        else:
            action = ToolInvocation(
                tool=tool_name,
                tool_input= last_message.tool_input
            )
        response = tool_executor.invoke(action)
        state["return_direct"] = True
        #state["answers"] = state["answer"].append(last_message.tool_input)
        return {"intermediate_steps": [(state['agent_outcome'],response)]}

    def should_continue(state):
        messages = [state['agent_outcome']] 
        last_message = messages[-1]
        if "Generated Question" in last_message.log:
            return "continue"
        else:
            arguments = state["return_direct"]
            if arguments is True:
                return "final"
            else:
                return "continue"
            
    def first_agent(inputs): 
        action = AgentActionMessageLog(
            tool="Search",
            tool_input = inputs["input"],
            log="",
            message_log=[]
        )
        return {"agent_outcome": action}

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", run_agent)
    workflow.add_node("action", execute_tools)
    workflow.add_node("final", execute_tools)
    # uncomment if you want to always calls a certain tool first
    workflow.add_node("first_agent", first_agent)

    # workflow.set_entry_point("agent")
    # uncomment if you want to always calls a certain tool first
    workflow.set_entry_point("first_agent")

    workflow.add_conditional_edges(

        "agent",
        should_continue,

        {
            "continue": "action",
            "final": "final",
            "end": END
        }
    )


    workflow.add_edge('action', 'agent')
    workflow.add_edge('final', END)
    # uncomment if you want to always calls a certain tool first
    workflow.add_edge('first_agent', 'action')
    app = workflow.compile()

    inputs = {"input": input_text, "chat_history": [], "return_direct": False}
    results = []
    #if st.button("Ready for interview"):
    #if inputs["chat_history"]:
        #st.markdown(inputs["chat_history"][-1])
    try:
        for s in app.stream(inputs):
            result = list(s.values())[0]
            results.append(result)
            #st.write(result)  # Display each step's output
            response = result
    except ValueError as e:
        response = str(e)
        if not response.startswith("Could not parse LLM output: `"):
            raise e
        response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
        ques = response.split(":")[-1].replace('"', '')
        if not(st.session_state.messages):
            with st.chat_message("assistant"):
                st.markdown(ques) 
        st.session_state.messages.append({"role": "assistant", "content": ques})
    
       
    if ans := st.chat_input(placeholder="Answer"):           
        inputs["chat_history"].append(ans)
        with st.chat_message("user"):
            st.markdown(ans)
        st.session_state.messages.append({"role": "user", "content": ans})
        with st.chat_message("assistant"):
            st.markdown(ques)  

    print("another query")

if __name__ == "__main__":
    main()
    
    
