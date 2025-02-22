import streamlit as st
import json
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.tracers import LangChainTracer
from langchain.prompts import PromptTemplate
from langsmith import Client
from datetime import datetime
import os
from dotenv import load_dotenv
from supabase import create_client
import uuid
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone
from pinecone import Pinecone as PineconeClient

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_ANON_KEY")  # Use anon key for regular operations
supabase = create_client(supabase_url, supabase_key)

# Initialize Pinecone
pc = PineconeClient(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)

index_name = "pdf-embeddings"

# Initialize vector store
embeddings = OpenAIEmbeddings()
vectorstore = Pinecone(
    pc.Index(index_name),
    embeddings,
    "text"  # This is the field in metadata that contains the text content
)

print(f"Successfully connected to Pinecone index: {index_name}")

# Initialize session state for user ID
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

# Initialize session state for conversation tracking
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = str(uuid.uuid4())
if "last_message_id" not in st.session_state:
    st.session_state.last_message_id = None

def save_message_to_supabase(user_id, role, content, model, parent_message_id=None, custom_prompt_id=None, created_from_prompt=False):
    """Save a chat message to Supabase with conversation tracking."""
    try:
        message_data = {
            "conversation_id": st.session_state.current_conversation_id,
            "user_id": user_id,
            "role": role,
            "content": content,
            "model": model,
            "timestamp": datetime.utcnow().isoformat(),
            "parent_message_id": parent_message_id,
            "custom_prompt_id": custom_prompt_id,
            "created_from_prompt": created_from_prompt
        }
        result = supabase.table("chat_messages").insert(message_data).execute()
        # Store the message ID if it's a user message
        if role == "user":
            st.session_state.last_message_id = result.data[0]['id']
        return result.data[0]['id']
    except Exception as e:
        st.error(f"Error saving message: {str(e)}")
        return None

def save_conversation_analytics(user_id, model, prompt_tokens, completion_tokens, duration_ms):
    """Save conversation analytics to Supabase."""
    try:
        analytics_data = {
            "user_id": user_id,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "duration_ms": duration_ms,
            "timestamp": datetime.utcnow().isoformat()
        }
        supabase.table("conversation_analytics").insert(analytics_data).execute()
    except Exception as e:
        st.error(f"Error saving analytics: {str(e)}")

def save_user_preferences(user_id, preferred_model):
    """Save user preferences to Supabase."""
    try:
        preferences_data = {
            "user_id": user_id,
            "preferred_model": preferred_model,
            "last_updated": datetime.utcnow().isoformat()
        }
        # Upsert preferences (insert if not exists, update if exists)
        supabase.table("user_preferences").upsert(preferences_data).execute()
    except Exception as e:
        st.error(f"Error saving preferences: {str(e)}")

def get_user_preferences(user_id):
    """Get user preferences from Supabase."""
    try:
        response = supabase.table("user_preferences").select("*").eq("user_id", user_id).execute()
        if response.data:
            return response.data[0]
        return None
    except Exception as e:
        st.error(f"Error getting preferences: {str(e)}")
        return None

def save_custom_prompt(user_id, prompt_name, prompt_content):
    """Save a custom prompt to Supabase."""
    try:
        prompt_data = {
            "user_id": user_id,
            "name": prompt_name,
            "content": prompt_content,
            "created_at": datetime.utcnow().isoformat()
        }
        supabase.table("custom_prompts").insert(prompt_data).execute()
    except Exception as e:
        st.error(f"Error saving prompt: {str(e)}")

def get_custom_prompts(user_id):
    """Get custom prompts for a user from Supabase."""
    try:
        response = supabase.table("custom_prompts").select("*").execute()
        return response.data
    except Exception as e:
        st.error(f"Error getting prompts: {str(e)}")
        return []

def delete_custom_prompt(user_id, prompt_id):
    """Delete a custom prompt from Supabase."""
    try:
        supabase.table("custom_prompts").delete().eq("id", prompt_id).eq("user_id", user_id).execute()
    except Exception as e:
        st.error(f"Error deleting prompt: {str(e)}")

def get_relevant_context(query):
    """Get relevant context from Pinecone based on query similarity."""
    try:
        # Create embeddings for the query
        embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        query_embedding = embeddings.embed_query(query)
        
        # Get the Pinecone index
        index = pc.Index("pdf-embeddings")
        
        # Query Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True
        )
        
        # Format the context from the results
        context = []
        for match in results.matches:
            if match.metadata and 'text' in match.metadata:
                context.append(match.metadata['text'])
        
        return "\n\n".join(context) if context else "No relevant context found."
    except Exception as e:
        print(f"Error in get_relevant_context: {str(e)}")
        return "Error retrieving context from the database."

def get_llm_chain(model="gpt-4"):
    """Initialize and return a LangChain conversation chain."""
    # Initialize the callback manager for tracing
    callback_manager = CallbackManager([
        StreamingStdOutCallbackHandler(),
        LangChainTracer(),
    ])

    # Initialize the appropriate LLM based on the model selection
    if model == "gpt-4":
        llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.7,
            streaming=True,
            callback_manager=callback_manager
        )
    else:
        # Use Ollama for other models
        llm = Ollama(
            model=model,
            callback_manager=callback_manager
        )

    # Create the prompt template
    template = """You are an AI assistant with knowledge of world constitutions.
    
Context from relevant constitutional documents:
{context}

Current conversation:
{history}
Human: {input}
Assistant: Let me help you with that based on the constitutional context provided."""

    prompt = PromptTemplate(
        input_variables=["history", "input", "context"],
        template=template
    )

    # Initialize and return the chain
    return LLMChain(
        llm=llm,
        prompt=prompt,
        memory=st.session_state.memory,
        verbose=True
    )

# Initialize LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "ollama-chat-app")
langsmith_client = Client()
tracer = LangChainTracer(
    project_name=os.getenv("LANGSMITH_PROJECT", "ollama-chat-app"),
    client=langsmith_client
)

# Initialize Streamlit page configuration
st.set_page_config(
    page_title="AI Chat App",
    page_icon="💬",
    layout="wide"
)

# Custom CSS to change sidebar close button to chevron
st.markdown("""
<style>
    button[kind="header"] {
        background: none!important;
        border: none;
        padding: 0!important;
        color: #262730;
        cursor: pointer;
    }
    button[kind="header"]::before {
        content: "←";
        font-size: 1.5rem;
        font-weight: bold;
    }
    .collapsed button[kind="header"]::before {
        content: "››››";
    }
    [data-testid="stSidebarNav"] {
        display: none;
    }
    /* Hide the default close button */
    .st-emotion-cache-czk5ss {
        display: none !important;
    }
    .st-emotion-cache-h1nyin {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Add custom CSS for the sidebar
st.markdown("""
    <style>
    .prompt-container {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    .prompt-title {
        color: #0e1117;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .button-row {
        display: flex;
        gap: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        input_key="input"
    )

if "context" not in st.session_state:
    st.session_state.context = ""

# Streamlit UI
st.title("💬 Chat with an AI with knowledge of the World Constitutions")
st.subheader("The AI currently only knows of world constitutions for countries that start with the letter A")

# Sidebar
with st.sidebar:
    # Model selection
    model = st.selectbox(
        "Choose your model",
        ["gpt-4", "llama2", "mistral", "codellama"],
        index=0
    )
    
    # Save user preference when model changes
    save_user_preferences(st.session_state.user_id, model)
    
    # Custom prompts section
    st.subheader("Custom Prompts")
    
    # Create new prompt section
    with st.expander("Create New Prompt", expanded=False):
        prompt_name = st.text_input("Prompt Name")
        prompt_content = st.text_area("Prompt Content")
        if st.button("Save Prompt", type="primary"):
            if prompt_name and prompt_content:
                save_custom_prompt(st.session_state.user_id, prompt_name, prompt_content)
                st.success("Prompt saved!")
                st.rerun()  # Refresh to show new prompt
    
    # Display saved prompts
    saved_prompts = get_custom_prompts(st.session_state.user_id)
    if saved_prompts:
        st.write("Saved Prompts:")
        for prompt in saved_prompts:
            st.markdown(f"""
                <div class="prompt-container">
                    <div class="prompt-title">{prompt['name']}</div>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                with st.expander("View Content", expanded=False):
                    st.text_area("Prompt", prompt['content'], disabled=True, key=f"content_{prompt['id']}")
            with col2:
                if st.button("Use", key=f"use_{prompt['id']}", type="primary"):
                    # Add user message
                    st.session_state.messages.append({"role": "user", "content": prompt['content']})
                    
                    # Save user message to Supabase with custom prompt reference
                    save_message_to_supabase(
                        st.session_state.user_id,
                        "user",
                        prompt['content'],
                        model,
                        custom_prompt_id=prompt['id'],
                        created_from_prompt=True
                    )
                    
                    # Get AI response using LangChain
                    try:
                        start_time = datetime.now()
                        chain = get_llm_chain(model)
                        response = chain.run(
                            input=prompt['content'],
                            context=st.session_state.context
                        )
                        end_time = datetime.now()
                        duration_ms = (end_time - start_time).total_seconds() * 1000
                        
                        if response:
                            # Add assistant message
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            
                            # Save assistant message to Supabase with parent message reference
                            save_message_to_supabase(
                                st.session_state.user_id,
                                "assistant",
                                response,
                                model,
                                parent_message_id=st.session_state.last_message_id
                            )
                            
                            # Save analytics
                            prompt_tokens = len(prompt['content'].split())
                            completion_tokens = len(response.split())
                            save_conversation_analytics(
                                st.session_state.user_id,
                                model,
                                prompt_tokens,
                                completion_tokens,
                                duration_ms
                            )
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                    
                    st.rerun()
            with col3:
                if st.button("🗑️", key=f"delete_{prompt['id']}", help="Delete prompt"):
                    delete_custom_prompt(st.session_state.user_id, prompt['id'])
                    st.rerun()
    else:
        st.info("No saved prompts yet. Create one above!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to discuss?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Save user message to Supabase
    save_message_to_supabase(st.session_state.user_id, "user", prompt, model)

    # Get AI response using LangChain
    with st.chat_message("assistant"):
        with st.spinner("🧠 Searching constitutional knowledge..."):
            try:
                start_time = datetime.now()
                
                # Get relevant context from vector database
                context = get_relevant_context(prompt)
                
                # Get response from LLM with context
                chain = get_llm_chain(model)
                response = chain.run(
                    input=prompt,
                    context=context
                )
                
                end_time = datetime.now()
                duration_ms = (end_time - start_time).total_seconds() * 1000
                
                if response:
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Save assistant message to Supabase with parent message reference
                    save_message_to_supabase(
                        st.session_state.user_id,
                        "assistant",
                        response,
                        model,
                        parent_message_id=st.session_state.last_message_id
                    )
                    
                    # Save analytics
                    prompt_tokens = len(prompt.split())
                    completion_tokens = len(response.split())
                    save_conversation_analytics(
                        st.session_state.user_id,
                        model,
                        prompt_tokens,
                        completion_tokens,
                        duration_ms
                    )

            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

# Clear chat button
if st.sidebar.button("Clear Chat!"):
    # Do not reset user_id to maintain prompt ownership
    st.session_state.messages = []
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        input_key="input"
    )
    # Start a new conversation
    st.session_state.current_conversation_id = str(uuid.uuid4())
    st.session_state.last_message_id = None
    st.rerun()
