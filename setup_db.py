from supabase import create_client
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Supabase client with service role key
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase = create_client(supabase_url, supabase_key)

# SQL commands to create tables and disable RLS
sql_commands = """
-- Enable UUID extension if not already enabled
create extension if not exists "uuid-ossp";

-- Chat Messages Table
create table if not exists chat_messages (
    id uuid default uuid_generate_v4() primary key,
    user_id text not null,
    role text not null,
    content text not null,
    model text not null,
    timestamp timestamptz not null
);

-- Create index on user_id for faster queries
create index if not exists chat_messages_user_id_idx on chat_messages(user_id);

-- Conversation Analytics Table
create table if not exists conversation_analytics (
    id uuid default uuid_generate_v4() primary key,
    user_id text not null,
    model text not null,
    prompt_tokens integer not null,
    completion_tokens integer not null,
    duration_ms float not null,
    timestamp timestamptz not null
);

-- Create index on user_id for faster queries
create index if not exists conversation_analytics_user_id_idx on conversation_analytics(user_id);

-- User Preferences Table
create table if not exists user_preferences (
    user_id text primary key,
    preferred_model text not null,
    last_updated timestamptz not null
);

-- Custom Prompts Table
create table if not exists custom_prompts (
    id uuid default uuid_generate_v4() primary key,
    user_id text not null,
    name text not null,
    content text not null,
    created_at timestamptz not null,
    -- Add unique constraint to prevent duplicate prompt names per user
    unique(user_id, name)
);

-- Create index on user_id for faster queries
create index if not exists custom_prompts_user_id_idx on custom_prompts(user_id);

-- Disable RLS for all tables
alter table chat_messages disable row level security;
alter table conversation_analytics disable row level security;
alter table user_preferences disable row level security;
alter table custom_prompts disable row level security;
"""

try:
    # Execute SQL commands
    result = supabase.rpc('exec_sql', {'sql_string': sql_commands}).execute()
    print("Database setup completed successfully!")
except Exception as e:
    print(f"Error setting up database: {str(e)}")
