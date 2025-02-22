-- Enable UUID extension if not already enabled
create extension if not exists "uuid-ossp";

-- Chat Messages Table
drop table if exists chat_messages;
create table chat_messages (
    id uuid default uuid_generate_v4() primary key,
    conversation_id uuid default uuid_generate_v4(),  -- Group messages in the same conversation
    user_id text not null,
    role text not null,
    content text not null,
    model text not null,
    timestamp timestamptz not null,
    parent_message_id uuid references chat_messages(id),  -- Reference to the message this is responding to
    custom_prompt_id uuid references custom_prompts(id),  -- Reference to the custom prompt if used
    created_from_prompt boolean default false  -- Flag to indicate if message was created from a custom prompt
);

-- Create indexes for better query performance
create index if not exists chat_messages_user_id_idx on chat_messages(user_id);
create index if not exists chat_messages_conversation_id_idx on chat_messages(conversation_id);
create index if not exists chat_messages_parent_message_id_idx on chat_messages(parent_message_id);

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

-- Temporarily disable RLS for development
alter table chat_messages disable row level security;
alter table conversation_analytics disable row level security;
alter table user_preferences disable row level security;
alter table custom_prompts disable row level security;
