# Supabase Persistent Learning Setup

This app can keep its prediction history and adaptive model weights in Supabase.
If Supabase is not configured, it falls back to local JSON files.

## 1. Create the table

In Supabase, open SQL Editor and run:

```sql
create table if not exists public.quant_state (
  key text primary key,
  payload jsonb not null,
  updated_at timestamptz default now()
);
```

## 2. Add Render environment variables

In Render, open your service, then go to Environment and add:

```text
SUPABASE_URL=https://YOUR_PROJECT_ID.supabase.co
SUPABASE_SERVICE_ROLE_KEY=YOUR_SERVICE_ROLE_KEY
SUPABASE_STATE_TABLE=quant_state
```

Use `SUPABASE_SERVICE_ROLE_KEY` only in Render/server settings.
Do not put this key in frontend JavaScript, GitHub, or Netlify.

## 3. Redeploy Render

After saving the environment variables, click:

```text
Manual Deploy -> Deploy latest commit
```

## What gets saved

- `prediction_log`: every ticker analysis and its 1-week prediction
- `adaptive_model_state`: learned model weights after old predictions are evaluated

After 7 days, the app compares prediction vs actual price, updates the model
weights gently, and stores the updated result in Supabase.

Educational use only. This is not financial advice.
