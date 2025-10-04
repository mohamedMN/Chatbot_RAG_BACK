"""
Configuration Supabase pour l'API RAG
"""
import os
from typing import Optional
import dotenv


dotenv()
class SupabaseConfig:
    """Configuration Supabase"""

    def __init__(self):
        # URL et clé Supabase depuis les variables d'environnement
        self.url: str = os.getenv("supabaseUrl", "")
        self.service_key: str = os.getenv("supabaseKey", "")

        if not self.url:
            raise ValueError(
                "Variables d'environnement SUPABASE_URL requises"
            )

    def get_client_config(self) -> dict:
        """Configuration pour le client Supabase"""
        return {
            "url": self.url,
            "key": self.key
        }

    def get_admin_config(self) -> dict:
        """Configuration admin avec service key"""
        return {
            "url": self.url,
            "key": self.service_key or self.key
        }


# Instance globale
supabase_config = SupabaseConfig()

# SQL pour créer les tables Supabase
SUPABASE_SETUP_SQL = """
-- Table des utilisateurs
CREATE TABLE IF NOT EXISTS users (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table des logs de questions
CREATE TABLE IF NOT EXISTS question_logs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    user_email TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    context_count INTEGER DEFAULT 0,
    response_time REAL DEFAULT 0.0,
    success BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table de configuration système
CREATE TABLE IF NOT EXISTS system_config (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    config_type TEXT UNIQUE NOT NULL,
    config_data JSONB NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table des sessions
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    user_email TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL
);

-- Indexes pour performance
CREATE INDEX IF NOT EXISTS idx_question_logs_user_email ON question_logs(user_email);
CREATE INDEX IF NOT EXISTS idx_question_logs_created_at ON question_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);

-- RLS (Row Level Security) - optionnel
-- ALTER TABLE users ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE question_logs ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE system_config ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;

-- Insérer un utilisateur admin par défaut
INSERT INTO users (email, password_hash, is_admin)
VALUES ('admin@example.com', 'sha256_hash_of_admin123', TRUE)
ON CONFLICT (email) DO NOTHING;

-- Configuration par défaut du modèle
INSERT INTO system_config (config_type, config_data)
VALUES 
(
    'model', 
    '{
        "model": "deepseek-coder:instruct",
        "temperature": 0.1,
        "top_p": 0.9,
        "top_k": 40,
        "num_ctx": 6144,
        "repeat_penalty": 1.1
    }'::jsonb
),
(
    'retrieval',
    '{
        "top_k": 20,
        "min_score": 0.3,
        "max_response_chars": 1800
    }'::jsonb
)
ON CONFLICT (config_type) DO NOTHING;
"""
