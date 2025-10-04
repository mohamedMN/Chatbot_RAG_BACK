"""
Configuration et setup pour Ollama avec deepseek-coder:instruct
"""
import os
from typing import Optional, Dict, Any

try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.llms import Ollama
        from langchain_community.chat_models import ChatOllama
        OLLAMA_AVAILABLE = True
    except ImportError:
        OLLAMA_AVAILABLE = False


class OllamaConfig:
    """Configuration pour Ollama LLM"""

    # Modèle par défaut
    DEFAULT_MODEL = "deepseek-coder:instruct"

    # Configuration par défaut
    DEFAULT_CONFIG = {
        "model": DEFAULT_MODEL,
        "temperature": 0.1,  # Plus déterministe pour les réponses techniques
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "num_ctx": 4096,  # Contexte plus large pour RAG
        "stop": ["Human:", "Assistant:", "User:"],
    }

    def __init__(self, model: Optional[str] = None, **kwargs):
        self.model = model or os.getenv("OLLAMA_MODEL", self.DEFAULT_MODEL)
        self.config = self.DEFAULT_CONFIG.copy()
        self.config.update(kwargs)

        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "Ollama non disponible. Installez avec: pip install langchain-ollama"
            )

    def create_llm(self, **override_params) -> 'ChatOllama':
        """
        Crée une instance ChatOllama configurée
        
        Args:
            **override_params: Paramètres à surcharger
            
        Returns:
            Instance ChatOllama configurée
        """
        config = self.config.copy()
        config.update(override_params)

        try:
            llm = ChatOllama(**config)

            # Test de connectivité
            self._test_connection(llm)

            return llm

        except Exception as e:
            raise Exception(f"Erreur lors de la création du LLM Ollama: {e}")

    def _test_connection(self, llm) -> None:
        """
        Test rapide de connectivité avec Ollama
        
        Args:
            llm: Instance LLM à tester
        """
        try:
            # Test simple
            response = llm.invoke("Test de connexion")
            print(f"Ollama connecté avec succès - Modèle: {self.model}")

        except Exception as e:
            print(f"Attention: Test de connexion Ollama échoué: {e}")
            print("Assurez-vous que:")
            print("1. Ollama est installé et en cours d'exécution")
            print("2. Le modèle est disponible: ollama pull deepseek-coder:instruct")
            print("3. Le service Ollama est accessible")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """
        Informations sur le modèle configuré
        
        Returns:
            Dictionnaire des informations
        """
        return {
            "model": self.model,
            "config": self.config,
            "available": OLLAMA_AVAILABLE,
            "recommended_for": [
                "Code analysis",
                "Technical documentation",
                "Integration questions",
                "webMethods expertise"
            ]
        }

    @staticmethod
    def setup_instructions() -> str:
        """
        Instructions de setup pour Ollama
        
        Returns:
            Instructions formatées
        """
        return """
Setup Ollama pour RAG:

1. Installation d'Ollama:
   curl -fsSL https://ollama.com/install.sh | sh

2. Démarrage du service:
   ollama serve

3. Téléchargement du modèle:
   ollama pull deepseek-coder:instruct

4. Test du modèle:
   ollama run deepseek-coder:instruct "Hello"

5. Variables d'environnement (optionnel):
   export OLLAMA_MODEL=deepseek-coder:instruct
   export OLLAMA_HOST=localhost:11434

6. Installation Python:
   pip install langchain-ollama
        """


def create_ollama_llm(model: Optional[str] = None, **kwargs) -> 'ChatOllama':
    """
    Fonction utilitaire pour créer un LLM Ollama configuré
    
    Args:
        model: Nom du modèle (optionnel)
        **kwargs: Paramètres supplémentaires
        
    Returns:
        Instance ChatOllama
    """
    config = OllamaConfig(model, **kwargs)
    return config.create_llm()


def get_deepseek_coder_config() -> Dict[str, Any]:
    """
    Configuration optimisée pour deepseek-coder:instruct
    
    Returns:
        Configuration recommandée
    """
    return {
        "model": "deepseek-coder:instruct",
        "temperature": 0.1,  # Très déterministe pour code/tech
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "num_ctx": 6144,  # Contexte élargi pour RAG
        "stop": ["```", "Human:", "User:", "\n\nHuman:", "\n\nUser:"],
        "system": "Tu es un expert en intégration et Software AG webMethods"
    }


def check_ollama_status() -> Dict[str, Any]:
    """
    Vérifie le statut d'Ollama
    
    Returns:
        Statut d'Ollama et des modèles
    """
    import requests

    status = {
        "ollama_running": False,
        "models_available": [],
        "deepseek_available": False,
        "error": None
    }

    try:
        # Test de base d'Ollama
        response = requests.get("http://localhost:11434/api/tags", timeout=5)

        if response.status_code == 200:
            status["ollama_running"] = True

            # Liste des modèles
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            status["models_available"] = models

            # Vérifie deepseek-coder
            status["deepseek_available"] = any(
                "deepseek-coder" in model for model in models
            )

    except requests.exceptions.RequestException as e:
        status["error"] = f"Ollama non accessible: {e}"
    except Exception as e:
        status["error"] = f"Erreur lors de la vérification: {e}"

    return status
