"""
Script de teste para verificar a disponibilidade do LangChain

Este script verifica se o LangChain está instalado e funcionando corretamente,
tentando inicializar as classes principais que serão usadas pela aplicação.
"""
import os
import sys

def check_langchain():
    """Verificar se o LangChain está disponível e suas dependências estão funcionando."""
    print("Verificando a disponibilidade do LangChain...")
    
    try:
        import langchain
        print(f"[OK] LangChain instalado - Versão: {langchain.__version__}")
        
        # Verificar componentes principais
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            print("[OK] RecursiveCharacterTextSplitter disponível")
        except ImportError as e:
            print(f"[ERRO] RecursiveCharacterTextSplitter não disponível: {e}")
        
        try:
            from langchain_openai import OpenAIEmbeddings
            print("[OK] OpenAIEmbeddings disponível")
        except ImportError as e:
            print(f"[ERRO] OpenAIEmbeddings não disponível: {e}")
        
        try:
            from langchain_community.vectorstores import FAISS
            print("[OK] FAISS disponível")
        except ImportError as e:
            print(f"[ERRO] FAISS não disponível: {e}")
        
        try:
            from langchain_core.documents import Document
            print("[OK] Document disponível")
        except ImportError as e:
            print(f"[ERRO] Document não disponível: {e}")
        
        try:
            from langchain_community.document_loaders import TextLoader
            print("[OK] TextLoader disponível")
        except ImportError as e:
            print(f"[ERRO] TextLoader não disponível: {e}")
        
        print("\nComponentes essenciais verificados.")
        
        # Verificar API key
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            print("[OK] OPENAI_API_KEY encontrada")
            
            # Teste simples da API (sem fazer chamadas reais)
            try:
                embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                print("[OK] OpenAIEmbeddings inicializado com sucesso")
            except Exception as e:
                print(f"[ERRO] Erro ao inicializar OpenAIEmbeddings: {e}")
        else:
            print("[ERRO] OPENAI_API_KEY não encontrada")
        
        return True
    
    except ImportError as e:
        print(f"[ERRO] LangChain não está instalado: {e}")
        return False

if __name__ == "__main__":
    check_langchain()
