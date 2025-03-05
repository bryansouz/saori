"""
Script de demonstração da integração LangChain

Este script demonstra como usar o LangChain para processamento de documentos
e busca semântica, similarmente ao que a aplicação Saori faz.
"""
import os
import json
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Verificar disponibilidade do LangChain
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
    print("[OK] LangChain está disponível")
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("[ERRO] LangChain não está disponível")
    exit(1)

def texto_exemplo():
    """Retorna um texto de exemplo para demonstrar a divisão de texto."""
    return """
# Desenvolvimento Web Moderno

## HTML e CSS
HTML (HyperText Markup Language) é a espinha dorsal de qualquer página web, fornecendo a estrutura básica que será estilizada com CSS (Cascading Style Sheets). O HTML5 trouxe novas tags semânticas como `<header>`, `<footer>`, `<article>` e `<section>`, melhorando a acessibilidade e SEO.

CSS3 introduziu recursos avançados como animações, transições, sombras e layouts flexíveis com Flexbox e Grid. Estes recursos permitem criar interfaces responsivas e visualmente atraentes sem depender de JavaScript.

## JavaScript e Frameworks
JavaScript é a linguagem de programação do navegador, permitindo interatividade em páginas web. O ECMAScript 6 (ES6) trouxe melhorias significativas como arrow functions, classes, e módulos.

Frameworks populares incluem:
- React: Desenvolvido pelo Facebook, usa um DOM virtual para atualizações eficientes
- Vue.js: Framework progressivo e fácil de aprender
- Angular: Framework completo mantido pelo Google

## Backend e APIs
No desenvolvimento backend, Node.js permite usar JavaScript no servidor, enquanto frameworks como Express simplificam a criação de APIs RESTful. Python com Django ou Flask, e PHP com Laravel são alternativas populares.

GraphQL está ganhando popularidade como alternativa ao REST, oferecendo mais flexibilidade nas consultas de dados.

## DevOps e Implantação
Ferramentas como Docker e Kubernetes facilitam a conteinerização e orquestração de aplicações. Plataformas de nuvem como AWS, Google Cloud e Azure oferecem serviços para implantação escalável.

Práticas de CI/CD (Integração Contínua/Entrega Contínua) automatizam testes e deployment, aumentando a velocidade e confiabilidade do desenvolvimento.
"""

def demo_text_splitting():
    """Demonstra como dividir um texto em chunks usando LangChain."""
    print("\n=== Demonstração de Divisão de Texto ===")
    texto = texto_exemplo()
    
    # Criar o text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n## ", "\n### ", "\n#### ", "\n", " ", ""]
    )
    
    # Dividir o texto
    chunks = text_splitter.split_text(texto)
    
    print(f"Texto original: {len(texto)} caracteres")
    print(f"Dividido em {len(chunks)} chunks")
    print("\nPrimeiros 3 chunks:")
    
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1} ({len(chunk)} caracteres):")
        print("-" * 50)
        print(chunk)
        print("-" * 50)
    
    return chunks

def demo_embeddings_and_search(chunks):
    """Demonstra como criar embeddings e realizar busca semântica."""
    print("\n=== Demonstração de Embeddings e Busca Semântica ===")
    
    # Verificar API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERRO] OPENAI_API_KEY não encontrada. Definindo uma chave fictícia para exemplo.")
        print("Para executar esse exemplo corretamente, defina a chave da API no arquivo .env")
        print("Este exemplo seguirá apenas com demonstração do código, sem fazer chamadas reais à API")
        return
    
    # Criar documentos
    documents = [Document(page_content=chunk, metadata={"source": "demo", "chunk_id": i}) 
                 for i, chunk in enumerate(chunks)]
    
    print(f"Criados {len(documents)} objetos Document")
    
    try:
        # Método alternativo de inicialização
        from openai import OpenAI
        print(f"Versão do OpenAI instalada e compatível com API v1.x")
        
        # Versão mais recente da API (1.x)
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        print("Modelo de embeddings inicializado")
        
        # Criar vectorstore
        db = FAISS.from_documents(documents, embeddings)
        print("Vectorstore FAISS criado com sucesso")
        
        # Realizar busca semântica
        query = "Quais são os frameworks JavaScript mais populares?"
        print(f"\nConsulta de exemplo: '{query}'")
        
        similar_docs = db.similarity_search_with_score(query, k=2)
        
        print("\nResultados da busca:")
        for i, (doc, score) in enumerate(similar_docs):
            print(f"\nResultado {i+1} (similaridade: {1-score:.4f}):")
            print("-" * 50)
            print(doc.page_content)
            print("-" * 50)
            print(f"Metadados: {doc.metadata}")
        
        return similar_docs
        
    except Exception as e:
        print(f"[ERRO] Falha ao inicializar embeddings: {e}")
        print("Esta demonstração mostra apenas a sintaxe, mas não pode executar as consultas sem as dependências corretas.")
        print("Para solucionar este problema, atualize as versões das bibliotecas:")
        print("pip install openai==1.0.0 langchain==0.0.312 langchain-community==0.0.16 langchain-openai==0.0.2")
        return None

if __name__ == "__main__":
    print("Demonstração da integração LangChain na aplicação Saori")
    if LANGCHAIN_AVAILABLE:
        chunks = demo_text_splitting()
        demo_embeddings_and_search(chunks)
    else:
        print("A demonstração não pode continuar sem o LangChain")
