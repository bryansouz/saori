# Saori Intelligence

Saori é um assistente de IA personalizado para ajudar alunos a obter informações sobre materiais de treinamento específicos.

## Recursos

- Interface amigável baseada em chat
- Consulta documentos personalizados para oferecer respostas específicas
- Projetado para servir como uma extensão do conhecimento do professor/treinador
- Responde com base apenas nos documentos fornecidos
- Integração com LangChain para processamento eficiente de grandes volumes de dados

## Como usar

1. Carregue seus documentos (PDF, DOCX, etc.)
2. Faça perguntas sobre o conteúdo dos documentos
3. Obtenha respostas baseadas apenas nos documentos carregados

## Comandos especiais

- `!teste` - Ativa o modo de teste, mostrando exatamente quais partes dos documentos estão sendo usadas
- `!reprocessar` - Reprocessa todos os documentos, gerando novos embeddings

## Requisitos

- Python 3.7+
- OpenAI API Key
- Streamlit
- LangChain (opcional, usado automaticamente se disponível)

## Instalação

```bash
pip install -r requirements.txt
```

## Configuração

Crie um arquivo `.env` com sua chave API da OpenAI:

```
OPENAI_API_KEY=sua-chave-aqui
```

## Iniciar a aplicação

```bash
streamlit run app.py
```

## LangChain Integration

A aplicação Saori 1.1 agora conta com integração opcional com o LangChain, que melhora o processamento de documentos de várias formas:

1. **Divisão de Texto Inteligente**: Utiliza o RecursiveCharacterTextSplitter do LangChain para dividir documentos em chunks de forma mais eficiente, preservando o contexto semântico.

2. **Embeddings Otimizados**: Usa a classe OpenAIEmbeddings do LangChain para gerar embeddings mais precisos.

3. **Armazenamento Vetorial com FAISS**: Implementa o FAISS para busca vetorial rápida e eficiente, mesmo com grandes volumes de dados.

4. **Carregamento de Documentos Flexível**: Suporta vários formatos de documentos usando os loaders integrados do LangChain.

A aplicação detecta automaticamente se o LangChain está disponível e alterna entre a implementação nativa e a do LangChain conforme necessário. Isso garante maior flexibilidade e robustez no processamento de documentos.

## Licença

Todos os direitos reservados
