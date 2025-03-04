# Saori Intelligence

Saori é um assistente de IA personalizado para ajudar alunos a obter informações sobre materiais de treinamento específicos.

## Recursos

- Interface amigável baseada em chat
- Consulta documentos personalizados para oferecer respostas específicas
- Projetado para servir como uma extensão do conhecimento do professor/treinador
- Responde com base apenas nos documentos fornecidos

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

## Licença

Todos os direitos reservados
