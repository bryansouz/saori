import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
import openai
import json

# Importar sistema de documentos
import documents

# Importar prompts
from prompts import SYSTEM_PROMPT, TEST_MODE_PROMPT

# Log para diagnóstico
print("Inicializando aplicação...")
print(f"Python version: {os.sys.version}")
print(f"OpenAI version: {openai.__version__}")
print(f"Streamlit version: {st.__version__}")

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()
print("Variáveis de ambiente carregadas")

# Inicializar estados da sessão para a chave API
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")
    print(f"API key encontrada no .env: {'Sim' if st.session_state.openai_api_key else 'Não'}")

# Configurar a API key do OpenAI a partir da sessão (será atualizada pela interface se necessário)
if st.session_state.openai_api_key:
    openai.api_key = st.session_state.openai_api_key
    print("API key configurada")
else:
    print("AVISO: API key não encontrada!")

# Configuração inicial do Streamlit
st.set_page_config(
    page_title="Saori Intelligence",
    page_icon="🏛️",
    layout="wide"
)

def get_completion(messages, model="gpt-3.5-turbo"):
    """
    Obter resposta do modelo OpenAI usando a nova sintaxe
    
    Args:
        messages: Lista de mensagens no formato esperado pela API
        model: Modelo OpenAI a ser usado
        
    Returns:
        Texto da resposta do modelo
    """
    try:
        # Certificar-se de que a API key está configurada
        if not st.session_state.openai_api_key:
            return "⚠️ API key não configurada! Por favor, configure a chave da API OpenAI."
            
        # Usar a chave API da sessão
        openai.api_key = st.session_state.openai_api_key
        
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.7,
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"Erro ao comunicar com a API OpenAI: {str(e)}"

def main():
    # Interface para inserir a chave API se não estiver configurada
    api_key_configured = st.session_state.openai_api_key != ""
    
    if not api_key_configured:
        st.error("⚠️ Chave da API OpenAI não configurada!")
        
        # Criar um formulário para a chave API
        with st.form("api_key_form"):
            api_key = st.text_input("Chave da API OpenAI:", 
                                    type="password", 
                                    help="Insira sua chave API da OpenAI (começa com 'sk-')")
            
            submit = st.form_submit_button("Salvar Chave API")
            
            if submit and api_key:
                # Salvar a chave API na sessão
                st.session_state.openai_api_key = api_key
                # Atualizar configuração da OpenAI
                openai.api_key = api_key
                st.success("✅ Chave API configurada com sucesso!")
                st.experimental_rerun()
            elif submit:
                st.warning("Por favor, insira uma chave API válida")
        
        # Mostrar instruções alternativas para usar arquivo .env
        st.info("Alternativamente, você pode configurar a chave API:\n"
                "1. Crie um arquivo `.env` na pasta raiz do projeto\n"
                "2. Adicione a linha: `OPENAI_API_KEY=sua-chave-aqui`\n"
                "3. Reinicie a aplicação")
        
        # Parar a execução até que a chave esteja configurada
        if not st.session_state.openai_api_key:
            st.stop()
    
    # Título
    st.title("Saori Intelligence")
    st.caption("Eagle")
    
    # Inicializar estado da sessão
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "chat"
    
    # Verificar se a pasta de documentos existe
    if not os.path.exists(documents.DOCUMENTS_DIR):
        os.makedirs(documents.DOCUMENTS_DIR)
        
    # Verificar se a pasta de chunks existe
    if not os.path.exists(documents.CHUNKS_DIR):
        os.makedirs(documents.CHUNKS_DIR)
    
    # Inicializar o arquivo de índice se não existir
    if not os.path.exists(documents.INDEX_FILE):
        with open(documents.INDEX_FILE, 'w') as f:
            f.write('{}')
    
    # Barra lateral para configurações
    with st.sidebar:
        st.header("Configurações")
        
        # Opção para reconfigurar a chave API
        if st.expander("Configuração da API"):
            current_key_masked = "•" * 20 + st.session_state.openai_api_key[-5:] if len(st.session_state.openai_api_key) > 5 else ""
            st.text(f"Chave atual: {current_key_masked}")
            
            # Formulário para alterar a chave
            with st.form("update_api_key"):
                new_key = st.text_input("Nova chave API:", type="password")
                update_key = st.form_submit_button("Atualizar")
                
                if update_key and new_key:
                    st.session_state.openai_api_key = new_key
                    openai.api_key = new_key
                    st.success("Chave atualizada!")
                    st.experimental_rerun()
        
        # Botões para alternar entre guias
        tab_col1, tab_col2, tab_col3 = st.columns(3)
        with tab_col1:
            if st.button("💬 Chat", use_container_width=True):
                st.session_state.active_tab = "chat"
        with tab_col2:
            if st.button("📚 Documentos", use_container_width=True):
                st.session_state.active_tab = "docs"
        with tab_col3:
            if st.button("🔧 Debug", use_container_width=True):
                st.session_state.active_tab = "debug"
        
        # Botão para limpar conversa (somente visível na guia chat)
        if st.session_state.active_tab == "chat":
            if st.button("Limpar Conversa", use_container_width=True):
                st.session_state.messages = []
                st.experimental_rerun()
            
        st.divider()
        st.caption("Saori - Versão 1.1 • 2025")
    
    # Guia de Documentos
    if st.session_state.active_tab == "docs":
        show_documents_interface()
    # Guia de Chat (padrão)
    elif st.session_state.active_tab == "chat":
        show_chat_interface()
    # Guia de Debug
    else:
        show_debug_interface()

def show_documents_interface():
    """Interface para gerenciamento de documentos"""
    st.header("Gerenciamento de Documentos")
    st.write("Adicione documentos para aumentar o conhecimento de Saori.")
    
    # Inicializar o processador de documentos
    doc_processor = documents.get_document_processor()
    
    # Interface para upload de novos documentos
    with st.expander("📤 Adicionar Novo Documento", expanded=True):
        uploaded_file = st.file_uploader("Selecione um arquivo", 
                                         type=["pdf", "docx", "txt", "md"],
                                         help="Formatos suportados: PDF, DOCX, TXT, Markdown")
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            with col1:
                doc_title = st.text_input("Título do documento:", 
                                          value=uploaded_file.name)
            with col2:
                doc_description = st.text_area("Descrição (opcional):", 
                                               height=100)
            
            if st.button("Processar Documento", use_container_width=True):
                with st.spinner("Processando documento..."):
                    success, message = documents.add_document_from_upload(
                        uploaded_file, doc_title, doc_description
                    )
                    
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
    
    # Listar documentos existentes
    st.subheader("Documentos Disponíveis")
    doc_list = doc_processor.get_document_list()
    
    if not doc_list:
        st.info("Nenhum documento encontrado. Adicione documentos para aumentar o conhecimento de Saori.")
    else:
        # Criar tabela de documentos
        for idx, doc in enumerate(doc_list):
            with st.container():
                col1, col2, col3 = st.columns([3, 6, 2])
                
                with col1:
                    if doc["file_type"] == "pdf":
                        st.write("📄 PDF")
                    elif doc["file_type"] == "docx":
                        st.write("📝 DOCX")
                    elif doc["file_type"] in ["txt", "md"]:
                        st.write("📃 TXT/MD")
                    else:
                        st.write("📁 Arquivo")
                
                with col2:
                    st.write(f"**{doc['title']}**")
                    st.caption(f"{doc.get('description', '')[:100]}...")
                    st.caption(f"Adicionado em: {doc.get('added_date', '').split('T')[0]}")
                
                with col3:
                    col3a, col3b = st.columns(2)
                    with col3a:
                        if st.button("🔄", key=f"reprocess_{idx}", help="Reprocessar documento"):
                            with st.spinner("Reprocessando documento..."):
                                success, message = doc_processor.reprocess_document(doc["id"])
                                if success:
                                    st.success(message)
                                    st.experimental_rerun()
                                else:
                                    st.error(message)
                    with col3b:
                        if st.button("🗑️", key=f"delete_{idx}", help="Remover documento"):
                            if doc_processor.remove_document(doc["id"]):
                                st.success("Documento removido com sucesso!")
                                st.experimental_rerun()
                            else:
                                st.error("Erro ao remover documento.")
                
                st.divider()

def show_chat_interface():
    """Interface de chat com o usuário"""
    # Exibir mensagens anteriores usando o chat_message nativo do Streamlit
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else None):
            st.write(msg["content"])
    
    # Limitar o histórico de mensagens para evitar exceder limites da API
    # Manter apenas as últimas 6 mensagens (3 pares de user-assistant)
    if len(st.session_state.messages) > 6:
        st.info(f"Histórico limitado para melhor desempenho. {len(st.session_state.messages) - 6} mensagens antigas foram resumidas.")
        st.session_state.messages = st.session_state.messages[-6:]
    
    # Entrada do usuário
    user_input = st.chat_input("Digite sua mensagem...")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Mostrar a mensagem do usuário
        with st.chat_message("user", avatar="🧑"):
            st.write(user_input)
        
        # Comandos especiais
        if "!limpar" in user_input or "!clear" in user_input:
            st.session_state.messages = []
            return
        
        # Modo de teste
        testing_mode = False
        if "!teste" in user_input:
            testing_mode = True
            user_input = user_input.replace("!teste", "").strip()
            st.session_state.messages.append({"role": "system", "content": TEST_MODE_PROMPT})
        
        # Comando para reprocessar documentos com embeddings
        if "!reprocessar" in user_input:
            st.session_state.messages.append({"role": "assistant", "content": "Iniciando reprocessamento de todos os documentos com geração de embeddings. Isso pode levar alguns minutos..."})
            
            try:
                success, message = documents.reprocess_all_documents()
                
                if success:
                    st.session_state.messages.append({"role": "assistant", "content": "✅ Documentos reprocessados com sucesso! Os embeddings foram gerados para todos os chunks."})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": f"❌ Erro ao reprocessar documentos: {message}"})
                
                st.experimental_rerun()
                return
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"❌ Erro ao reprocessar documentos: {str(e)}"})
                st.experimental_rerun()
                return
        
        # Preparar as mensagens para o modelo
        with st.spinner("Consultando banco de dados..."):
            try:
                # Buscar conhecimento relevante dos documentos
                relevant_knowledge = documents.get_relevant_knowledge(user_input)
                
                # Preparar mensagens para a API
                api_messages = []
                
                # Obtém apenas as últimas 4 mensagens do histórico para reduzir o tamanho da solicitação
                recent_messages = st.session_state.messages[-4:] if len(st.session_state.messages) > 4 else st.session_state.messages
                
                for msg in recent_messages:
                    api_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                
                # Prepara o sistema com as instruções sobre como responder
                system_content = SYSTEM_PROMPT
                
                # Adiciona o conhecimento relevante
                system_content += relevant_knowledge
                
                api_messages.insert(0, {"role": "system", "content": system_content})
                
                # Obter resposta
                response = get_completion(api_messages, model="gpt-4")
                
                # Adicionar resposta ao histórico
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Exibir resposta
                with st.chat_message("assistant"):
                    st.write(response)
            
            except Exception as e:
                st.error(f"Erro ao processar sua solicitação: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

def show_debug_interface():
    """Interface para depuração e diagnóstico"""
    st.header("Ferramentas de Diagnóstico")
    st.warning("Esta área é destinada para desenvolvedores e solução de problemas.")
    
    tabs = st.tabs(["Diagnóstico do Sistema", "Teste de PDF", "Gestão de Documentos"])
    
    # Tab de Diagnóstico do Sistema
    with tabs[0]:
        st.subheader("Informações do Sistema")
        
        # Sistema e Python
        import platform
        import sys
        import os  # Reimportar o módulo os localmente para garantir acesso
        
        system_info = {
            "Sistema Operacional": f"{platform.system()} {platform.release()}",
            "Versão do Python": platform.python_version(),
            "Diretório de Trabalho": os.getcwd()
        }
        
        for k, v in system_info.items():
            st.write(f"**{k}:** {v}")
            
        # Verificar dependências instaladas
        st.subheader("Dependências")
        
        # Lista de dependências críticas
        dependencies = [
            {"name": "OpenAI", "module": "openai", "test": "import openai; print(openai.__version__)"},
            {"name": "Streamlit", "module": "streamlit", "test": "import streamlit; print(streamlit.__version__)"},
            {"name": "PyMuPDF", "module": "fitz", "test": "import fitz; print(fitz.__version__)"},
            {"name": "python-docx", "module": "docx", "test": "import docx; print(docx.__version__)"},
            {"name": "LangChain", "module": "langchain", "test": "import langchain; print(langchain.__version__)"}
        ]
        
        for dep in dependencies:
            try:
                # Executar o teste para obter a versão
                import os
                result = os.popen(f"python -c \"{dep['test']}\"").read().strip()
                st.success(f"✅ {dep['name']} - Versão: {result}")
            except Exception:
                try:
                    # Tentar apenas importar o módulo se o teste falhar
                    __import__(dep["module"])
                    st.success(f"✅ {dep['name']} - Instalado (versão desconhecida)")
                except ImportError:
                    st.error(f"❌ {dep['name']} - Não instalado")
        
        # Verificar estrutura de diretórios
        st.subheader("Estrutura de Diretórios")
        
        from documents import DOCUMENTS_DIR, CHUNKS_DIR, INDEX_FILE
        
        directories = [
            {"path": DOCUMENTS_DIR, "name": "Documentos", "required": True},
            {"path": CHUNKS_DIR, "name": "Chunks", "required": True}
        ]
        
        files = [
            {"path": INDEX_FILE, "name": "Índice de Documentos", "required": True}
        ]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Diretórios:**")
            for dir_info in directories:
                exists = os.path.exists(dir_info["path"])
                if exists:
                    items = os.listdir(dir_info["path"])
                    st.success(f"✅ {dir_info['name']} - {len(items)} itens")
                elif dir_info["required"]:
                    st.error(f"❌ {dir_info['name']} - Não encontrado")
                else:
                    st.warning(f"⚠️ {dir_info['name']} - Não encontrado")
        
        with col2:
            st.write("**Arquivos:**")
            for file_info in files:
                exists = os.path.exists(file_info["path"])
                if exists:
                    try:
                        size = os.path.getsize(file_info["path"])
                        st.success(f"✅ {file_info['name']} - {size} bytes")
                    except:
                        st.success(f"✅ {file_info['name']} - Existe")
                elif file_info["required"]:
                    st.error(f"❌ {file_info['name']} - Não encontrado")
                else:
                    st.warning(f"⚠️ {file_info['name']} - Não encontrado")
        
        # Diagnóstico do sistema
        st.subheader("Ações de Diagnóstico")
        
        run_diag = st.button("🔍 Executar Diagnóstico Completo", use_container_width=True)
        
        if run_diag:
            with st.spinner("Executando diagnóstico..."):
                st.write("**Resultados do diagnóstico:**")
                
                # 1. Verificar diretórios
                diag_results = []
                
                for dir_info in directories:
                    if not os.path.exists(dir_info["path"]):
                        diag_results.append({
                            "status": "error",
                            "message": f"Diretório {dir_info['name']} não existe",
                            "action": f"Criar diretório: `{dir_info['path']}`"
                        })
                        if dir_info["required"]:
                            try:
                                os.makedirs(dir_info["path"], exist_ok=True)
                                diag_results.append({
                                    "status": "success", 
                                    "message": f"Diretório {dir_info['name']} criado automaticamente"
                                })
                            except:
                                diag_results.append({
                                    "status": "error",
                                    "message": f"Falha ao criar diretório {dir_info['name']}"
                                })
                
                # 2. Verificar arquivo de índice
                index_exists = os.path.exists(INDEX_FILE)
                if not index_exists:
                    diag_results.append({
                        "status": "error",
                        "message": "Arquivo de índice não existe",
                        "action": "Criar um novo arquivo de índice"
                    })
                    try:
                        with open(INDEX_FILE, 'w', encoding='utf-8') as f:
                            json.dump({}, f)
                        diag_results.append({
                            "status": "success",
                            "message": "Novo arquivo de índice criado"
                        })
                    except:
                        diag_results.append({
                            "status": "error",
                            "message": "Falha ao criar arquivo de índice"
                        })
                else:
                    # Verificar se o arquivo de índice é um JSON válido
                    try:
                        with open(INDEX_FILE, 'r', encoding='utf-8') as f:
                            index = json.load(f)
                        diag_results.append({
                            "status": "success",
                            "message": f"Arquivo de índice é válido: {len(index)} documentos"
                        })
                    except:
                        diag_results.append({
                            "status": "error",
                            "message": "Arquivo de índice é inválido",
                            "action": "Reconstruir índice"
                        })
                
                # 3. Verificar consistência entre índice e arquivos
                try:
                    doc_processor = documents.get_document_processor()
                    doc_list = doc_processor.get_document_list()
                    
                    for doc in doc_list:
                        # Verificar se o arquivo do documento existe
                        doc_file = os.path.join(DOCUMENTS_DIR, doc.get("filename", ""))
                        if not os.path.exists(doc_file):
                            diag_results.append({
                                "status": "warning",
                                "message": f"Arquivo do documento '{doc.get('title')}' não encontrado",
                                "action": "Remover documento do índice"
                            })
                        
                        # Verificar se o arquivo de chunks existe
                        chunks_file = os.path.join(CHUNKS_DIR, f"{doc.get('id')}.json")
                        if not os.path.exists(chunks_file):
                            diag_results.append({
                                "status": "warning",
                                "message": f"Arquivo de chunks para '{doc.get('title')}' não encontrado",
                                "action": "Reprocessar documento ou removê-lo"
                            })
                except Exception as e:
                    diag_results.append({
                        "status": "error",
                        "message": f"Erro ao verificar documentos: {str(e)}"
                    })
                
                # Mostrar resultados
                for result in diag_results:
                    if result["status"] == "error":
                        st.error(result["message"])
                    elif result["status"] == "warning":
                        st.warning(result["message"])
                    else:
                        st.success(result["message"])
                    
                    if "action" in result:
                        st.info(f"Ação recomendada: {result['action']}")
                        
                # Mostrar resultados finais
                if not diag_results:
                    st.success("✅ Sistema em bom estado. Nenhum problema encontrado.")
                elif all(r["status"] == "success" for r in diag_results):
                    st.success("✅ Todos os problemas foram corrigidos automaticamente.")
                else:
                    st.warning("⚠️ Foram encontrados problemas que requerem atenção.")
    
    # Tab de Teste de PDF
    with tabs[1]:
        st.subheader("Teste de Processamento de PDF")
        
        # Verificar se a biblioteca PyMuPDF está instalada
        try:
            import fitz
            pymupdf_ok = True
            st.success(f"✅ PyMuPDF está instalado (versão {fitz.__version__})")
        except ImportError:
            pymupdf_ok = False
            st.error("❌ PyMuPDF não está instalado. Instale com: `pip install pymupdf`")
        
        if pymupdf_ok:
            pdf_file = st.file_uploader("Selecione um PDF para teste", type=["pdf"])
            
            if pdf_file:
                st.write(f"**Arquivo selecionado:** {pdf_file.name}")
                st.write(f"**Tamanho:** {pdf_file.size} bytes")
                
                if st.button("Testar Extração de Texto", use_container_width=True):
                    with st.spinner("Processando PDF..."):
                        try:
                            # Salvar o arquivo temporariamente
                            temp_path = os.path.join("documents", "debug_" + pdf_file.name)
                            with open(temp_path, "wb") as f:
                                f.write(pdf_file.getbuffer())
                            
                            # Tentar abrir o PDF
                            st.info(f"Tentando abrir o PDF: {temp_path}")
                            
                            doc = fitz.open(temp_path)
                            st.success(f"PDF aberto com sucesso! Páginas: {len(doc)}")
                            
                            # Extrair texto
                            total_text = ""
                            
                            for i, page in enumerate(doc):
                                page_text = page.get_text()
                                total_text += page_text
                                st.write(f"Página {i+1}: {len(page_text)} caracteres")
                            
                            # Mostrar uma amostra do texto extraído
                            st.subheader("Amostra do texto extraído:")
                            st.info(total_text[:500] + "..." if len(total_text) > 500 else total_text)
                            
                            # Limpar
                            doc.close()
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                            
                        except Exception as e:
                            st.error(f"Erro ao processar o PDF: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
    
    # Tab de Gestão de Documentos
    with tabs[2]:
        st.subheader("Gestão de Documentos")
        
        # Listar documentos no diretório
        if os.path.exists(documents.DOCUMENTS_DIR):
            docs = os.listdir(documents.DOCUMENTS_DIR)
            
            st.write(f"**Arquivos no diretório de documentos:** {len(docs)}")
            
            if not docs:
                st.info("Nenhum arquivo encontrado")
            else:
                for doc in docs:
                    doc_path = os.path.join(documents.DOCUMENTS_DIR, doc)
                    doc_size = os.path.getsize(doc_path)
                    st.write(f"📄 **{doc}** - {doc_size} bytes")
        
        # Listar chunks
        if os.path.exists(documents.CHUNKS_DIR):
            chunks = os.listdir(documents.CHUNKS_DIR)
            
            st.write(f"**Arquivos no diretório de chunks:** {len(chunks)}")
            
            if not chunks:
                st.info("Nenhum arquivo de chunks encontrado")
            else:
                for chunk in chunks:
                    chunk_path = os.path.join(documents.CHUNKS_DIR, chunk)
                    chunk_size = os.path.getsize(chunk_path)
                    
                    # Verificar se é um arquivo JSON válido
                    try:
                        with open(chunk_path, 'r', encoding='utf-8') as f:
                            chunk_data = json.load(f)
                            st.write(f"📄 **{chunk}** - {chunk_size} bytes ({len(chunk_data)} chunks)")
                    except:
                        st.write(f"📄 **{chunk}** - {chunk_size} bytes (arquivo inválido)")
        
        # Ações de manutenção
        st.subheader("Ações de Manutenção")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Reconstruir Índice", use_container_width=True):
                with st.spinner("Reconstruindo índice..."):
                    try:
                        if documents.rebuild_document_index():
                            st.success("Índice reconstruído com sucesso!")
                            st.experimental_rerun()
                        else:
                            st.error("Falha ao reconstruir índice")
                    except Exception as e:
                        st.error(f"Erro: {str(e)}")
        
        with col2:
            if st.button("🧹 Limpar Tudo", use_container_width=True):
                if st.checkbox("Confirmar limpeza de todos os dados?"):
                    with st.spinner("Limpando dados..."):
                        try:
                            # Limpar índice
                            with open(documents.INDEX_FILE, 'w', encoding='utf-8') as f:
                                json.dump({}, f)
                                
                            # Limpar diretórios
                            import shutil
                            for file in os.listdir(documents.DOCUMENTS_DIR):
                                file_path = os.path.join(documents.DOCUMENTS_DIR, file)
                                try:
                                    if os.path.isfile(file_path):
                                        os.remove(file_path)
                                except:
                                    pass
                                    
                            for file in os.listdir(documents.CHUNKS_DIR):
                                file_path = os.path.join(documents.CHUNKS_DIR, file)
                                try:
                                    if os.path.isfile(file_path):
                                        os.remove(file_path)
                                except:
                                    pass
                                    
                            st.success("Sistema limpo com sucesso!")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Erro ao limpar sistema: {str(e)}")

if __name__ == "__main__":
    try:
        print("Iniciando aplicação Saori...")
        main()
        print("Aplicação iniciada com sucesso!")
    except Exception as e:
        print(f"ERRO AO INICIAR: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Mostrar erro no Streamlit também
        st.error(f"Erro ao iniciar a aplicação: {str(e)}")
        st.code(traceback.format_exc())
