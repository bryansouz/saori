"""
Script de checklist pré-deployment para a aplicação Saori

Este script verifica se tudo está pronto para o deployment:
1. Verifica dependências
2. Verifica a estrutura de arquivos
3. Verifica o arquivo .env
4. Sugere melhorias caso necessário
"""
import os
import sys
import importlib
import pkg_resources

def check_dependencies():
    """Verifica se todas as dependências necessárias estão instaladas."""
    print("\n=== Verificando Dependências ===")
    
    required_packages = {
        "openai": "0.28.1",
        "streamlit": "1.32.0",
        "python-dotenv": "1.0.0",
        "numpy": "1.26.0",
        "pymupdf": "1.23.22",
        "python-docx": "1.0.1",
        "langchain": "0.0.331",
        "langchain-community": "0.0.16",
        "langchain-openai": "0.0.2",
        "faiss-cpu": "1.7.4",
        "tiktoken": "0.5.1"
    }
    
    all_dependencies_met = True
    missing_packages = []
    version_mismatch = []
    
    for package, required_version in required_packages.items():
        try:
            # Verificar se o pacote está instalado
            imported = importlib.import_module(package.replace("-", "_"))
            
            # Obter a versão instalada
            try:
                installed_version = imported.__version__
            except AttributeError:
                try:
                    installed_version = pkg_resources.get_distribution(package).version
                except:
                    installed_version = "Desconhecida"
            
            # Verificar versão mínima
            if ">" in required_version:
                min_version = required_version.replace(">=", "")
                is_correct_version = installed_version >= min_version
                version_str = f">= {min_version}"
            else:
                is_correct_version = installed_version == required_version
                version_str = required_version
                
            if is_correct_version:
                print(f"[OK] {package} - Instalado: {installed_version}")
            else:
                print(f"[ALERTA] {package} - Instalado: {installed_version}, Requerido: {version_str}")
                version_mismatch.append(package)
                all_dependencies_met = False
                
        except ImportError:
            print(f"[ERRO] {package} - Não instalado")
            missing_packages.append(package)
            all_dependencies_met = False
    
    if not all_dependencies_met:
        print("\nAlgumas dependências precisam ser instaladas ou atualizadas:")
        
        if missing_packages:
            print("\nPacotes faltando:")
            for package in missing_packages:
                print(f"  - {package}")
        
        if version_mismatch:
            print("\nPacotes com versão desatualizada:")
            for package in version_mismatch:
                print(f"  - {package}")
        
        print("\nComando para instalar todas as dependências:")
        print("pip install -r requirements.txt")
    
    return all_dependencies_met

def check_file_structure():
    """Verifica se todos os arquivos necessários existem."""
    print("\n=== Verificando Estrutura de Arquivos ===")
    
    required_files = [
        "app.py",
        "documents.py",
        "requirements.txt",
        ".env"
    ]
    
    required_dirs = [
        "docs",
        "index"
    ]
    
    all_files_exist = True
    
    # Verificar arquivos
    for file in required_files:
        if os.path.isfile(file):
            print(f"[OK] Arquivo {file} encontrado")
        else:
            print(f"[ERRO] Arquivo {file} não encontrado")
            all_files_exist = False
    
    # Verificar diretórios
    for directory in required_dirs:
        if os.path.isdir(directory):
            print(f"[OK] Diretório {directory} encontrado")
        else:
            print(f"[ALERTA] Diretório {directory} não encontrado (será criado automaticamente)")
    
    return all_files_exist

def check_env_file():
    """Verifica se o arquivo .env contém as variáveis necessárias."""
    print("\n=== Verificando Arquivo .env ===")
    
    env_file = ".env"
    required_vars = [
        "OPENAI_API_KEY"
    ]
    
    if not os.path.isfile(env_file):
        print(f"[ERRO] Arquivo {env_file} não encontrado")
        print("É necessário criar um arquivo .env com as seguintes variáveis:")
        for var in required_vars:
            print(f"  - {var}")
        return False
    
    # Ler arquivo .env
    env_vars = {}
    with open(env_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                env_vars[key.strip()] = value.strip()
    
    all_vars_defined = True
    
    # Verificar variáveis
    for var in required_vars:
        if var in env_vars and env_vars[var]:
            # Não mostrar a chave completa para segurança
            val = env_vars[var]
            masked_val = val[:4] + "*" * (len(val) - 8) + val[-4:] if len(val) > 8 else "********"
            print(f"[OK] Variável {var} definida como {masked_val}")
        else:
            print(f"[ERRO] Variável {var} não definida")
            all_vars_defined = False
    
    return all_vars_defined

def deployment_readiness():
    """Avalia a prontidão para deployment e sugere melhorias."""
    print("\n=== Avaliação de Prontidão para Deployment ===")
    
    dependencies_ok = check_dependencies()
    files_ok = check_file_structure()
    env_ok = check_env_file()
    
    if dependencies_ok and files_ok and env_ok:
        print("\n[SUCESSO] O sistema está pronto para deployment!")
        print("Para iniciar a aplicação, execute:")
        print("streamlit run app.py")
        return True
    else:
        print("\n[ATENÇÃO] Há problemas que precisam ser resolvidos antes do deployment:")
        
        if not dependencies_ok:
            print("- Instale todas as dependências usando 'pip install -r requirements.txt'")
        
        if not files_ok:
            print("- Certifique-se de que todos os arquivos necessários estão presentes")
        
        if not env_ok:
            print("- Configure corretamente o arquivo .env com as variáveis necessárias")
        
        return False

if __name__ == "__main__":
    print("=== Checklist Pré-Deployment da Aplicação Saori ===")
    deployment_readiness()
