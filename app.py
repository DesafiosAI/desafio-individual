import gradio as gr
import pandas as pd
import os
import matplotlib
import zipfile
import io
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory

# --- CONFIGURAÇÃO INICIAL ---
# Define o backend do Matplotlib para evitar problemas em ambientes sem GUI
matplotlib.use('Agg')

# --- INSTRUÇÕES PARA O AGENTE (PROMPT DE SISTEMA) ---
AGENT_PREFIX = """
Você é um analista de dados de classe mundial trabalhando com um DataFrame pandas em Python. Seu objetivo é ser uma ferramenta de E.D.A. (Análise Exploratória de Dados) útil, precisa e colaborativa.

Você tem acesso à seguinte ferramenta:
- python_repl_ast: Um REPL (Read-eval-print loop) Python que pode executar comandos pandas no DataFrame carregado. Use esta ferramenta para responder a perguntas sobre os dados. O DataFrame está disponível como `df`.

Capacidades de Análise:
-   **Limpeza de Dados:** Você é capaz de realizar operações de limpeza de dados. Ao ser solicitado, você pode:
    -   Verificar a existência de linhas duplicadas usando `df.duplicated().sum()`.
    -   Remover linhas duplicadas usando `df.drop_duplicates(inplace=True)`.
    -   Identificar e remover colunas que não são úteis para análise estatística (ex: colunas com identificadores únicos ou com variância zero) quando o usuário solicitar.
-   **Análise Descritiva:** Quando for solicitado a "descrever os dados" ou a fazer uma análise geral, você deve fornecer uma análise abrangente, não apenas uma amostra. Utilize comandos como `df.info()`, `df.describe()` para estatísticas numéricas, e verifique os valores nulos (`df.isnull().sum()`) para fornecer uma visão completa da estrutura e qualidade dos dados.
-   **Análise Estatística:** Você é capaz de realizar análises estatísticas. Isso inclui, mas não se limita a: comparar médias entre grupos (usando `groupby`), identificar outliers usando métodos estatísticos (como o Z-score ou IQR), e realizar testes de correlação.
-   **Visualização de Dados:** Ao ser solicitado a criar um gráfico ou visualização:
    1.  Use a biblioteca `matplotlib.pyplot`.
    2.  Gere o gráfico.
    3.  **Crucialmente, você DEVE salvar o gráfico em um arquivo chamado `temp_plot.png`**.
    4.  Após salvar o gráfico, descreva-o e o que ele mostra em sua resposta final.

Instruções Gerais:
-   Seja sempre fiel aos dados no DataFrame. Não invente ou alucine informações.
-   Ao fornecer uma resposta, primeiro pense passo a passo sobre como você chegará à solução.
-   Se um pedido do usuário for muito amplo ou vago (ex: 'crie um gráfico de tudo'), em vez de falhar, explique por que o pedido é problemático (ex: 'um gráfico com 28 variáveis seria ilegível') e sugira uma alternativa mais específica e útil (ex: 'Que tal um boxplot para as colunas X, Y e Z?').
-   Forneça respostas claras e concisas com base nos resultados da execução do seu código.
-   Se você não souber a resposta ou não conseguir encontrá-la nos dados, diga isso.
"""


# --- FUNÇÃO PRINCIPAL DO AGENTE ---
# Esta função é o cérebro da aplicação.
def processar_pergunta(historico_chat, agente_executor, memoria):
    """
    Processa a última pergunta do usuário, interage com o agente e retorna a resposta.
    """
    # Limpa qualquer gráfico da interação anterior para evitar mostrar o gráfico errado
    if os.path.exists('temp_plot.png'):
        os.remove('temp_plot.png')

    try:
        pergunta_usuario = historico_chat[-1]["content"]
        
        # Lógica especial para a pergunta sobre conclusões
        if "conclusões" in pergunta_usuario.lower():
            conversa_resumo = [msg["content"] for msg in memoria.chat_memory.messages if isinstance(msg.content, str)]
            resumo_conversa = "\n".join(conversa_resumo)
            
            prompt_conclusoes = f"""
            Com base na seguinte sessão de análise de dados, gere um resumo executivo com as principais conclusões e insights. Seja estruturado e direto.
            
            Histórico da Análise:
            {resumo_conversa}
            
            Conclusões Finais:
            """
            api_key = os.getenv("API_KEY")
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.1)
            resposta_final = llm.invoke(prompt_conclusoes).content
            historico_chat.append({"role": "assistant", "content": resposta_final})
            return historico_chat

        # Executa o agente para as outras perguntas
        resposta_agente = agente_executor.invoke(pergunta_usuario)
        resposta_texto = resposta_agente['output']
        
        # Adiciona a resposta de texto do agente ao histórico
        historico_chat.append({"role": "assistant", "content": resposta_texto})
        
        # Verifica se um gráfico foi criado e o adiciona como uma nova mensagem
        if os.path.exists('temp_plot.png'):
            # Adiciona o arquivo de imagem como uma nova mensagem do assistente
            historico_chat.append({"role": "assistant", "content": ("temp_plot.png",)})

    except Exception as e:
        historico_chat.append({"role": "assistant", "content": f"Ocorreu um erro: {e}"})
        
    return historico_chat

# --- FUNÇÕES DA INTERFACE ---
def criar_agente(arquivo):
    """
    Cria e retorna o agente e a memória quando um arquivo é carregado.
    Lida com arquivos .csv, .zip e .xlsx.
    """
    if arquivo is None:
        return None, None, gr.update(value="Por favor, carregue um arquivo (CSV, ZIP, Excel).", interactive=False), []
    
    try:
        nome_arquivo_original = arquivo.name
        df = None
        nome_arquivo_processado = ""
        
        def carregar_csv_robusto(caminho_ou_buffer):
            try:
                return pd.read_csv(caminho_ou_buffer, sep=None, engine='python', on_bad_lines='warn', encoding='utf-8')
            except Exception as e:
                raise ValueError(f"Não foi possível analisar o arquivo CSV. Erro: {e}")

        if nome_arquivo_original.lower().endswith('.zip'):
            with zipfile.ZipFile(nome_arquivo_original, 'r') as zip_ref:
                # Procura pelo primeiro arquivo csv ou excel dentro do zip
                nome_interno = next((nome for nome in zip_ref.namelist() if nome.lower().endswith(('.csv', '.xlsx', '.xls'))), None)
                
                if nome_interno:
                    with zip_ref.open(nome_interno) as f:
                        if nome_interno.lower().endswith('.csv'):
                            df = carregar_csv_robusto(f)
                        else:
                            df = pd.read_excel(f)
                    nome_arquivo_processado = nome_interno
                else:
                    raise ValueError("Nenhum arquivo .csv ou .xlsx encontrado dentro do arquivo .zip.")
        
        elif nome_arquivo_original.lower().endswith('.csv'):
            df = carregar_csv_robusto(nome_arquivo_original)
            nome_arquivo_processado = os.path.basename(nome_arquivo_original)

        elif nome_arquivo_original.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(nome_arquivo_original)
            nome_arquivo_processado = os.path.basename(nome_arquivo_original)
        
        else:
            raise ValueError("Formato de arquivo não suportado. Por favor, carregue um .csv, .zip, ou .xlsx.")

        api_key = os.getenv("API_KEY")
        if not api_key:
            raise ValueError("Chave de API do Google não encontrada. Verifique seu arquivo .env ou as variáveis de ambiente.")
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)
        
        memoria = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="output")
        
        agente = create_pandas_dataframe_agent(
            llm, 
            df, 
            prefix=AGENT_PREFIX,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            memory=memoria, 
            verbose=True, 
            handle_parsing_errors=True,
            allow_dangerous_code=True
        )
        
        mensagem_inicial = f"""
        Arquivo '{nome_arquivo_processado}' carregado com sucesso. Ele tem {df.shape[0]} linhas e {df.shape[1]} colunas.

        Eu sou seu assistente de análise de dados. Você pode me pedir para:
        - **Descrever os dados** (estatísticas, tipos de colunas, valores nulos).
        - **Limpar os dados** (ex: remover duplicatas).
        - **Realizar análises estatísticas** (ex: comparar médias, correlações).
        - **Criar visualizações** (ex: histogramas, gráficos de dispersão, boxplots).

        Pode começar a perguntar!
        """
        historico_inicial = [{"role": "assistant", "content": mensagem_inicial}]

        return agente, memoria, gr.update(value="", placeholder="Faça sua pergunta aqui...", interactive=True), historico_inicial
        
    except Exception as e:
        return None, None, gr.update(value=f"Erro ao carregar o arquivo: {e}", interactive=False), []

def adicionar_pergunta(historico, pergunta):
    historico.append({"role": "user", "content": pergunta})
    return historico, ""

# --- INTERFACE GRÁFICA COM GRADIO ---
with gr.Blocks(theme=gr.themes.Soft(), title="Agente de Análise de Dados") as demo:
    gr.Markdown("# 🤖 Agente Autônomo para Análise de Dados (E.D.A.)")
    gr.Markdown("Faça o upload de um arquivo CSV, Excel (ou um ZIP contendo um) e faça perguntas sobre seus dados em linguagem natural.")

    with gr.Row():
        with gr.Column(scale=1):
            arquivo_csv = gr.File(label="Carregue seu arquivo aqui (CSV, ZIP, XLSX)", file_types=[".csv", ".zip", ".xlsx", ".xls"])
            gr.Markdown("---")
            gr.Markdown("### Perguntas de Exemplo:")
            gr.Markdown("- Descreva os dados.")
            gr.Markdown("- Verifique e remova as linhas duplicadas.")
            gr.Markdown("- Gere um histograma para a coluna 'idade'.")
            gr.Markdown("- Compare a média de 'salário' por 'departamento'.")
            gr.Markdown("- Existem outliers na coluna 'vendas'? Mostre-me um boxplot.")
            gr.Markdown("- Quais são as suas conclusões?")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat de Análise", height=600, avatar_images=("user.png", "bot.png"), type="messages")
            caixa_texto = gr.Textbox(
                label="Faça a sua pergunta aqui...", 
                placeholder="Ex: Qual a média da coluna 'idade'?",
                interactive=False
            )
    
    estado_agente = gr.State()
    estado_memoria = gr.State()

    arquivo_csv.upload(criar_agente, arquivo_csv, [estado_agente, estado_memoria, caixa_texto, chatbot])
    
    caixa_texto.submit(
        adicionar_pergunta, [chatbot, caixa_texto], [chatbot, caixa_texto]
    ).then(
        processar_pergunta, [chatbot, estado_agente, estado_memoria], [chatbot]
    )

if __name__ == "__main__":
    if os.getenv("API_KEY") is None:
        print("Carregando chave de API a partir do arquivo .env para execução local.")
        from dotenv import load_dotenv
        load_dotenv()
    
    demo.launch(debug=True)