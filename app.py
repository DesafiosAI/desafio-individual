import gradio as gr
import pandas as pd
import os
import matplotlib
import zipfile
import io
import traceback
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory

# --- CONFIGURAÇÃO INICIAL ---
# Define o backend do Matplotlib para evitar problemas em ambientes sem GUI
matplotlib.use('Agg')

# --- INSTRUÇÕES PARA O AGENTE (PROMPT DE SISTEMA) ---
AGENT_PREFIX = """
Você é um analista de dados de classe mundial, trabalhando com um DataFrame pandas em Python. Seu objetivo é ser uma ferramenta de E.D.A. (Análise Exploratória de Dados) útil, precisa e colaborativa.

Você tem acesso à seguinte ferramenta:
- python_repl_ast: Um REPL Python que pode executar comandos pandas no DataFrame `df`.

**DIRETRIZ CRÍTICA: ANÁLISE CONTEXTUAL E DE MÚLTIPLAS ETAPAS**
Sua memória (`chat_history`) contém as perguntas e respostas anteriores. Quando você recebe uma nova pergunta, é IMPERATIVO que você primeiro analise se ela é um acompanhamento da anterior.
- Se o usuário perguntar sobre "eles", "dessa turma", "esses dados", ou qualquer outro termo que se refira a um subconjunto de dados previamente discutido, sua **PRIMEIRA AÇÃO** deve ser recriar o filtro ou o subconjunto de dados da etapa anterior.
- Você deve realizar sua nova análise **APENAS** neste subconjunto de dados recriado, e não no DataFrame original completo.
- Esta é a regra mais importante. Falhar em manter o contexto resultará em respostas incorretas e contraditórias.

**Exemplo do Processo de Pensamento Correto:**
1.  **Usuário pergunta:** "Quantos professores estão na turma 0769?"
2.  **Seu pensamento:** "Preciso filtrar o DataFrame pela coluna 'TURMA' para encontrar o texto '0769' e depois contar as linhas."
3.  **Sua Ação:** `df[df['TURMA'].str.contains('0769', case=False)].shape[0]` -> **Resultado:** 15.
4.  **Sua Resposta:** "Existem 15 professores associados à turma 0769."
5.  **Usuário pergunta:** "Quais as diretorias de ensino deles?"
6.  **Seu pensamento:** "'Deles' refere-se aos 15 professores da pergunta anterior. Minha primeira ação é recriar esse filtro. Depois, encontrarei os valores únicos na coluna 'DIRETORIA DE ENSINO' desse grupo filtrado."
7.  **Sua Ação:** `filtered_df = df[df['TURMA'].str.contains('0769', case=False)]; filtered_df['DIRETORIA DE ENSINO'].unique()`
8.  **Sua Resposta:** "As diretorias de ensino para essa turma são: [lista de diretorias]."

**Outras Capacidades:**
-   **Busca Inteligente:** Use `str.contains(case=False)` para buscas parciais.
-   **Limpeza:** Remova duplicatas com `df.drop_duplicates(inplace=True)` quando solicitado.
-   **Análise Descritiva:** Use `df.info()`, `df.describe()`, etc., para descrições gerais.
-   **Visualização:** Crie gráficos com `matplotlib.pyplot` e salve-os em `temp_plot.png`.
-   **Proatividade:** Se um pedido for vago, peça esclarecimentos ou sugira uma alternativa.

**IMPRESCINDÍVEL: Todas as suas respostas, sem exceção, devem ser em português do Brasil.**
"""


# --- FUNÇÃO PRINCIPAL DO AGENTE ---
def processar_pergunta(historico_chat, agente_executor, memoria):
    """
    Processa a última pergunta do usuário, interage com o agente e retorna a resposta.
    """
    if os.path.exists('temp_plot.png'):
        os.remove('temp_plot.png')

    try:
        pergunta_usuario = historico_chat[-1]["content"]
        
        if "conclusões" in pergunta_usuario.lower():
            conversa_resumo = [f"Usuário: {msg['content']}" if msg['role'] == 'user' else f"Agente: {msg['content']}" for msg in historico_chat]
            resumo_conversa = "\n".join(conversa_resumo)
            
            prompt_conclusoes = f"""
            Com base no seguinte histórico de análise de dados, gere um resumo executivo com as principais conclusões e insights. Seja estruturado e direto.
            Histórico:
            {resumo_conversa}
            Conclusões Finais (em português do Brasil):
            """
            api_key = os.getenv("API_KEY")
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.1)
            resposta_final = llm.invoke(prompt_conclusoes).content
            historico_chat.append({"role": "assistant", "content": resposta_final})
            return historico_chat

        resposta_agente = agente_executor.invoke(pergunta_usuario)
        resposta_texto = resposta_agente['output']
        
        historico_chat.append({"role": "assistant", "content": resposta_texto})
        
        if os.path.exists('temp_plot.png'):
            historico_chat.append({"role": "assistant", "content": ("temp_plot.png",)})

    except Exception as e:
        # Tratamento de erro aprimorado: mostra o erro completo no chat
        error_details = traceback.format_exc()
        historico_chat.append({"role": "assistant", "content": f"Ocorreu um erro:\n\n```\n{e}\n\n{error_details}\n```"})
        
    return historico_chat

# --- FUNÇÕES DA INTERFACE ---
def criar_agente(arquivo):
    """
    Cria e retorna o agente e a memória quando um arquivo é carregado.
    """
    if arquivo is None:
        return None, None, gr.update(interactive=False), []
    
    try:
        nome_arquivo_original = arquivo.name
        df = None
        
        def carregar_csv_robusto(caminho_ou_buffer):
            try:
                return pd.read_csv(caminho_ou_buffer, sep=None, engine='python', on_bad_lines='warn', encoding='utf-8')
            except Exception as e:
                raise ValueError(f"Não foi possível analisar o arquivo CSV. Erro: {e}")

        if nome_arquivo_original.lower().endswith('.zip'):
            with zipfile.ZipFile(nome_arquivo_original, 'r') as zip_ref:
                nome_interno = next((nome for nome in zip_ref.namelist() if nome.lower().endswith(('.csv', '.xlsx', '.xls'))), None)
                if nome_interno:
                    with zip_ref.open(nome_interno) as f:
                        df = pd.read_excel(f) if nome_interno.lower().endswith(('.xlsx', '.xls')) else carregar_csv_robusto(f)
                    nome_arquivo_processado = nome_interno
                else:
                    raise ValueError("Nenhum arquivo .csv ou .xlsx encontrado dentro do .zip.")
        
        elif nome_arquivo_original.lower().endswith('.csv'):
            df = carregar_csv_robusto(nome_arquivo_original)
            nome_arquivo_processado = os.path.basename(nome_arquivo_original)

        elif nome_arquivo_original.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(nome_arquivo_original)
            nome_arquivo_processado = os.path.basename(nome_arquivo_original)
        
        else:
            raise ValueError("Formato de arquivo não suportado.")

        api_key = os.getenv("API_KEY")
        if not api_key:
            raise ValueError("Chave de API do Google não encontrada.")
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)
        memoria = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="output")
        agente = create_pandas_dataframe_agent(
            llm, df, prefix=AGENT_PREFIX, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            memory=memoria, verbose=True, handle_parsing_errors=True, allow_dangerous_code=True
        )
        
        mensagem_inicial = f"""
        Arquivo '{nome_arquivo_processado}' carregado com sucesso. Ele tem {df.shape[0]} linhas e {df.shape[1]} colunas.
        Eu sou seu assistente de análise de dados. Como posso ajudar?
        """
        historico_inicial = [{"role": "assistant", "content": mensagem_inicial}]

        return agente, memoria, gr.update(placeholder="Faça sua pergunta aqui...", interactive=True), historico_inicial
        
    except Exception as e:
        error_details = traceback.format_exc()
        # Exibe o erro de carregamento detalhado no chat
        historico_erro = [{"role": "assistant", "content": f"Erro ao carregar o arquivo:\n\n```\n{e}\n\n{error_details}\n```"}]
        return None, None, gr.update(value="", interactive=False), historico_erro

def adicionar_pergunta(historico, pergunta):
    return historico + [{"role": "user", "content": pergunta}], ""

# --- INTERFACE GRÁFICA COM GRADIO ---
with gr.Blocks(theme=gr.themes.Soft(), title="Agente de Análise de Dados") as demo:
    gr.Markdown("# 🤖 Agente Autônomo para Análise de Dados (E.D.A.)")
    gr.Markdown("Faça o upload de um arquivo (CSV, Excel, ZIP) e faça perguntas sobre seus dados em linguagem natural.")

    with gr.Row():
        with gr.Column(scale=1):
            arquivo_csv = gr.File(label="Carregue seu arquivo aqui", file_types=[".csv", ".zip", ".xlsx", ".xls"])
            gr.Markdown("---")
            gr.Markdown("### Exemplos de Perguntas:")
            gr.Markdown("- Descreva os dados.")
            gr.Markdown("- Verifique e remova linhas duplicadas.")
            gr.Markdown("- Gere um histograma para a coluna 'idade'.")
            gr.Markdown("- Compare a média de 'salário' por 'departamento'.")
            gr.Markdown("- Existem outliers na coluna 'vendas'? Mostre-me um boxplot.")
            gr.Markdown("- Quais são as suas conclusões?")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat de Análise", height=600, avatar_images=("user.png", "bot.png"), type="messages")
            caixa_texto = gr.Textbox(
                label="Faça a sua pergunta", 
                placeholder="Carregue um arquivo para começar...",
                interactive=False
            )
            botao_enviar = gr.Button("Enviar", variant="primary")
    
    estado_agente = gr.State()
    estado_memoria = gr.State()

    arquivo_csv.upload(criar_agente, arquivo_csv, [estado_agente, estado_memoria, caixa_texto, chatbot])
    
    # Ações de envio (botão e Enter)
    acao_submit = [adicionar_pergunta, [chatbot, caixa_texto], [chatbot, caixa_texto]]
    acao_processar = [processar_pergunta, [chatbot, estado_agente, estado_memoria], [chatbot]]
    
    caixa_texto.submit(*acao_submit).then(*acao_processar)
    botao_enviar.click(*acao_submit).then(*acao_processar)

if __name__ == "__main__":
    if os.getenv("API_KEY") is None:
        print("Carregando chave de API a partir do arquivo .env para execução local.")
        from dotenv import load_dotenv
        load_dotenv()
    
    demo.launch(debug=True)