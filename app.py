"""
APP INTERATIVO - PROJETO FINAL DE PCP (UnB)
Otimização de Políticas de Estoque de Segurança sob Variabilidade de Demanda
Apex Cimentos S.A. - Eixo Central: Gestão de Estoques

Este app reaproduz, de forma interativa, a análise feita no notebook:
1. Carga e diagnóstico dos dados de demanda
2. Previsão de demanda (3 modelos, escolhendo o melhor por MAPE)
3. Dimensionamento do Estoque de Segurança por nível de serviço
4. Simulação retrospectiva de rupturas (backtesting)
5. Conexão com a Lei de Little (WIP / Throughput)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import SimpleExpSmoothing
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# CONFIGURAÇÃO GERAL DA PÁGINA
# =========================================================
st.set_page_config(
    page_title="PCP - Estoque de Segurança | Apex Cimentos",
    page_icon="🏗️",
    layout="wide",
)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = [12, 5]
plt.rcParams["font.size"] = 11

NOME_ARQUIVO = "BaseDados_Demanda.xlsx"
NOME_ABA = "Dados_Demanda"
DATA_CORTE = "2026-05-01"


# =========================================================
# FUNÇÕES DE CÁLCULO (mesma lógica do notebook original)
# =========================================================
@st.cache_data
def carregar_dados():
    df = pd.read_excel(NOME_ARQUIVO, sheet_name=NOME_ABA)
    df["Data"] = pd.to_datetime(df["Data"], format="%d/%m/%Y")
    df = df.sort_values(by="Data").reset_index(drop=True)
    return df


@st.cache_data
def calcular_metricas_descritivas(df):
    metricas = {}
    for material in df["Material"].unique():
        df_mat = df[df["Material"] == material]
        media = df_mat["Vol.(TON)"].mean()
        desvio = df_mat["Vol.(TON)"].astype(float).std()
        cv = (desvio / media) * 100
        metricas[material] = {"media": media, "desvio": desvio, "cv": cv}
    return metricas


@st.cache_data
def rodar_previsoes(df):
    """Treina os 3 modelos por material e seleciona o vencedor pelo menor MAPE."""
    base_treino = df[df["Data"] < DATA_CORTE].reset_index(drop=True)
    base_teste = df[df["Data"] >= DATA_CORTE].reset_index(drop=True)

    resultados_erro = {}   # todos os modelos, para o gráfico comparativo
    melhores_modelos = {}  # só o vencedor de cada material

    for material in df["Material"].unique():
        treino_mat = base_treino[base_treino["Material"] == material].copy()
        teste_mat = base_teste[base_teste["Material"] == material].copy()

        y_real = teste_mat["Vol.(TON)"].values
        n_teste = len(teste_mat)

        # Modelo A: Média Móvel de 7 dias
        ultimo_valor = treino_mat["Vol.(TON)"].tail(7).mean()
        pred_ma = np.full(n_teste, ultimo_valor)

        # Modelo B: Suavização Exponencial Simples
        model_ses = SimpleExpSmoothing(treino_mat["Vol.(TON)"]).fit(
            smoothing_level=0.3, optimized=False
        )
        pred_ses = model_ses.forecast(n_teste).values

        # Modelo C: Regressão Linear Simples
        X_train = np.array(treino_mat.index).reshape(-1, 1)
        y_train = treino_mat["Vol.(TON)"].values
        X_test = np.array(teste_mat.index).reshape(-1, 1)
        model_lr = LinearRegression().fit(X_train, y_train)
        pred_lr = model_lr.predict(X_test)

        modelos = {
            "Média Móvel": pred_ma,
            "Suavização Exp.": pred_ses,
            "Regressão Linear": pred_lr,
        }

        erros_material = {}
        for nome, pred in modelos.items():
            mae = np.mean(np.abs(y_real - pred))
            rmse = np.sqrt(np.mean((y_real - pred) ** 2))
            mape = np.mean(np.abs((y_real - pred) / np.where(y_real == 0, 1, y_real))) * 100
            erros_material[nome] = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "predicoes": pred}

        vencedor = min(erros_material, key=lambda k: erros_material[k]["MAPE"])

        melhores_modelos[material] = {
            "modelo_nome": vencedor,
            "pred_valores": erros_material[vencedor]["predicoes"],
            "mae_vencedor": erros_material[vencedor]["MAE"],
            "rmse_vencedor": erros_material[vencedor]["RMSE"],
            "mape_vencedor": erros_material[vencedor]["MAPE"],
        }
        resultados_erro[material] = erros_material

    return base_treino, base_teste, resultados_erro, melhores_modelos


def calcular_estoque_seguranca(rmse, nivel_servico):
    z = norm.ppf(nivel_servico)
    return z * rmse, z


def simular_backtest(y_real, y_pred, es_inicial):
    """Reproduz a simulação de rupturas do notebook, guardando a série de estoque."""
    estoque_atual = es_inicial
    rupturas = 0
    volume_nao_atendido = 0
    serie_estoque = []

    for t in range(len(y_real)):
        estoque_atual += (y_pred[t] - y_real[t])
        serie_estoque.append(max(0, estoque_atual))
        if estoque_atual < 0:
            rupturas += 1
            volume_nao_atendido += abs(estoque_atual)
            estoque_atual = 0

    estoque_medio = np.mean(serie_estoque)
    fill_rate = ((y_real.sum() - volume_nao_atendido) / y_real.sum()) * 100
    return {
        "rupturas": rupturas,
        "fill_rate": fill_rate,
        "estoque_medio": estoque_medio,
        "serie_estoque": serie_estoque,
    }


# =========================================================
# CARGA DE DADOS
# =========================================================
try:
    df = carregar_dados()
except FileNotFoundError:
    st.error(
        f"Não encontrei o arquivo '{NOME_ARQUIVO}'. "
        "Ele precisa estar na MESMA pasta do arquivo app.py."
    )
    st.stop()

metricas = calcular_metricas_descritivas(df)
base_treino, base_teste, resultados_erro, melhores_modelos = rodar_previsoes(df)

# =========================================================
# BARRA LATERAL (CONTROLES DO USUÁRIO)
# =========================================================
st.sidebar.title("⚙️ Painel de Controle")
st.sidebar.markdown("Ajuste os parâmetros e veja os resultados mudarem em tempo real.")

materiais = sorted(df["Material"].unique().tolist())
material_selecionado = st.sidebar.selectbox("Selecione o Material (SKU):", materiais)

ns_percentual = st.sidebar.slider(
    "Nível de Serviço Alvo (%)",
    min_value=80.0,
    max_value=99.9,
    value=95.0,
    step=0.5,
    help="Probabilidade de NÃO haver ruptura de estoque no período.",
)
ns_decimal = ns_percentual / 100

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Sobre:** App desenvolvido a partir do Projeto Final de PCP - "
    "Eixo II (Gestão de Estoques), integrado com Previsão de Demanda "
    "e Dinâmica de Sistemas (Lei de Little)."
)

# =========================================================
# CABEÇALHO
# =========================================================
st.title("🏗️ Otimização de Estoque de Segurança — Apex Cimentos S.A.")
st.markdown(
    "Diagnóstico operacional sob variabilidade de demanda, integrando "
    "**previsão de demanda**, **gestão de estoques** e **dinâmica de fluxo (Lei de Little)**."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Visão Geral", "📈 Previsão de Demanda", "📦 Estoque & Simulação", "✅ Decisão Recomendada"]
)

# =========================================================
# TAB 1 - VISÃO GERAL
# =========================================================
with tab1:
    st.subheader("Série Histórica de Expedição (12 meses)")

    fig, ax = plt.subplots()
    for material in materiais:
        df_mat = df[df["Material"] == material]
        ax.plot(df_mat["Data"], df_mat["Vol.(TON)"], label=f"Material {material}", linewidth=1.3, alpha=0.85)
    ax.set_xlabel("Data")
    ax.set_ylabel("Volume Expedido (TON/dia)")
    ax.legend()
    ax.set_title("Expedição Diária por Material")
    st.pyplot(fig)

    st.subheader("Indicadores Estatísticos por Material")
    linhas = []
    for material in materiais:
        m = metricas[material]
        linhas.append({
            "Material": material,
            "Média Diária (TON)": round(m["media"], 2),
            "Desvio Padrão (TON)": round(m["desvio"], 2),
            "Coef. de Variação (%)": round(m["cv"], 2),
            "Volatilidade": "Alta" if m["cv"] > 50 else "Moderada",
        })
    st.dataframe(pd.DataFrame(linhas), use_container_width=True, hide_index=True)

    st.caption(
        f"Base treino: {len(base_treino)} registros | "
        f"Base teste (a partir de {DATA_CORTE}): {len(base_teste)} registros."
    )

# =========================================================
# TAB 2 - PREVISÃO DE DEMANDA
# =========================================================
with tab2:
    st.subheader(f"Comparação de Modelos — Material {material_selecionado}")

    erros_mat = resultados_erro[material_selecionado]
    tabela_erros = pd.DataFrame({
        nome: {"MAE (TON)": v["MAE"], "RMSE (TON)": v["RMSE"], "MAPE (%)": v["MAPE"]}
        for nome, v in erros_mat.items()
    }).T.round(2)
    st.dataframe(tabela_erros, use_container_width=True)

    vencedor = melhores_modelos[material_selecionado]["modelo_nome"]
    st.success(f"🏆 Modelo selecionado (menor MAPE): **{vencedor}**")

    st.markdown("**Previsto vs. Real — mês de teste**")
    teste_mat = base_teste[base_teste["Material"] == material_selecionado]
    y_real = teste_mat["Vol.(TON)"].values
    y_pred = melhores_modelos[material_selecionado]["pred_valores"]

    fig2, ax2 = plt.subplots()
    ax2.plot(teste_mat["Data"], y_real, label="Real", marker="o", markersize=3)
    ax2.plot(teste_mat["Data"], y_pred, label=f"Previsto ({vencedor})", linestyle="--")
    ax2.set_ylabel("Volume (TON)")
    ax2.legend()
    st.pyplot(fig2)

# =========================================================
# TAB 3 - ESTOQUE DE SEGURANÇA E SIMULAÇÃO
# =========================================================
with tab3:
    st.subheader(f"Material {material_selecionado} — Nível de Serviço: {ns_percentual:.1f}%")

    rmse_vencedor = melhores_modelos[material_selecionado]["rmse_vencedor"]
    es_calculado, z_fator = calcular_estoque_seguranca(rmse_vencedor, ns_decimal)

    teste_mat = base_teste[base_teste["Material"] == material_selecionado]
    y_real = teste_mat["Vol.(TON)"].values
    y_pred = melhores_modelos[material_selecionado]["pred_valores"]

    sim = simular_backtest(y_real, y_pred, es_calculado)
    throughput = metricas[material_selecionado]["media"]
    tempo_ciclo = sim["estoque_medio"] / throughput

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Estoque de Segurança", f"{es_calculado:.1f} TON", help=f"Fator Z = {z_fator:.2f}")
    col2.metric("Estoque Médio Simulado", f"{sim['estoque_medio']:.1f} TON")
    col3.metric("Rupturas no período", f"{sim['rupturas']}")
    col4.metric("Fill Rate Real", f"{sim['fill_rate']:.1f}%")

    st.metric("Tempo de Ciclo do Estoque (Lei de Little)", f"{tempo_ciclo:.2f} dias")

    st.markdown("**Evolução do estoque simulado ao longo do mês de teste**")
    fig3, ax3 = plt.subplots()
    ax3.plot(teste_mat["Data"], sim["serie_estoque"], color="darkorange", label="Estoque simulado")
    ax3.axhline(0, color="red", linestyle=":", label="Ruptura (zero)")
    ax3.set_ylabel("Estoque (TON)")
    ax3.legend()
    st.pyplot(fig3)

    st.markdown("---")
    st.subheader("Análise de Sensibilidade — Comparando os 3 Cenários Clássicos")
    linhas_cen = []
    for ns_fixo in [0.90, 0.95, 0.99]:
        es_fixo, _ = calcular_estoque_seguranca(rmse_vencedor, ns_fixo)
        sim_fixo = simular_backtest(y_real, y_pred, es_fixo)
        tc_fixo = sim_fixo["estoque_medio"] / throughput
        linhas_cen.append({
            "Nível de Serviço": f"{ns_fixo*100:.0f}%",
            "Estoque de Segurança (TON)": round(es_fixo, 2),
            "Estoque Médio (TON)": round(sim_fixo["estoque_medio"], 2),
            "Rupturas": sim_fixo["rupturas"],
            "Fill Rate Real (%)": round(sim_fixo["fill_rate"], 2),
            "Tempo de Ciclo (dias)": round(tc_fixo, 2),
        })
    st.dataframe(pd.DataFrame(linhas_cen), use_container_width=True, hide_index=True)

# =========================================================
# TAB 4 - DECISÃO RECOMENDADA
# =========================================================
with tab4:
    st.subheader("Recomendação de Política de Estoque")

    st.markdown(
        f"""
Para o material **{material_selecionado}**, com o nível de serviço configurado em
**{ns_percentual:.1f}%**, o modelo de previsão **{vencedor}** foi o mais preciso
(MAPE = {melhores_modelos[material_selecionado]['mape_vencedor']:.2f}%).

- **Estoque de Segurança recomendado:** {es_calculado:.1f} TON
- **Estoque médio necessário em operação:** {sim['estoque_medio']:.1f} TON
- **Rupturas esperadas no período analisado:** {sim['rupturas']}
- **Nível de serviço real obtido na simulação:** {sim['fill_rate']:.1f}%
- **Tempo de ciclo do estoque:** {tempo_ciclo:.2f} dias

**Trade-off:** aumentar o nível de serviço alvo reduz rupturas e eleva o fill rate,
mas exige mais estoque médio imobilizado (maior capital parado e maior tempo de
ciclo, pela Lei de Little). A tabela de sensibilidade na aba anterior mostra esse
trade-off entre os cenários de 90%, 95% e 99%.
"""
    )

    st.info(
        "Use o controle de nível de serviço na barra lateral para testar outros "
        "cenários e observar como o trade-off entre estoque, rupturas e tempo de "
        "ciclo se comporta para cada material."
    )
