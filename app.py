import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ==========================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(page_title="Dashboard de Risco - FPD & Safras", layout="wide")

# ==========================================
# FUNÇÕES MATEMÁTICAS E TRATAMENTO
# ==========================================
def calc_xirr(cashflows, days):
    """
    Função matemática blindada para calcular a TIR Anualizada (XIRR).
    Utiliza o método da bisseção e lida com cenários extremos (ex: 100% de perda).
    """
    # Remove dias com CF zerado para otimizar
    cf_days = [(c, d) for c, d in zip(cashflows, days) if c != 0]
    if not cf_days: return 0.0
    
    has_pos = any(c > 0 for c, d in cf_days)
    has_neg = any(c < 0 for c, d in cf_days)
    
    if not has_pos:
        return -1.0 # Sem fluxos positivos: perda de 100% (-100% TIR)
    if not has_neg:
        return 0.0  # Sem investimento inicial: não há como calcular TIR
    
    def xnpv(rate):
        if rate <= -1.0: return float('inf')
        try:
            return sum([c / (1.0 + rate)**(d / 365.0) for c, d in cf_days])
        except:
            return float('inf')

    # Limites de busca: de -99.9% a 100.000%
    left, right = -0.999, 1000.0
            
    # Bisseção
    for _ in range(100):
        mid = (left + right) / 2.0
        val_mid = xnpv(mid)
        
        if abs(val_mid) < 1e-5:
            break
            
        # Na estrutura normal (CF0 < 0, CFn > 0), o VPL cai conforme a taxa sobe
        if xnpv(left) > 0:
            if val_mid > 0:
                left = mid
            else:
                right = mid
        else:
            if val_mid > 0:
                right = mid
            else:
                left = mid
                
    return mid

@st.cache_data
def calcular_vintage_par_otimizado(df_base, data_ref_global, dias_par=90):
    """
    Realiza uma análise de PAR (Portfolio at Risk) Vintage otimizada.
    Avalia em lote o status de cada contrato no último dia de cada mês (Snapshot).
    Se qualquer parcela estiver atrasada > X dias na data do snapshot, o saldo 
    remanescente do contrato inteiro (Efeito Vagão) é considerado em risco.
    """
    df = df_base.copy()
    
    # Identificador primário do contrato
    id_col = 'CCB_NUMEROCCB' if 'CCB_NUMEROCCB' in df.columns else ('PROPOSTA_ID' if 'PROPOSTA_ID' in df.columns else None)
    if not id_col:
        return pd.DataFrame()
        
    for col in ['DATA_VENCIMENTO', 'VALOR_DA_PARCELA', 'SAFRA']:
        if col not in df.columns:
            return pd.DataFrame()
            
    df = df.dropna(subset=['DATA_VENCIMENTO', 'VALOR_DA_PARCELA', 'SAFRA']).copy()
    
    # Converte safra para Period (Mês)
    df['mes_safra'] = pd.to_datetime(df['SAFRA']).dt.to_period('M')
    data_final_analise = pd.to_datetime(data_ref_global)
    
    # --- Cálculos Iniciais Agregados ---
    # Valor Total Originado por contrato e safra (Denominador Real)
    map_valor_total_contrato = df.groupby(id_col)['VALOR_DA_PARCELA'].sum()
    map_valor_originado_safra = df.drop_duplicates(subset=[id_col]).groupby('mes_safra')[id_col].apply(lambda x: map_valor_total_contrato[x].sum())
    
    lista_safras = sorted(df['mes_safra'].unique())
    resultados_por_safra = []
    
    for safra_atual in lista_safras:
        df_safra = df[df['mes_safra'] == safra_atual].copy()
        contratos_da_safra = df_safra[[id_col]].drop_duplicates()
        
        # Mapeia a linha do tempo (Meses on Book) da safra atual até a data de referência
        meses_analise_safra = pd.period_range(start=safra_atual, end=data_final_analise.to_period('M'), freq='M')
        if len(meses_analise_safra) == 0:
            continue
            
        # Cria Matriz: [Todos os Contratos da Safra] x [Todos os Meses de Análise]
        df_vintage = pd.MultiIndex.from_product([contratos_da_safra[id_col], meses_analise_safra], names=[id_col, 'mes_analise']).to_frame(index=False)
        df_vintage['mes_safra'] = safra_atual
        
        # Faz o join com as parcelas (multiplica as linhas, teremos Parcela x Mês)
        # Correção do KeyError: Usa [id_col, 'mes_safra'] para evitar renomeação _x e _y das colunas
        df_vintage = pd.merge(df_vintage, df_safra, on=[id_col, 'mes_safra'], how='left')
        df_vintage['data_fim_mes_analise'] = df_vintage['mes_analise'].dt.to_timestamp(how='end')
        
        # Trava a data do snapshot para não olhar o futuro além da data de referência global do painel
        df_vintage['data_corte_snapshot'] = df_vintage['data_fim_mes_analise'].clip(upper=data_final_analise)
        
        # --- Lógica de Snapshots Point-in-Time ---
        # 1. Simula a coluna VALOR_PAGO na foto daquele mês
        valor_pago_ate_snapshot = np.where(
            (df_vintage['DATA_PAGAMENTO'].notna()) & (df_vintage['DATA_PAGAMENTO'] <= df_vintage['data_corte_snapshot']),
            df_vintage['VALOR_DA_PARCELA'], 0
        )
        
        valor_remanescente_snapshot = df_vintage['VALOR_DA_PARCELA'] - valor_pago_ate_snapshot
        parcela_nao_quitada_snapshot = valor_remanescente_snapshot >= 0.01
        
        dias_de_atraso_snapshot = (df_vintage['data_corte_snapshot'] - df_vintage['DATA_VENCIMENTO']).dt.days
        
        # 2. Gatilho de Atraso
        gatilho_snapshot = parcela_nao_quitada_snapshot & (df_vintage['DATA_VENCIMENTO'] <= df_vintage['data_corte_snapshot']) & (dias_de_atraso_snapshot > dias_par)
        df_vintage['gatilho_parcela'] = gatilho_snapshot
        
        # 3. Efeito Vagão
        contrato_inadimplente_no_mes = df_vintage.groupby([id_col, 'mes_analise'])['gatilho_parcela'].transform('any')
        
        df_vintage['valor_remanescente_snapshot'] = valor_remanescente_snapshot
        map_saldo_devedor_fim_mes = df_vintage.groupby([id_col, 'mes_analise'])['valor_remanescente_snapshot'].transform('sum')
        
        df_vintage['valor_atrasado_final'] = np.where(contrato_inadimplente_no_mes, map_saldo_devedor_fim_mes, 0)
        
        # 4. Agrega resultados e limpa linhas multiplicadas pelas parcelas
        resultados_agg = df_vintage.drop_duplicates(subset=[id_col, 'mes_analise'])
        df_final_safra = resultados_agg.groupby(['mes_safra', 'mes_analise'])['valor_atrasado_final'].sum().reset_index()
        
        # Traz os valores base do Total Originado para calcular o % de PAR
        df_final_safra['Total_Originado'] = map_valor_originado_safra.get(safra_atual, 0)
        df_final_safra['PAR (%)'] = 0.0
        mask_orig = df_final_safra['Total_Originado'] > 0
        df_final_safra.loc[mask_orig, 'PAR (%)'] = (df_final_safra.loc[mask_orig, 'valor_atrasado_final'] / df_final_safra.loc[mask_orig, 'Total_Originado']) * 100
        
        # MOB = Month on Book
        df_final_safra['MOB'] = (df_final_safra['mes_analise'] - df_final_safra['mes_safra']).apply(lambda x: x.n)
        
        resultados_por_safra.append(df_final_safra)
        
    if resultados_por_safra:
        return pd.concat(resultados_por_safra, ignore_index=True)
    return pd.DataFrame()

@st.cache_data
def load_data(uploaded_file):
    try:
        # Lê o CSV tentando diferentes separadores. Padrão no Brasil é ';'
        try:
            # Tenta primeiro com ponto e vírgula, ignorando linhas corrompidas
            df = pd.read_csv(uploaded_file, sep=';', low_memory=False, on_bad_lines='skip')
            # Se leu tudo como uma coluna só, o separador provavelmente era vírgula
            if len(df.columns) == 1:
                raise ValueError("Separador incorreto")
        except:
            # Retorna o ponteiro do arquivo para o início
            uploaded_file.seek(0)
            # Tenta com vírgula
            df = pd.read_csv(uploaded_file, sep=',', low_memory=False, on_bad_lines='skip')
        
        # Tratamento de Datas
        df['DATA_VENCIMENTO'] = pd.to_datetime(df['DATA_VENCIMENTO'], errors='coerce')
        df['DATA_PAGAMENTO'] = pd.to_datetime(df['DATA_PAGAMENTO'], errors='coerce')
        df['DATA_AVERBACAO'] = pd.to_datetime(df['DATA_AVERBACAO'], errors='coerce')
        
        # Preenchimento de Nulos em valores financeiros cruciais com conversão SEGURA
        cols_financeiras = ['SALDO_ATRASO', 'SALDO_NAO_ESCRIT', 'VALOR_DA_PARCELA', 'VALOR_PRINCIPAL', 'TX_JUROS_MES', 'VALOR_PRINCIPAL_JUROS']
        for col in cols_financeiras:
            if col in df.columns:
                # Função SUPER segura para lidar com números (resolve sumiço de milhões por causa de separadores)
                if df[col].dtype == object:
                    def safe_float_convert(val):
                        try:
                            if pd.isna(val): return 0.0
                            val_str = str(val).strip().replace(' ', '')
                            if not val_str: return 0.0
                            
                            last_comma = val_str.rfind(',')
                            last_dot = val_str.rfind('.')
                            
                            if last_comma > -1 and last_dot > -1:
                                if last_comma > last_dot:
                                    # Padrão BR: Ex -> 1.000.000,00
                                    val_str = val_str.replace('.', '').replace(',', '.')
                                else:
                                    # Padrão US: Ex -> 1,000,000.00
                                    val_str = val_str.replace(',', '')
                            elif last_comma > -1:
                                if val_str.count(',') > 1:
                                    # Ex: 1,000,000
                                    val_str = val_str.replace(',', '')
                                else:
                                    # Ex: 1000,00
                                    val_str = val_str.replace(',', '.')
                            elif last_dot > -1:
                                if val_str.count('.') > 1:
                                    # Ex: 1.000.000
                                    val_str = val_str.replace('.', '')
                            
                            return float(val_str)
                        except:
                            return 0.0
                    df[col] = df[col].apply(safe_float_convert)
                df[col] = df[col].fillna(0)
        
        # Tratamento da Safra e Criação de Variáveis a Valor Presente
        if 'SAFRA' in df.columns:
            # Converte para string e remove '.0' caso o pandas tenha lido como número flutuante
            df['SAFRA'] = df['SAFRA'].astype(str).str.replace(r'\.0$', '', regex=True)
            # Converte para formato Data considerando Ano e Mês (força o dia 01)
            df['SAFRA'] = pd.to_datetime(df['SAFRA'], format='%Y%m', errors='coerce').dt.date
            
            # --- CÁLCULO DE VALOR PRESENTE NA ORIGINAÇÃO ---
            if 'DATA_VENCIMENTO' in df.columns and 'TX_JUROS_MES' in df.columns:
                # 1. Utilizar a DATA_AVERBACAO como Data de Originação
                if 'DATA_AVERBACAO' in df.columns:
                    df['DATA_ORIGINACAO'] = df['DATA_AVERBACAO']
                else:
                    df['DATA_ORIGINACAO'] = pd.NaT
                
                # 2. Calcular Dias Úteis entre a Originação e o Vencimento (Filtro base do Numpy: Seg a Sex)
                valid_dates = df['DATA_VENCIMENTO'].notna() & df['DATA_ORIGINACAO'].notna()
                
                if valid_dates.any():
                    orig_dates = df.loc[valid_dates, 'DATA_ORIGINACAO'].values.astype('datetime64[D]')
                    venc_dates = df.loc[valid_dates, 'DATA_VENCIMENTO'].values.astype('datetime64[D]')
                    
                    # Motor de contagem de dias úteis
                    prazo_du = np.busday_count(orig_dates, venc_dates)
                    
                    df['PRAZO_DU'] = np.nan
                    df.loc[valid_dates, 'PRAZO_DU'] = prazo_du
                    df['PRAZO_DU'] = df['PRAZO_DU'].clip(lower=0) # Trava em 0 caso o vencimento seja anterior
                    
                    # 3. Trazer o Valor da Parcela a Valor Presente
                    # Fórmula: VP = VF / ((1 + Taxa) ^ (Dias_Uteis / 21))
                    df['Valor_Principal_Originacao'] = df['VALOR_DA_PARCELA'] / ((1 + df['TX_JUROS_MES']) ** (df['PRAZO_DU'] / 21.0))
                    df['Valor_Principal_Originacao'] = df['Valor_Principal_Originacao'].fillna(0)
                else:
                    df['Valor_Principal_Originacao'] = 0.0
            else:
                df['Valor_Principal_Originacao'] = 0.0
            
        # Garante que NUMERO_PARCELA é interpretado como número
        if 'NUMERO_PARCELA' in df.columns:
            # Limpa qualquer formato em texto (ex: "1/12" ou "01") garantindo a leitura do dígito
            if df['NUMERO_PARCELA'].dtype == object:
                df['NUMERO_PARCELA'] = df['NUMERO_PARCELA'].astype(str).str.extract(r'(\d+)', expand=False)
            df['NUMERO_PARCELA'] = pd.to_numeric(df['NUMERO_PARCELA'], errors='coerce')
                
        return df
    except Exception as e:
        st.error(f"Erro ao processar o arquivo CSV: {e}")
        return None

# ==========================================
# INTERFACE DO USUÁRIO (SIDEBAR)
# ==========================================
st.sidebar.title("📊 Upload de Dados")
uploaded_file = st.sidebar.file_uploader("Faça o upload do arquivo 'Dados.csv'", type=['csv'])

if uploaded_file is not None:
    # Carrega os dados (em cache)
    df = load_data(uploaded_file)
    
    if df is not None and not df.empty:
        # ------------------------------------------
        # CONFIGURAÇÃO GERAL E DA DATA DE REFERÊNCIA
        # ------------------------------------------
        st.sidebar.markdown("---")
        st.sidebar.title("📅 Configurações Globais")
        
        # Define o default como a maior data de vencimento encontrada na base (ou hoje, se tudo for nulo)
        if 'DATA_VENCIMENTO' in df.columns:
            max_vencimento = df['DATA_VENCIMENTO'].max()
            default_date = max_vencimento.date() if pd.notnull(max_vencimento) else datetime.today().date()
        else:
            default_date = datetime.today().date()
        
        data_input = st.sidebar.date_input(
            "Data de Referência (Cálculo de Atraso)", 
            value=default_date,
            help="Esta data atua como uma 'Fotografia do Momento'. Pagamentos feitos APÓS esta data serão ignorados (considerados não pagos) para fins da análise."
        )
        # Mantém como Timestamp nativo do Pandas para evitar erros de comparação no XIRR
        data_referencia = pd.to_datetime(data_input)

        # Filtro de dias de tolerância para FPD/SPD/TDP
        dias_atraso_tolerancia = st.sidebar.number_input(
            "Tolerância FPD/SPD/TDP (Dias)",
            min_value=0, max_value=365, value=15, step=1,
            help="Número de dias de atraso permitidos antes de considerar a parcela inadimplente para o cálculo de FPD, SPD e TDP."
        )
        
        # Novo Filtro: Controle dos dias para a métrica de Perda (Over X) e PAR
        dias_over = st.sidebar.number_input(
            "Tolerância Perda / PAR (Dias)",
            min_value=0, max_value=3650, value=90, step=1,
            help="Número de dias para definir o corte do indicador de Perda e da Curva Vintage de PAR (ex: 90 dias para 'Over 90')."
        )
        
        # Cálculo dinâmico de Dias de Atraso e Over X com base na data selecionada e histórico de pagamento
        if 'DATA_VENCIMENTO' in df.columns and 'DATA_PAGAMENTO' in df.columns:
            
            # SNAPSHOT TEMPORAL: A parcela já estava paga NA data de referência?
            pago_ate_ref = df['DATA_PAGAMENTO'].notna() & (df['DATA_PAGAMENTO'] <= data_referencia)
            nao_pago_ate_ref = ~pago_ate_ref # Inclui os nulos e os pagos APÓS a data de referência
            
            # Atraso para quem NÃO pagou até a referência (calculado da referência até o vencimento)
            atraso_pendente = (data_referencia - df['DATA_VENCIMENTO']).dt.days
            
            # Atraso real para quem JÁ pagou até a referência (Data Pagamento - Data Vencimento)
            atraso_pago = (df['DATA_PAGAMENTO'] - df['DATA_VENCIMENTO']).dt.days
            
            # Combina as lógicas baseando-se no status "fotografado" na data de referência
            df['DIAS_ATRASO'] = atraso_pago.where(pago_ate_ref, atraso_pendente)
            
            # Identificador base para o contrato. Tenta 'PROPOSTA_ID', senão 'CCB_NUMEROCCB', senão usa a própria linha
            id_col = 'PROPOSTA_ID' if 'PROPOSTA_ID' in df.columns else ('CCB_NUMEROCCB' if 'CCB_NUMEROCCB' in df.columns else None)
            
            # Avalia quais parcelas individualmente estão Over X (na data de referência) usando o novo filtro
            df['PARCELA_OVER'] = df['DIAS_ATRASO'] > dias_over
            
            if id_col:
                # Efeito Vagão: Se uma parcela do contrato for > X, o contrato todo recebe a flag
                contratos_over = df[df['PARCELA_OVER']][id_col].unique()
                df['CONTRATO_OVER'] = df[id_col].isin(contratos_over)
                
                # A parcela entra na conta de OVER financeiro se o contrato é Over E ELA PRÓPRIA NÃO estava paga na data de ref.
                df['IS_OVER'] = df['CONTRATO_OVER'] & nao_pago_ate_ref
            else:
                # Fallback caso não ache coluna de agrupamento (não deve acontecer com sua base)
                df['IS_OVER'] = df['PARCELA_OVER'] & nao_pago_ate_ref
                
            df['VALOR_OVER'] = df['VALOR_DA_PARCELA'].where(df['IS_OVER'], 0)
            
            # Regra de Inadimplência (> X dias baseados no filtro) para FPD, SPD e TDP
            df['IS_DEFAULT'] = df['DIAS_ATRASO'] > dias_atraso_tolerancia
                
            df['VALOR_INADIMPLENTE'] = df['VALOR_DA_PARCELA'].where(df['IS_DEFAULT'], 0)
        else:
            df['VALOR_OVER'] = 0
            df['IS_DEFAULT'] = False
            df['VALOR_INADIMPLENTE'] = 0

        # ------------------------------------------
        # FILTROS
        # ------------------------------------------
        st.sidebar.markdown("---")
        st.sidebar.title("🔍 Filtros de Análise")
        
        produtos = df['TIPO_PRODUTO'].dropna().unique().tolist() if 'TIPO_PRODUTO' in df.columns else []
        fx_toj = df['FX_TOJ'].dropna().unique().tolist() if 'FX_TOJ' in df.columns else []
        status_emp = df['STATUS_EMPREGADA'].dropna().unique().tolist() if 'STATUS_EMPREGADA' in df.columns else []
        cnpjs = df['CNPJ_RAIZ'].dropna().unique().tolist() if 'CNPJ_RAIZ' in df.columns else []
        safras = sorted([s for s in df['SAFRA'].unique() if pd.notnull(s)]) if 'SAFRA' in df.columns else []
        tipos_fundo = df['TIPO_FUNDO'].dropna().unique().tolist() if 'TIPO_FUNDO' in df.columns else []
        
        f_produto = st.sidebar.multiselect("Tipo de Produto", produtos, default=produtos) if produtos else []
        f_fx_toj = st.sidebar.multiselect("Faixa TOJ (FX_TOJ)", fx_toj, default=fx_toj) if fx_toj else []
        f_status = st.sidebar.multiselect("Status Empregada", status_emp, default=status_emp) if status_emp else []
        f_cnpj = st.sidebar.multiselect("CNPJ Raiz (Top 50)", cnpjs[:50]) if cnpjs else []
        f_fundo = st.sidebar.multiselect("Tipo de Fundo", tipos_fundo, default=tipos_fundo) if tipos_fundo else []
        f_safra = st.sidebar.multiselect("Safra", safras, default=safras) if safras else []
        
        # Aplicação dos Filtros
        df_filtered = df.copy()
        if f_produto: df_filtered = df_filtered[df_filtered['TIPO_PRODUTO'].isin(f_produto)]
        if f_fx_toj: df_filtered = df_filtered[df_filtered['FX_TOJ'].isin(f_fx_toj)]
        if f_status: df_filtered = df_filtered[df_filtered['STATUS_EMPREGADA'].isin(f_status)]
        if f_cnpj: df_filtered = df_filtered[df_filtered['CNPJ_RAIZ'].isin(f_cnpj)]
        if f_fundo: df_filtered = df_filtered[df_filtered['TIPO_FUNDO'].isin(f_fundo)]
        if f_safra: df_filtered = df_filtered[df_filtered['SAFRA'].isin(f_safra)]

        st.title("📈 Análise de Risco de Crédito Institucional")
        
        # --- CÁLCULO DINÂMICO DE VALOR PRESENTE (DATA DE REFERÊNCIA) ---
        # Este bloco garante que contratos futuros em aberto sejam trazidos a VP para o offset da TIR
        if 'DATA_VENCIMENTO' in df_filtered.columns and 'TX_JUROS_MES' in df_filtered.columns and 'VALOR_DA_PARCELA' in df_filtered.columns:
            # Inicializa a coluna (o padrão para parcelas já vencidas é o valor de face da parcela)
            df_filtered['Valor_Presente_Calculado'] = df_filtered['VALOR_DA_PARCELA'].copy()
            
            # Identifica as parcelas a vencer (cujo vencimento é posterior à data de referência do snapshot)
            mask_a_vencer = (df_filtered['DATA_VENCIMENTO'] > data_referencia) & df_filtered['DATA_VENCIMENTO'].notna()
            
            if mask_a_vencer.any():
                # Formatação de datas para contagem exata de dias úteis com o Numpy
                ref_array = np.full(mask_a_vencer.sum(), data_referencia.date(), dtype='datetime64[D]')
                venc_array = df_filtered.loc[mask_a_vencer, 'DATA_VENCIMENTO'].dt.date.values.astype('datetime64[D]')
                
                # Conta os dias úteis entre a data de referência e o vencimento futuro
                du_a_vencer = np.busday_count(ref_array, venc_array)
                du_a_vencer = np.clip(du_a_vencer, 0, None)
                
                # Formula Financeira de desconto: VP = VF / ((1 + Taxa) ^ (DiasUteis / 21))
                tx = df_filtered.loc[mask_a_vencer, 'TX_JUROS_MES']
                parcela_futura = df_filtered.loc[mask_a_vencer, 'VALOR_DA_PARCELA']
                df_filtered.loc[mask_a_vencer, 'Valor_Presente_Calculado'] = parcela_futura / ((1 + tx) ** (du_a_vencer / 21.0))
        else:
            df_filtered['Valor_Presente_Calculado'] = df_filtered['VALOR_DA_PARCELA'] if 'VALOR_DA_PARCELA' in df_filtered.columns else 0.0

        # Correção no Global: Revertido para o uso seguro de NUMERO_PARCELA == 1 para pegar o valor total originado fidedigno
        if 'VALOR_DA_PARCELA' in df_filtered.columns and 'NUMERO_PARCELA' in df_filtered.columns:
            volume_originado = df_filtered[df_filtered['NUMERO_PARCELA'] == 1]['VALOR_DA_PARCELA'].sum()
            st.markdown(f"**Volume Total Originado Filtrado (Valor da Parcela - Parcela 1):** R$ {volume_originado:,.2f}")
        else:
            volume_total = df_filtered['VALOR_DA_PARCELA'].sum() if 'VALOR_DA_PARCELA' in df_filtered.columns else 0
            st.markdown(f"**Volume Total Filtrado (Soma de Parcelas):** R$ {volume_total:,.2f}")
        
        # ==========================================
        # ABAS DO DASHBOARD
        # ==========================================
        tab_orig, tab_tir, tab1, tab2, tab3, tab_coll, tab_par = st.tabs([
            "Originação",
            "Análise de TIR",
            "FPD, SPD e TDP", 
            "Análise de Não Escriturado", 
            f"FPD vs Perda Over {dias_over}",
            "Collection",
            "PAR (Vintage)"
        ])

        # ------------------------------------------
        # TAB ORIGINAÇÃO
        # ------------------------------------------
        with tab_orig:
            st.header("Originação por Safra")
            st.markdown("Comparativo do montante de originação entre o **Valor da Parcela** e o **Valor Presente Calculado (Valor Principal Originação)**, considerando **todas as parcelas** do contrato. A Taxa e o Prazo Médio são ponderados pelo **Valor Principal com Juros** máximo de cada contrato.")

            if 'SAFRA' in df_filtered.columns and 'VALOR_DA_PARCELA' in df_filtered.columns and 'Valor_Principal_Originacao' in df_filtered.columns:
                
                id_col_orig = 'CCB_NUMEROCCB' if 'CCB_NUMEROCCB' in df_filtered.columns else ('PROPOSTA_ID' if 'PROPOSTA_ID' in df_filtered.columns else None)
                
                # 1. Volume de Originação (Soma de todas as parcelas)
                agg_orig = df_filtered.groupby('SAFRA').agg({
                    'VALOR_DA_PARCELA': 'sum',
                    'Valor_Principal_Originacao': 'sum'
                }).reset_index()
                
                agg_orig.rename(columns={
                    'VALOR_DA_PARCELA': 'Soma_Valor_Parcela',
                    'Valor_Principal_Originacao': 'Soma_VP_Calculado'
                }, inplace=True)

                # 2. Cálculo das Médias Ponderadas (Agrupado por Contrato Único)
                if id_col_orig and 'VALOR_PRINCIPAL_JUROS' in df_filtered.columns:
                    agg_dict = {'VALOR_PRINCIPAL_JUROS': 'max'}
                    if 'TX_JUROS_MES' in df_filtered.columns:
                        agg_dict['TX_JUROS_MES'] = 'first'
                    if 'PRAZO_DU' in df_filtered.columns:
                        agg_dict['PRAZO_DU'] = 'max' # O Prazo total do contrato é o prazo da última parcela
                        
                    # Extrai um registro único por contrato
                    df_contracts = df_filtered.groupby(['SAFRA', id_col_orig]).agg(agg_dict).reset_index()
                    
                    # Prepara as multiplicações usando o Valor_Principal_Juros máximo como PESO
                    if 'TX_JUROS_MES' in df_contracts.columns:
                        df_contracts['TX_VP'] = df_contracts['TX_JUROS_MES'] * df_contracts['VALOR_PRINCIPAL_JUROS']
                    if 'PRAZO_DU' in df_contracts.columns:
                        df_contracts['PRAZO_VP'] = df_contracts['PRAZO_DU'] * df_contracts['VALOR_PRINCIPAL_JUROS']
                        
                    # Agrupa tudo por Safra
                    agg_weights_cols = {'VALOR_PRINCIPAL_JUROS': 'sum'}
                    if 'TX_VP' in df_contracts.columns: agg_weights_cols['TX_VP'] = 'sum'
                    if 'PRAZO_VP' in df_contracts.columns: agg_weights_cols['PRAZO_VP'] = 'sum'
                    
                    agg_weights = df_contracts.groupby('SAFRA').agg(agg_weights_cols).reset_index()
                    
                    mask = agg_weights['VALOR_PRINCIPAL_JUROS'] > 0
                    
                    # Realiza a divisão ponderada final
                    if 'TX_VP' in agg_weights.columns:
                        agg_weights['TX_MEDIA_PONDERADA'] = 0.0
                        agg_weights.loc[mask, 'TX_MEDIA_PONDERADA'] = agg_weights.loc[mask, 'TX_VP'] / agg_weights.loc[mask, 'VALOR_PRINCIPAL_JUROS']
                        agg_orig = pd.merge(agg_orig, agg_weights[['SAFRA', 'TX_MEDIA_PONDERADA']], on='SAFRA', how='left')
                        
                    if 'PRAZO_VP' in agg_weights.columns:
                        agg_weights['PRAZO_MEDIO_PONDERADO'] = 0.0
                        # Divide por 21 para converter de Dias Úteis para Meses
                        agg_weights.loc[mask, 'PRAZO_MEDIO_PONDERADO'] = (agg_weights.loc[mask, 'PRAZO_VP'] / agg_weights.loc[mask, 'VALOR_PRINCIPAL_JUROS']) / 21.0
                        agg_orig = pd.merge(agg_orig, agg_weights[['SAFRA', 'PRAZO_MEDIO_PONDERADO']], on='SAFRA', how='left')
                else:
                    st.info("💡 AVISO: Não foi possível calcular Taxa e Prazo Ponderado por contrato pois faltam colunas de ID do Contrato ou 'VALOR_PRINCIPAL_JUROS'.")

                fig_orig = go.Figure()
                
                fig_orig.add_trace(go.Bar(
                    x=agg_orig['SAFRA'], y=agg_orig['Soma_Valor_Parcela'],
                    name='Soma Valor da Parcela', marker_color='#1f77b4', yaxis='y1'
                ))
                
                fig_orig.add_trace(go.Bar(
                    x=agg_orig['SAFRA'], y=agg_orig['Soma_VP_Calculado'],
                    name='Soma Valor Principal (Originação)', marker_color='#2ca02c', yaxis='y1'
                ))
                
                # Adiciona a linha de taxa se a coluna existir (Eixo Secundário y2)
                if 'TX_MEDIA_PONDERADA' in agg_orig.columns:
                    fig_orig.add_trace(go.Scatter(
                        x=agg_orig['SAFRA'], y=agg_orig['TX_MEDIA_PONDERADA'],
                        name='Taxa Média Ponderada', mode='lines+markers', marker_color='red', yaxis='y2'
                    ))

                # Adiciona a linha de Prazo Médio se a coluna existir (Terceiro Eixo y3)
                if 'PRAZO_MEDIO_PONDERADO' in agg_orig.columns:
                    fig_orig.add_trace(go.Scatter(
                        x=agg_orig['SAFRA'], y=agg_orig['PRAZO_MEDIO_PONDERADO'],
                        name='Prazo Médio Ponderado (Meses)', mode='lines+markers', marker_color='purple', yaxis='y3'
                    ))

                # Configuração avançada de layout para comportar 3 eixos verticais
                fig_orig.update_layout(
                    title='Comparativo de Originação, Taxa e Prazo Médio Ponderado por Safra',
                    xaxis=dict(domain=[0, 0.85]), # Reduz a área do x para caber o 3º eixo sem sobrepor
                    yaxis=dict(title='Volume (R$)'),
                    yaxis2=dict(title='Taxa Média Ponderada (%)', overlaying='y', side='right'),
                    yaxis3=dict(title='Prazo Médio Ponderado (Meses)', overlaying='y', side='right', position=1, anchor='free', showgrid=False),
                    barmode='group', hovermode="x unified"
                )
                st.plotly_chart(fig_orig, use_container_width=True)

                # Configuração da tabela para exibição
                tabela_cols_rename = {
                    'Soma_Valor_Parcela': 'Soma Valor da Parcela',
                    'Soma_VP_Calculado': 'Soma Valor Principal (Originação)'
                }
                format_dict = {
                    'Soma Valor da Parcela': 'R$ {:,.2f}',
                    'Soma Valor Principal (Originação)': 'R$ {:,.2f}'
                }
                
                if 'TX_MEDIA_PONDERADA' in agg_orig.columns:
                    tabela_cols_rename['TX_MEDIA_PONDERADA'] = 'Taxa Média Ponderada (%)'
                    format_dict['Taxa Média Ponderada (%)'] = '{:.4f}%'
                    
                if 'PRAZO_MEDIO_PONDERADO' in agg_orig.columns:
                    tabela_cols_rename['PRAZO_MEDIO_PONDERADO'] = 'Prazo Médio Ponderado (Meses)'
                    format_dict['Prazo Médio Ponderado (Meses)'] = '{:.1f}'
                
                tabela_orig = agg_orig.rename(columns=tabela_cols_rename)
                
                # Remove as colunas temporárias de multiplicação do visor
                tabela_orig = tabela_orig.drop(columns=['TX_VP', 'PRAZO_VP'], errors='ignore')

                st.dataframe(tabela_orig.style.format(format_dict), use_container_width=True)
            else:
                st.warning("Colunas necessárias ('VALOR_DA_PARCELA', 'Valor_Principal_Originacao' ou 'SAFRA') ausentes.")
        
        # ------------------------------------------
        # TAB TIR (TAXA INTERNA DE RETORNO)
        # ------------------------------------------
        with tab_tir:
            st.header("Análise de TIR Anualizada (XIRR) por Safra")
            st.markdown("""
            **Lógica do Cálculo Atualizada (Precisão Diária):**
            * **D0 a Dn (Originação):** As saídas de caixa ocorrem nos **dias exatos da `DATA_AVERBACAO`** dos contratos, garantindo que o tempo real do dinheiro "na rua" não seja encurtado artificialmente.
            * **Fluxos Intermediários:** Pagamentos de `VALOR_DA_PARCELA` computados em seus dias exatos até a Data de Referência.
            * **Último Offset (Terminal):** Valor remanescente projetado na Data de Referência. Parcelas já vencidas entram pelo valor de face, e parcelas a vencer são trazidas a **Valor Presente** (descapitalizadas até a foto), abatendo-se o **% de Haircut**.
            """)
            
            if 'SAFRA' in df_filtered.columns and 'Valor_Principal_Originacao' in df_filtered.columns and 'DATA_AVERBACAO' in df_filtered.columns:
                # 1. Obter Vintages 
                safras_presentes = sorted([s for s in df_filtered['SAFRA'].unique() if pd.notnull(s)])
                safras_str_list = [s.strftime('%Y-%m') for s in safras_presentes]
                
                if not safras_str_list:
                    st.info("Não há safras disponíveis na seleção de filtros atual.")
                else:
                    st.subheader("Configuração de Haircut por Safra (%)")
                    st.write("Ajuste o percentual de perda (Haircut) a ser aplicado no saldo não pago na Data de Referência.")
                    hc_df = pd.DataFrame({
                        'Safra': safras_str_list,
                        'Haircut (%)': [50.0] * len(safras_str_list) # Default 50%
                    })
                    
                    # CHAVE DINÂMICA: Previne que o Streamlit trave (StreamlitAPIException) quando o filtro muda a quantidade de safras
                    editor_key = "hc_editor_" + "_".join(safras_str_list)
                    
                    # Permite ao usuário editar diretamente a tabela
                    hc_df_edited = st.data_editor(hc_df, hide_index=True, use_container_width=True, key=editor_key)
                    hc_dict = dict(zip(hc_df_edited['Safra'], hc_df_edited['Haircut (%)']))
                    
                    dict_cfs = {}
                    terminal_vals = {} # Dicionário para salvar os saldos terminais
                    tir_results = {}
                    
                    for safra_date, safra_str in zip(safras_presentes, safras_str_list):
                        df_s = df_filtered[df_filtered['SAFRA'] == safra_date].copy()
                        
                        # Exclui linhas sem data de averbação para não quebrar a matemática
                        df_s = df_s.dropna(subset=['DATA_AVERBACAO'])
                        if df_s.empty: continue
                        
                        # O marco zero (D0) passa a ser a data da PRIMEIRA averbação que ocorreu nesta Safra
                        min_date = pd.to_datetime(df_s['DATA_AVERBACAO'].min())
                        
                        cf_series = {}
                        
                        # 1. Fluxos de Saída (Originação) alocados nos seus dias exatos
                        # Define qual valor saiu (fallback para o valor da parcela se o VP estiver zerado)
                        df_s['Desembolso_Real'] = np.where(df_s['Valor_Principal_Originacao'] > 0, df_s['Valor_Principal_Originacao'], df_s['VALOR_DA_PARCELA'])
                        orig_agg = df_s.groupby('DATA_AVERBACAO')['Desembolso_Real'].sum().reset_index()
                        
                        for _, row in orig_agg.iterrows():
                            offset_orig = (pd.to_datetime(row['DATA_AVERBACAO']) - min_date).days
                            cf_series[offset_orig] = cf_series.get(offset_orig, 0) - row['Desembolso_Real']
                        
                        # 2. Fluxos Pagos até a Data de Referência
                        pagos = df_s[df_s['DATA_PAGAMENTO'].notna() & (df_s['DATA_PAGAMENTO'] <= data_referencia)]
                        if not pagos.empty:
                            pagos_agg = pagos.groupby('DATA_PAGAMENTO')['VALOR_DA_PARCELA'].sum().reset_index()
                            for _, row in pagos_agg.iterrows():
                                offset_pag = (pd.to_datetime(row['DATA_PAGAMENTO']) - min_date).days
                                # Para evitar que pagamentos no mesmo dia exato da 1ª averbação zerassem a saída de caixa
                                # forçamos o D0 de pagamentos a cair no Offset 1 no mínimo (matematicamente mais seguro)
                                if offset_pag <= 0: offset_pag = 1
                                cf_series[offset_pag] = cf_series.get(offset_pag, 0) + row['VALOR_DA_PARCELA']
                                
                        # 3. Fluxo Terminal (A Receber / Não Pago) com Haircut
                        nao_pagos = df_s[(df_s['DATA_PAGAMENTO'].isna()) | (df_s['DATA_PAGAMENTO'] > data_referencia)]
                        terminal_offset = (pd.to_datetime(data_referencia) - min_date).days
                        
                        # Proteção extrema: se a data de referência for antes do primeiro dia da safra
                        if terminal_offset <= 0: terminal_offset = 1 
                        
                        # Puxa o haircut específico desta safra que foi originada e abate
                        haircut_pct = hc_dict.get(safra_str, 50.0) / 100.0
                        
                        # Soma a nova coluna Valor_Presente_Calculado (vencidos em valor de face, a vencer já em VP)
                        terminal_val = nao_pagos['Valor_Presente_Calculado'].sum() * (1.0 - haircut_pct)
                        
                        # Aloca o valor terminal no último dia para cálculo matemático
                        cf_series[terminal_offset] = cf_series.get(terminal_offset, 0) + terminal_val
                        
                        # Salva para mostrar no rodapé da tabela visual
                        terminal_vals[safra_str] = terminal_val
                        
                        # Salva no dicionário global de safras
                        dict_cfs[safra_str] = cf_series
                    
                    # Consolida as informações no DataFrame de Matriz Offset vs Safra
                    df_tir = pd.DataFrame(dict_cfs).fillna(0)
                    df_tir.index.name = 'Offset (Dias)'
                    df_tir = df_tir.sort_index()
                    
                    # Garante que todos os dias entre o Min e Max Offset existam no DataFrame para não quebrar a matemática
                    if not df_tir.empty:
                        min_idx, max_idx = int(df_tir.index.min()), int(df_tir.index.max())
                        full_idx = range(min_idx, max_idx + 1)
                        df_tir = df_tir.reindex(full_idx, fill_value=0)
                    
                    # Calcula a TIR Matemática
                    for safra_str in safras_str_list:
                        if safra_str in df_tir.columns:
                            cfs = df_tir[safra_str].values
                            days = df_tir.index.values
                            tir = calc_xirr(cfs, days)
                            tir_results[safra_str] = (tir * 100) if tir is not None else 0.0
                    
                    # Gráfico
                    st.subheader("TIR Anualizada (%) por Safra")
                    if not tir_results:
                        st.info("Dados insuficientes para formar o gráfico de TIR.")
                    else:
                        tir_df_plot = pd.DataFrame(list(tir_results.items()), columns=['Safra', 'TIR Anualizada (%)'])
                        fig_tir = px.bar(tir_df_plot, x='Safra', y='TIR Anualizada (%)', text_auto='.2f', 
                                         title=f'TIR Anualizada (Snapshot: {data_referencia.strftime("%d/%m/%Y")})')
                        fig_tir.update_layout(yaxis_ticksuffix="%")
                        st.plotly_chart(fig_tir, use_container_width=True)
                    
                    # Tabela Auxiliar de Fluxos de Caixa (Detalhe Expansível)
                    with st.expander("Ver Matriz de Fluxo de Caixa Diário por Safra (Offset Real)"):
                        st.info("💡 **Nota Explicativa da Nova Matriz:** Para corrigir distorções que inflavam artificialmente a rentabilidade, a matriz agora distribui as saídas de caixa (`D0`, `D1`, `D2`, etc.) nos **dias exatos das averbações**. O marco zero de cada coluna é a primeira data de averbação daquela Safra específica. A última linha exibe o Valor Terminal aplicado.")
                        
                        # Criação de um DataFrame Visual que adiciona a linha de fundo
                        df_tir_vis = df_tir.copy()
                        # Converte o index numérico para string para aceitar o texto especial
                        df_tir_vis.index = df_tir_vis.index.astype(str)
                        
                        # Aloca rigorosamente o Saldo Terminal no final de cada coluna respectiva
                        df_tir_vis.loc['➔ Saldo Terminal Aplicado'] = pd.Series(terminal_vals).fillna(0)
                        
                        st.dataframe(df_tir_vis.style.format("R$ {:,.2f}"), use_container_width=True)
            else:
                st.warning("Colunas necessárias ('Valor_Principal_Originacao', 'DATA_AVERBACAO' ou 'SAFRA') ausentes para cálculo da TIR.")

        # ------------------------------------------
        # TAB 1: FPD, SPD, TDP
        # ------------------------------------------
        with tab1:
            st.header("Análise de FPD, SPD e TDP por Safra")
            
            # Novo seletor de Visão de Risco
            visao_pd = st.radio(
                "Selecione a Visão de Risco para os Indicadores:",
                options=["Parcela (Visão de Fluxo/Caixa)", "Contrato (Efeito Vagão / Visão de Exposição)"],
                horizontal=True,
                help="**Parcela:** Foca no valor individual do boleto. \n\n**Contrato:** Foca no risco do crédito. Se o cliente atrasar a primeira parcela, o valor do empréstimo inteiro é contabilizado como perda de largada (FPD)."
            )
            
            if visao_pd == "Parcela (Visão de Fluxo/Caixa)":
                st.markdown("Taxa calculada por Volume Financeiro (**Valor da Parcela Inadimplente** / **Valor da Parcela Esperada**).")
            else:
                st.markdown("Taxa calculada por Exposição Total (**Valor do Contrato Inteiro Inadimplente** / **Valor do Contrato Origination Total**).")

            st.info(f"💡 **Regra:** Considera-se Inadimplente (FPD/SPD/TDP) parcelas que acumularam **mais de {dias_atraso_tolerancia} dias de atraso**, seja no pagamento realizado ou no saldo pendente atual na fotografia da data de referência. **Nota:** Apenas parcelas cujo vencimento já ocorreu até a Data de Referência entram no cálculo.")
            
            if 'NUMERO_PARCELA' in df_filtered.columns and 'SAFRA' in df_filtered.columns:
                
                # Identifica a coluna de ID do contrato para o Efeito Vagão
                id_col_pd = 'CCB_NUMEROCCB' if 'CCB_NUMEROCCB' in df_filtered.columns else ('PROPOSTA_ID' if 'PROPOSTA_ID' in df_filtered.columns else None)
                
                # Pré-calcula o valor total do contrato (Soma de todas as parcelas de cada ID)
                if id_col_pd:
                    map_valor_total_contrato_pd = df_filtered.groupby(id_col_pd)['VALOR_DA_PARCELA'].sum()
                
                def calc_pd(dataframe, parcela, visao):
                    df_parc = dataframe[(dataframe['NUMERO_PARCELA'] == parcela) & (dataframe['DATA_VENCIMENTO'] <= data_referencia)].copy()
                    
                    if visao == "Contrato (Efeito Vagão / Visão de Exposição)" and id_col_pd:
                        # Substitui o valor da parcela individual pelo valor do contrato inteiro
                        df_parc['VALOR_BASE'] = df_parc[id_col_pd].map(map_valor_total_contrato_pd)
                        df_parc['VALOR_INAD'] = np.where(df_parc['IS_DEFAULT'], df_parc['VALOR_BASE'], 0)
                    else:
                        # Mantém o fluxo normal pela parcela isolada
                        df_parc['VALOR_BASE'] = df_parc['VALOR_DA_PARCELA']
                        df_parc['VALOR_INAD'] = df_parc['VALOR_INADIMPLENTE']
                    
                    agg = df_parc.groupby('SAFRA').agg(
                        Total_Esperado=('VALOR_BASE', 'sum'),
                        Total_Inadimplente=('VALOR_INAD', 'sum')
                    ).reset_index()
                    
                    agg['Taxa (%)'] = 0.0
                    mask = agg['Total_Esperado'] > 0
                    agg.loc[mask, 'Taxa (%)'] = (agg.loc[mask, 'Total_Inadimplente'] / agg.loc[mask, 'Total_Esperado']) * 100
                    agg['Taxa (%)'] = agg['Taxa (%)'].fillna(0)
                    return agg
                
                fpd_data = calc_pd(df_filtered, 1, visao_pd)
                spd_data = calc_pd(df_filtered, 2, visao_pd)
                tdp_data = calc_pd(df_filtered, 3, visao_pd)
                
                pd_merged = pd.DataFrame({'SAFRA': fpd_data['SAFRA']})
                pd_merged = pd_merged.merge(fpd_data[['SAFRA', 'Taxa (%)']].rename(columns={'Taxa (%)': 'FPD (%)'}), on='SAFRA', how='left')
                pd_merged = pd_merged.merge(spd_data[['SAFRA', 'Taxa (%)']].rename(columns={'Taxa (%)': 'SPD (%)'}), on='SAFRA', how='left')
                pd_merged = pd_merged.merge(tdp_data[['SAFRA', 'Taxa (%)']].rename(columns={'Taxa (%)': 'TDP (%)'}), on='SAFRA', how='left')
                
                fig_pd = px.line(pd_merged, x='SAFRA', y=['FPD (%)', 'SPD (%)', 'TDP (%)'], 
                                 markers=True, title=f'Evolução de FPD, SPD e TDP por Safra - Visão {visao_pd.split(" ")[0]}',
                                 labels={'value': 'Taxa de Inadimplência (%)', 'variable': 'Indicador'})
                fig_pd.update_layout(yaxis_ticksuffix="%")
                st.plotly_chart(fig_pd, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("FPD Médio (Filtro)", f"{pd_merged['FPD (%)'].mean():.2f}%")
                col2.metric("SPD Médio (Filtro)", f"{pd_merged['SPD (%)'].mean():.2f}%")
                col3.metric("TDP Médio (Filtro)", f"{pd_merged['TDP (%)'].mean():.2f}%")
                
                # --- NOVO: Tabela de Detalhamento do FPD ---
                st.write("---")
                st.write(f"#### Detalhamento FPD ({visao_pd.split(' ')[0]})")
                
                # Renomeia colunas para a tabela de exibição
                fpd_table_display = fpd_data.rename(columns={
                    'SAFRA': 'Safra',
                    'Total_Inadimplente': 'Inadimplência (Numerador)',
                    'Total_Esperado': 'Total Esperado (Denominador)',
                    'Taxa (%)': 'FPD (%)'
                })
                
                # Exibe a tabela com a formatação financeira
                st.dataframe(fpd_table_display.style.format({
                    'Inadimplência (Numerador)': 'R$ {:,.2f}',
                    'Total Esperado (Denominador)': 'R$ {:,.2f}',
                    'FPD (%)': '{:.2f}%'
                }), use_container_width=True)
                
            else:
                st.warning("Colunas 'NUMERO_PARCELA' ou 'SAFRA' não encontradas para gerar essa análise.")

        # ------------------------------------------
        # TAB 2: ANÁLISE NÃO ESCRITURADO
        # ------------------------------------------
        with tab2:
            st.header("Análise de Risco: Não Escriturado")
            st.markdown("Proporção do volume que não foi averbado/descontado (Saldo Não Escriturado).")
            st.info("💡 **Regra do Cálculo:** A razão está sendo calculada considerando apenas as linhas de **Parcela 1**. Nelas, soma-se a coluna `SALDO_NAO_ESCRIT` para compor o Numerador e soma-se a coluna `VALOR_PRINCIPAL_JUROS` para compor o Denominador, agrupando por Safra.")
            
            if 'SAFRA' in df_filtered.columns and 'SALDO_NAO_ESCRIT' in df_filtered.columns and 'VALOR_PRINCIPAL_JUROS' in df_filtered.columns and 'NUMERO_PARCELA' in df_filtered.columns:
                
                # Filtro estrito para considerar apenas as parcelas 1
                df_parcela_1 = df_filtered[df_filtered['NUMERO_PARCELA'] == 1]
                
                # Cálculo do Numerador
                num_escrit = df_parcela_1.groupby('SAFRA')['SALDO_NAO_ESCRIT'].sum().reset_index()
                num_escrit.rename(columns={'SALDO_NAO_ESCRIT': 'Numerador_Nao_Escriturado'}, inplace=True)
                
                # Cálculo do Denominador
                den_escrit = df_parcela_1.groupby('SAFRA')['VALOR_PRINCIPAL_JUROS'].sum().reset_index()
                den_escrit.rename(columns={'VALOR_PRINCIPAL_JUROS': 'Denominador_Valor_Principal_Juros'}, inplace=True)
                
                agg_escrit = pd.merge(num_escrit, den_escrit, on='SAFRA', how='left')
                agg_escrit['Numerador_Nao_Escriturado'] = agg_escrit['Numerador_Nao_Escriturado'].fillna(0)
                agg_escrit['Denominador_Valor_Principal_Juros'] = agg_escrit['Denominador_Valor_Principal_Juros'].fillna(0)
                
                agg_escrit['% Não Escriturado'] = 0.0
                mask = agg_escrit['Denominador_Valor_Principal_Juros'] > 0
                agg_escrit.loc[mask, '% Não Escriturado'] = (agg_escrit.loc[mask, 'Numerador_Nao_Escriturado'] / agg_escrit.loc[mask, 'Denominador_Valor_Principal_Juros']) * 100
                
                fig_escrit = go.Figure()
                fig_escrit.add_trace(go.Bar(
                    x=agg_escrit['SAFRA'], y=agg_escrit['Numerador_Nao_Escriturado'],
                    name='Soma Não Escriturado (R$)', marker_color='indianred', yaxis='y1'
                ))
                fig_escrit.add_trace(go.Scatter(
                    x=agg_escrit['SAFRA'], y=agg_escrit['% Não Escriturado'],
                    name='% Não Escriturado', mode='lines+markers', marker_color='black', yaxis='y2'
                ))
                
                fig_escrit.update_layout(
                    title='Soma Não Escriturado vs % Sobre o Originado (VPJ) por Safra',
                    yaxis=dict(title='Volume Não Escriturado (R$)'),
                    yaxis2=dict(title='Taxa (%)', overlaying='y', side='right', ticksuffix='%'),
                    barmode='group', hovermode="x unified"
                )
                st.plotly_chart(fig_escrit, use_container_width=True)
                
                tabela_exibicao = agg_escrit.rename(columns={
                    'Numerador_Nao_Escriturado': 'Saldo Não Escriturado (Numerador)',
                    'Denominador_Valor_Principal_Juros': 'Valor Principal Juros (Denominador)'
                })
                st.dataframe(tabela_exibicao.style.format({
                    'Saldo Não Escriturado (Numerador)': 'R$ {:,.2f}', 
                    'Valor Principal Juros (Denominador)': 'R$ {:,.2f}', 
                    '% Não Escriturado': '{:.2f}%'
                }), use_container_width=True)
            else:
                st.warning("Colunas 'SALDO_NAO_ESCRIT', 'VALOR_PRINCIPAL_JUROS' ou 'NUMERO_PARCELA' ausentes.")

        # ------------------------------------------
        # TAB 3: FPD vs PERDA OVER X & NÃO ESCRITURADO
        # ------------------------------------------
        with tab3:
            st.header(f"Correlação Global: FPD vs Perda (Over {dias_over}) vs Não Escriturado")
            st.markdown(f"Visão unificada de risco da Safra. Acompanhe se o calote inicial (FPD) é reflexo de problemas operacionais (Não Escriturado) ou se é um risco de crédito real que se consolida na Perda (Over {dias_over}).")
            st.info(f"💡 **Regra Over {dias_over} (Efeito Vagão):** Se qualquer parcela de um contrato atingir mais de {dias_over} dias de atraso na fotografia da Data de Referência, o contrato inteiro é considerado inadimplente. **Nota:** Para não diluir a taxa, consideramos no denominador apenas as parcelas cuja data de vencimento ocorreu há pelo menos {dias_over} dias da Data de Referência.")
            
            if 'SAFRA' in df_filtered.columns:
                # --- CÁLCULO DA TABELA 1: COM FILTRO DE MATURIDADE ---
                df_over_eligible = df_filtered[(data_referencia - df_filtered['DATA_VENCIMENTO']).dt.days >= dias_over]
                
                over_data = df_over_eligible.groupby('SAFRA').agg(
                    Total_Volume=('VALOR_DA_PARCELA', 'sum'),
                    Volume_Over=('VALOR_OVER', 'sum')
                ).reset_index()
                over_data[f'Over {dias_over} (%)'] = (over_data['Volume_Over'] / over_data['Total_Volume']) * 100
                over_data[f'Over {dias_over} (%)'] = over_data[f'Over {dias_over} (%)'].fillna(0)
                
                comp_over = fpd_data[['SAFRA', 'Taxa (%)']].rename(columns={'Taxa (%)': 'FPD (%)'})
                comp_over = comp_over.merge(over_data[['SAFRA', f'Over {dias_over} (%)']], on='SAFRA', how='left')
                
                # --- CÁLCULO DA TABELA 2: SEM FILTRO DE MATURIDADE (DILUÍDO) ---
                over_data_full = df_filtered.groupby('SAFRA').agg(
                    Total_Volume=('VALOR_DA_PARCELA', 'sum'),
                    Volume_Over=('VALOR_OVER', 'sum')
                ).reset_index()
                over_data_full[f'Over {dias_over} (Diluído) (%)'] = (over_data_full['Volume_Over'] / over_data_full['Total_Volume']) * 100
                over_data_full[f'Over {dias_over} (Diluído) (%)'] = over_data_full[f'Over {dias_over} (Diluído) (%)'].fillna(0)
                
                comp_over_full = fpd_data[['SAFRA', 'Taxa (%)']].rename(columns={'Taxa (%)': 'FPD (%)'})
                comp_over_full = comp_over_full.merge(over_data_full[['SAFRA', f'Over {dias_over} (Diluído) (%)']], on='SAFRA', how='left')

                # Junta os dados para o gráfico consolidado
                grafico_dados = comp_over.merge(comp_over_full[['SAFRA', f'Over {dias_over} (Diluído) (%)']], on='SAFRA', how='left')
                
                # Incorpora a métrica de Não Escriturado se ela existir na execução da aba 2
                has_escriturado = 'agg_escrit' in locals() and '% Não Escriturado' in agg_escrit.columns
                if has_escriturado:
                    grafico_dados = grafico_dados.merge(agg_escrit[['SAFRA', '% Não Escriturado']], on='SAFRA', how='left')
                    comp_over = comp_over.merge(agg_escrit[['SAFRA', '% Não Escriturado']], on='SAFRA', how='left')
                    comp_over_full = comp_over_full.merge(agg_escrit[['SAFRA', '% Não Escriturado']], on='SAFRA', how='left')

                fig_comp2 = go.Figure()
                fig_comp2.add_trace(go.Scatter(
                    x=grafico_dados['SAFRA'], y=grafico_dados['FPD (%)'],
                    name='FPD (%)', mode='lines+markers', line=dict(color='blue', width=3)
                ))
                fig_comp2.add_trace(go.Scatter(
                    x=grafico_dados['SAFRA'], y=grafico_dados[f'Over {dias_over} (%)'],
                    name=f'Perda Over {dias_over} (Maturada) (%)', mode='lines+markers', line=dict(color='red', width=3, dash='dot')
                ))
                fig_comp2.add_trace(go.Scatter(
                    x=grafico_dados['SAFRA'], y=grafico_dados[f'Over {dias_over} (Diluído) (%)'],
                    name=f'Perda Over {dias_over} (Diluída) (%)', mode='lines+markers', line=dict(color='orange', width=2, dash='dash')
                ))
                
                if has_escriturado:
                    fig_comp2.add_trace(go.Scatter(
                        x=grafico_dados['SAFRA'], y=grafico_dados['% Não Escriturado'],
                        name='% Não Escriturado', mode='lines+markers', line=dict(color='green', width=2, dash='dashdot')
                    ))
                
                fig_comp2.update_layout(
                    title=f'Acompanhamento de FPD vs Perda Efetiva (> {dias_over} Dias) vs Não Escriturado (Ref: {data_referencia.strftime("%d/%m/%Y")})',
                    yaxis_title='Taxa (%)', yaxis_ticksuffix="%",
                    hovermode="x unified"
                )
                st.plotly_chart(fig_comp2, use_container_width=True)
                
                format_dict = {'FPD (%)': '{:.2f}%', f'Over {dias_over} (%)': '{:.2f}%'}
                format_dict_full = {'FPD (%)': '{:.2f}%', f'Over {dias_over} (Diluído) (%)': '{:.2f}%'}
                if has_escriturado:
                    format_dict['% Não Escriturado'] = '{:.2f}%'
                    format_dict_full['% Não Escriturado'] = '{:.2f}%'

                st.write("#### Tabela 1: Visão com Filtro de Maturidade (Recomendado)")
                st.markdown(f"Considera no denominador **apenas** parcelas cujo vencimento ocorreu há pelo menos {dias_over} dias (evita diluição).")
                st.dataframe(comp_over.style.format(format_dict), use_container_width=True)

                st.write("---")
                st.write("#### Tabela 2: Visão sem Filtro de Maturidade (Safra Inteira)")
                st.markdown(f"Considera no denominador **todas as parcelas** originadas na safra, mesmo as que venceram há pouco tempo ou ainda vão vencer (causa diluição na taxa de perda).")
                st.dataframe(comp_over_full.style.format(format_dict_full), use_container_width=True)
            else:
                st.warning("Dados insuficientes para este comparativo.")
                
        # ------------------------------------------
        # TAB COLL: CURVAS DE COLLECTION
        # ------------------------------------------
        with tab_coll:
            st.header("Curvas de Collection por Safra")
            st.markdown("Acompanhamento da eficiência de arrecadação da carteira (Taxa de Collection) ao longo do tempo (Meses desde a Originação).")
            st.info("💡 **Regra do Cálculo:** Para cada mês de vida da safra (*Month on Book* - MOB), o **Numerador** é a soma de tudo que foi pago até aquele mês. Para evitar distorções (>100%) devido a **pré-pagamentos**, o **Denominador** soma todas as parcelas que já venceram até o mês atual **MAIS** as parcelas futuras que o cliente optou por pré-pagar (tornando-as exigíveis antecipadamente).")
            
            if 'SAFRA' in df_filtered.columns and 'VALOR_DA_PARCELA' in df_filtered.columns and 'DATA_VENCIMENTO' in df_filtered.columns and 'DATA_PAGAMENTO' in df_filtered.columns:
                df_c = df_filtered.copy()
                
                # Converte SAFRA para datetime para facilitar contas matemáticas
                df_c['SAFRA_DT'] = pd.to_datetime(df_c['SAFRA'])
                
                # Calcula o Offset (MOB - Month on Book) em meses para o Vencimento e para o Pagamento
                df_c['MOB_VENC'] = (df_c['DATA_VENCIMENTO'].dt.year - df_c['SAFRA_DT'].dt.year) * 12 + (df_c['DATA_VENCIMENTO'].dt.month - df_c['SAFRA_DT'].dt.month)
                df_c['MOB_PAG'] = (df_c['DATA_PAGAMENTO'].dt.year - df_c['SAFRA_DT'].dt.year) * 12 + (df_c['DATA_PAGAMENTO'].dt.month - df_c['SAFRA_DT'].dt.month)
                
                safras_coll = sorted([s for s in df_c['SAFRA'].unique() if pd.notnull(s)])
                
                records = []
                for safra in safras_coll:
                    safra_str = safra.strftime('%Y-%m')
                    df_s = df_c[df_c['SAFRA'] == safra]
                    
                    safra_dt_val = pd.to_datetime(safra)
                    # Snapshot limite: Não vamos plotar collections para meses que ainda nem chegaram na nossa 'Foto'
                    mob_snapshot = (data_referencia.year - safra_dt_val.year) * 12 + (data_referencia.month - safra_dt_val.month)
                    
                    max_mob_venc = int(df_s['MOB_VENC'].max()) if pd.notna(df_s['MOB_VENC'].max()) else 0
                    
                    # Vamos calcular a curva até o final da vida das parcelas da safra OU até a foto atual (o que for menor)
                    limite_offset = min(mob_snapshot, max_mob_venc)
                    
                    if limite_offset < 0:
                        continue
                        
                    for m in range(0, limite_offset + 1):
                        
                        # --- AJUSTE INTELIGENTE PARA PRÉ-PAGAMENTOS ---
                        # Condição 1: A parcela vencia até o offset atual (m) - Segue o fluxo normal.
                        mask_venc_ate_m = df_s['MOB_VENC'] <= m
                        
                        # Condição 2 (O Truque): A parcela vencia APÓS o offset atual (m), MAS o cliente pré-pagou ela até o offset atual.
                        # Como esse dinheiro vai entrar no Numerador, a parcela PRECISA entrar no Denominador do mês para não estourar os 100%.
                        mask_pre_pago = (df_s['MOB_VENC'] > m) & (df_s['DATA_PAGAMENTO'].notna()) & (df_s['DATA_PAGAMENTO'] <= data_referencia) & (df_s['MOB_PAG'] <= m)
                        
                        # Denominador Dinâmico: Vencido + Pré-Pagos
                        den = df_s[mask_venc_ate_m | mask_pre_pago]['VALOR_DA_PARCELA'].sum()
                        
                        # Numerador: Soma do Valor_Parcela pago ATÉ o offset atual (m) E que obedeça a data de corte do painel
                        num = df_s[
                            (df_s['DATA_PAGAMENTO'].notna()) & 
                            (df_s['DATA_PAGAMENTO'] <= data_referencia) & 
                            (df_s['MOB_PAG'] <= m)
                        ]['VALOR_DA_PARCELA'].sum()
                        
                        if den > 0:
                            coll_pct = (num / den) * 100
                            records.append({
                                'SAFRA': safra_str,
                                'Offset (Meses)': m,
                                'Collection (%)': coll_pct,
                                'Total Pago': num,
                                'Total Exigível Ajustado': den
                            })
                
                if records:
                    df_coll_plot = pd.DataFrame(records)
                    
                    fig_coll = px.line(df_coll_plot, x='Offset (Meses)', y='Collection (%)', color='SAFRA', markers=True,
                                      title=f'Curvas de Collection (Acumulado) por Safra (Ref: {data_referencia.strftime("%d/%m/%Y")})')
                    
                    fig_coll.update_layout(yaxis_ticksuffix="%", hovermode="x unified")
                    st.plotly_chart(fig_coll, use_container_width=True)
                    
                    # Prepara a tabela no estilo Cohort/Vintage (Linhas = Safras, Colunas = Offset)
                    df_pivot = df_coll_plot.pivot(index='SAFRA', columns='Offset (Meses)', values='Collection (%)')
                    
                    st.write("#### Tabela de Evolução: Collection (%) vs Month on Book (MOB)")
                    st.dataframe(df_pivot.style.format("{:.2f}%", na_rep="-"), use_container_width=True)
                else:
                    st.info("Não há dados válidos de pagamento e vencimento para gerar as curvas na data de referência selecionada.")
            else:
                st.warning("Colunas necessárias ('VALOR_DA_PARCELA', 'DATA_VENCIMENTO', 'DATA_PAGAMENTO' ou 'SAFRA') ausentes.")

        # ------------------------------------------
        # TAB PAR: ANÁLISE DE PORTFOLIO AT RISK (VINTAGE)
        # ------------------------------------------
        with tab_par:
            st.header(f"Curvas de Portfolio at Risk (PAR > {dias_over} Dias)")
            st.markdown("O PAR (Portfolio at Risk) responde à pergunta: ***De todo o valor que eu emprestei numa Safra, qual é a proporção do Saldo Devedor que está comprometida por contratos em atraso?***")
            st.info(f"💡 **Regra do Cálculo (Point-in-Time):** Avaliamos o final de cada mês (Month on Book). Se naquela foto específica o contrato tinha ao menos uma parcela com mais de **{dias_over} dias de atraso** (ainda não paga), o **Efeito Vagão** é acionado e **todo o saldo devedor restante daquele contrato** é contabilizado como 'Em Risco'.")
            
            with st.spinner('Construindo matrizes mensais de vintage... Isso pode levar alguns segundos dependendo do tamanho da base.'):
                df_par_vintage = calcular_vintage_par_otimizado(df_filtered, data_referencia, dias_over)
                
            if not df_par_vintage.empty:
                # Formata a safra para exibição no gráfico
                df_par_vintage['Safra'] = df_par_vintage['mes_safra'].dt.strftime('%Y-%m')
                
                # Gráfico
                fig_par = px.line(df_par_vintage, x='MOB', y='PAR (%)', color='Safra', markers=True,
                                  title=f'Evolução do PAR > {dias_over} Dias por Safra (MOB)')
                
                fig_par.update_layout(
                    xaxis_title='Month on Book (MOB)',
                    yaxis_title='PAR (%)',
                    yaxis_ticksuffix="%", 
                    hovermode="x unified"
                )
                st.plotly_chart(fig_par, use_container_width=True)
                
                # Tabela Auxiliar (Pivot)
                st.write("#### Tabela de Evolução: PAR (%) vs Month on Book (MOB)")
                df_pivot_par = df_par_vintage.pivot(index='Safra', columns='MOB', values='PAR (%)')
                st.dataframe(df_pivot_par.style.format("{:.2f}%", na_rep="-"), use_container_width=True)
            else:
                st.warning("Não foi possível gerar a análise de PAR. Certifique-se de ter os IDs de contrato, Valores de Parcela e Datas preenchidos na base.")

    else:
        st.error("Não foi possível carregar os dados corretamente. Verifique se o arquivo CSV está nos padrões.")
else:
    st.info("👆 Por favor, faça o upload do arquivo 'Dados.csv' no menu lateral à esquerda para iniciar as análises.")