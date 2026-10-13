
# ai_service.py
import os
import json
from google import genai
from google.genai import types
from models import TipoAnalise

class GeminiService:
    def __init__(self):
        # O SDK moderno busca a variável de ambiente GEMINI_API_KEY automaticamente
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("A variável de ambiente GEMINI_API_KEY não foi configurada no ambiente.")
        
        self.client = genai.Client()
        self.model_text = "gemini-1.5-pro"         # Raciocínio lógico e análise complexa
        self.model_embed = "text-embedding-004"    # Modelo oficial para geração de vetores (768 dimensões)

    def gerar_embedding(self, texto: str) -> list[float]:
        """
        Gera o vetor semântico de 768 dimensões para o banco de dados vetorial (pgvector).
        Toda a computação matemática ocorre na infraestrutura do Google.
        """
        resposta = self.client.models.embed_content(
            model=self.model_embed,
            contents=texto
        )
        # Retorna a lista de floats do primeiro fragmento enviado
        return resposta.embeddings.values

    def analisar_riscos_contrato(self, texto_contrato: str) -> dict:
        """
        [FASE 1: DIAGNÓSTICO COGNITIVO]
        Solicita ao Gemini uma varredura cirúrgica mapeando evidências estritas 
        para a taxonomia de 96 riscos específicos de contratações públicas brasileiras.
        """
        prompt = f"""
        Você é um auditor de elite e especialista em Direito Administrativo, Contratos Públicos 
        e na Lei nº 14.133/2021 (Nova Lei de Licitações).
        
        Sua missão é analisar o texto do contrato ou minuta fornecido e procurar EVIDÊNCIAS 
        explícitas ou implícitas relacionadas estritamente à seguinte taxonomia de riscos:

        [MATRIZ DE RISCOS REGULAMENTARES E OPERACIONAIS]
        - Especificação inadequada do objeto; Objeto genérico ou impreciso; Objeto excessivamente restritivo;
        - Ausência de Estudo Técnico Preliminar; Ausência de justificativa da contratação;
        - Pesquisa de preços insuficiente; Pesquisa de preços desatualizada; Sobrepreço; Superfaturamento;
        - Subdimensionamento da demanda; Superdimensionamento da demanda;
        - Ausência de análise de riscos; Mapa de riscos incompleto; Ausência de planejamento anual;
        - Incompatibilidade orçamentária; Falta de segregação de funções; Conflito de interesses;
        - Restrição à competitividade; Direcionamento de licitação; Exigências excessivas de habilitação;
        - Conluio entre licitantes; Fraude documental; Proposta inexequível; Proposta com preços artificialmente reduzidos;
        - Judicialização previsível do certame; Ausência de parecer jurídico obrigatório;
        - Inconsistência entre edital e contrato; Matriz de riscos ausente; Matriz de riscos incompleta;
        - Distribution inadequada de responsabilidades; Desequilíbrio econômico-financeiro não tratado;
        - Cláusulas ambíguas; Cláusulas contraditórias; Garantias contratuais insuficientes; Garantias contratuais excessivas;
        - Inexistência de indicadores de desempenho; Inexistência de acordo de nível de serviço (SLA);
        - Ausência de plano de fiscalização; Fiscalização inadequada; Gestor contratual não designado;
        - Fiscal contratual não designado; Critérios de aceite indefinidos; Cronograma físico-financeiro inadequado;
        - Atraso de execução; Paralisação contratual; Inexecução parcial; Inexecução total;
        - Subcontratação irregular; Subcontratação não autorizada; Produto fora das especificações;
        - Serviço com qualidade inferior; Falência da contratada; Recuperação judicial da contratada;
        - Insuficiência econômico-financeira da contratada; Dependência excessiva de fornecedor único;
        - Perda de conhecimento institucional; Rotatividade excessiva de fiscais; Ausência de mecanismos de continuidade;
        - Ausência de plano de contingência; Reajuste contratual inadequado; Risco inflacionário; Risco cambial;
        - Pagamento indevido; Medição incorreta; Liquidação irregular da despesa;
        - Inadimplência trabalhista da contratada; Inadimplência fiscal da contratada; Inadimplência previdenciária da contratada;
        - Risco ambiental; Ausência de licenciamento ambiental; Descumprimento de requisitos ESG;
        - Violação de direitos humanos; Trabalho infantil em cadeia de fornecimento; Trabalho análogo à escravidão;
        - Violação da LGPD; Tratamento inadequado de dados pessoais; Compartilhamento indevido de dados;
        - Ausência de cláusulas de confidencialidade; Falhas de segurança da informação; Ausência de requisitos de cibersegurança;
        - Ataques cibernéticos; Perda documental; Ausência de trilha de auditoria; Manipulação de registros;
        - Fraude contratual; Corrupção; Pagamento por serviços não executados; Recebimento irregular do objeto;
        - Descumprimento das obrigações legais; Nulidade do contrato; Nulidade da licitação;
        - Descumprimento de determinações dos órgãos de controle; Responsabilização dos agentes públicos;
        - Dano ao erário; Prejuízo à continuidade do serviço público; Descumprimento do interesse público.

        Se encontrar brechas, termos ausentes ou cláusulas perigosas correlacionadas a qualquer item acima, 
        retorne-os estruturadamente. Mantenha o tom formal, técnico e justificado juridicamente.

        Texto do Contrato/Documento para Análise:
        {texto_contrato}
        """

        # Mapeamento do Structured Output para garantir acoplamento perfeito com o Postgres JSONB
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "OBJECT",
                "properties": {
                    "score_confianca": {
                        "type": "NUMBER", 
                        "description": "Grau de certeza da IA na análise geral, variando de 0.0 a 1.0."
                    },
                    "riscos_identificados": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "nivel_risco": {
                                    "type": "STRING", 
                                    "description": "Classificação baseada na gravidade: CRITICO, ALTO, MEDIO ou BAIXO."
                                },
                                "descricao": {
                                    "type": "STRING", 
                                    "description": "O risco encontrado (usando estritamente a nomenclatura da Matriz) acompanhado da evidência ou trecho textual que o justifica."
                                },
                                "plano_mitigacao": {
                                    "type": "STRING", 
                                    "description": "Ação recomendada fundamentada na Lei 14.133/21 para mitigar o cenário."
                                }
                            },
                            "required": ["nivel_risco", "descricao", "plano_mitigacao"]
                        }
                    }
                },
                "required": ["score_confianca", "riscos_identificados"]
            }
        )

        resposta = self.client.models.generate_content(
            model=self.model_text,
            contents=prompt,
            config=config
        )
        return json.loads(resposta.text)

    def gerar_minuta_mitigada(self, texto_original: str, riscos_identificados: list) -> str:
        """
        [FASE 2: REESCRITA / CORREÇÃO ATIVA]
        Mecanismo automatizado focado em 'consertar' a minuta. Através de Engenharia de Prompt (Role-playing),
        a IA atua sob o perfil de um Procurador para neutralizar brechas perigosas no texto original.
        """
        riscos_formatados = "\n".join([f"- {r}" for r in riscos_identificados])
        
        prompt = f"""
        Você é um Procurador do Estado especialista em Direito Administrativo.
        Sua tarefa é REESCREVER e CONSERTAR a minuta de contrato fornecida abaixo.
        
        Você deve neutralizar e mitigar os seguintes riscos jurídicos identificados na versão anterior:
        {riscos_formatados}
        
        Diretrizes de Proteção ao Interesse Público:
        - Ajuste cláusulas ambíguas, abusivas ou leoninas.
        - Garanta equilíbrio econômico-financeiro e regras claras de reajuste.
        - Injete os indicadores de desempenho e penalidades legais cabíveis (Lei 14.133/21).
        
        Retorne APENAS o texto integral e corrigido do novo contrato estruturado, pronto para assinatura.
        
        Texto Original da Minuta:
        {texto_original}
        """

        resposta = self.client.models.generate_content(
            model=self.model_text,
            contents=prompt
        )
        return respuesta.text

    def analisar_ocorrencias_execucao(self, clausulas_contrato: str, relatorio_fiscalizacao: str) -> dict:
        """
        [FASE 3: APOIO À FISCALIZAÇÃO CONTÍNUA]
        Executado durante toda a vigência. Cruza os acontecimentos práticos relatados no diário 
        de campo/fiscalização com o contrato original para gerar alertas rápidos de descumprimento.
        """
        prompt = f"""
        Você é um auditor responsável pelo controle e fiscalização de contratos públicos.
        Analise o diário de obras / relatório de fiscalização fornecido e cruze os fatos 
        com as obrigações estipuladas nas cláusulas do contrato.
        
        Cláusulas Contratuais Relevantes:
        {clausulas_contrato}
        
        Relatório de Fatos/Ocorrências do Fiscal:
        {relatorio_fiscalizacao}
        
        Identifique se os fatos relatados configuram violações contratuais, atrasos injustificados, 
	subcontratações irregulares ou qualquer risco de execução listado na sua matriz de controle.
        """
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "OBJECT",
                "properties": {
                    "alertas_fiscalizacao": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "gravidade": {"type": "STRING", "description": "CRITICO, ALTO, MEDIO ou BAIXO."},
                                "fato_relatado": {"type": "STRING", "description": "O que aconteceu na vigência prático."},
                                "clausula_violada": {"type": "STRING", "description": "A regra do contrato que foi descumprida."},
                                "recomendacao_acao": {"type": "STRING", "description": "Ex: Aplicar glosa, notificar contratada, propor multa rescisória."}
                            },
                            "required": ["gravidade", "fato_relatado", "clausula_violada", "recomendacao_acao"]
                        }
                    }
                },
                "required": ["alertas_fiscalizacao"]
            }
        )

        resposta = self.client.models.generate_content(
            model=self.model_text,
            contents=prompt,
            config=config
        )
        return json.loads(resposta.text)