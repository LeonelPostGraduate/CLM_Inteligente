
# models.py
import uuid
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Dict, Any

# ==============================================================================
# ENUMS COM PROPRIEDADES DE EXIBIÇÃO (PADRÃO PARA RELATÓRIOS/INTERFACE)
# ==============================================================================

class StatusContrato(Enum):
    PLANEJAMENTO = "PLANEJAMENTO"
    EM_ELABORACAO = "EM_ELABORACAO"
    EM_APROVACAO = "EM_APROVACAO"
    AGUARDANDO_ASSINATURA = "AGUARDANDO_ASSINATURA"
    ASSINADO = "ASSINADO"
    EM_EXECUCAO = "EM_EXECUCAO"
    ALTERADO = "ALTERADO"
    SUSPENSO = "SUSPENSO"
    RESCINDIDO = "RESCINDIDO"
    RENOVADO = "RENOVADO"
    ENCERRADO = "ENCERRADO"
    VENCIDO = "VENCIDO"
    CANCELADO = "CANCELADO"

    @property
    def descricao(self) -> str:
        """Retorna o nome amigável e acentuado para exibição ao usuário final."""
        mapeamento = {
            StatusContrato.PLANEJAMENTO: "Planejamento",
            StatusContrato.EM_ELABORACAO: "Em Elaboração",
            StatusContrato.EM_APROVACAO: "Em Aprovação",
            StatusContrato.AGUARDANDO_ASSINATURA: "Aguardando Assinatura",
            StatusContrato.ASSINADO: "Assinado",
            StatusContrato.EM_EXECUCAO: "Em Execução",
            StatusContrato.ALTERADO: "Alterado",
            StatusContrato.SUSPENSO: "Suspenso",
            StatusContrato.RESCINDIDO: "Rescindido",
            StatusContrato.RENOVADO: "Renovado",
            StatusContrato.ENCERRADO: "Encerrado",
            StatusContrato.VENCIDO: "Vencido",
            StatusContrato.CANCELADO: "Cancelado"
        }
        return mapeamento[self]


class TipoAnalise(Enum):
    ANALISE_RISCO = "ANALISE_RISCO"
    EXTRACAO_METADADOS = "EXTRACAO_METADADOS"
    CONFORMIDADE_LEGAL = "CONFORMIDADE_LEGAL"
    AUDITORIA_FISCALIZACAO = "AUDITORIA_FISCALIZACAO"
    GERACAO_MINUTA = "GERACAO_MINUTA"
    CONSULTA_NORMATIVA = "CONSULTA_NORMATIVA"

    @property
    def descricao(self) -> str:
        """Retorna o nome amigável e acentuado para fins de interface ou auditoria."""
        mapeamento = {
            TipoAnalise.ANALISE_RISCO: "Análise de Risco",
            TipoAnalise.EXTRACAO_METADADOS: "Extração de Metadados",
            TipoAnalise.CONFORMIDADE_LEGAL: "Conformidade Legal",
            TipoAnalise.AUDITORIA_FISCALIZACAO: "Auditoria e Fiscalização",
            TipoAnalise.GERACAO_MINUTA: "Geração de Minuta",
            TipoAnalise.CONSULTA_NORMATIVA: "Consulta Normativa"
        }
        return mapeamento[self]


# ==============================================================================
# SEGURANÇA E CONTROLE DE ACESSO (RBAC)
# ==============================================================================

class Permissao:
    def __init__(self, codigo: str, descricao: str, id_permissao: Optional[uuid.UUID] = None):
        self.id: uuid.UUID = id_permissao or uuid.uuid4()
        self.codigo: str = codigo
        self.descricao: str = descricao


class Perfil:
    def __init__(self, nome: str, id_perfil: Optional[uuid.UUID] = None):
        self.id: uuid.UUID = id_perfil or uuid.uuid4()
        self.nome: str = nome
        self.permissoes: List[Permissao] = []


class Usuario:
    def __init__(self, nome: str, email: str, cpf: str, ativo: bool = True, id_usuario: Optional[uuid.UUID] = None):
        self.id: uuid.UUID = id_usuario or uuid.uuid4()
        self.nome: str = nome
        self.email: str = email
        self.cpf: str = cpf
        self.ativo: bool = ativo
        self.perfis: List[Perfil] = []

    def autenticar(self) -> bool:
        return self.ativo


# ==============================================================================
# FORNECEDORES
# ==============================================================================

class RepresentanteLegal:
    def __init__(self, nome: str, cpf: str, cargo: str, usuario: Optional[Usuario] = None, id_rep: Optional[uuid.UUID] = None):
        self.id: uuid.UUID = id_rep or uuid.uuid4()
        self.nome: str = nome
        self.cpf: str = cpf
        self.cargo: str = cargo
        self.usuario: Optional[Usuario] = usuario


class Fornecedor:
    def __init__(self, cnpj: str, razao_social: str, nome_fantasia: str, id_fornecedor: Optional[uuid.UUID] = None):
        self.id: uuid.UUID = id_fornecedor or uuid.uuid4()
        self.cnpj: str = cnpj
        self.razao_social: str = razao_social
        self.nome_fantasia: str = nome_fantasia
        self.representantes_legais: List[RepresentanteLegal] = []


# ==============================================================================
# WORKFLOW
# ==============================================================================

class EtapaWorkflowProcesso:
    def __init__(self, nome: str, ordem: int, status: str, id_etapa: Optional[uuid.UUID] = None):
        self.id: uuid.UUID = id_etapa or uuid.uuid4()
        self.nome: str = nome
        self.ordem: int = ordem
        self.status: str = status


class WorkflowProcesso:
    def __init__(self, nome: str, id_workflow: Optional[uuid.UUID] = None):
        self.id: uuid.UUID = id_workflow or uuid.uuid4()
        self.nome: str = nome
        self.etapas: List[EtapaWorkflowProcesso] = []


class EtapaWorkflowContrato:
    def __init__(self, nome: str, ordem: int, status: str, id_etapa: Optional[uuid.UUID] = None):
        self.id: uuid.UUID = id_etapa or uuid.uuid4()
        self.nome: str = nome
        self.ordem: int = ordem
        self.status: str = status


class WorkflowContrato:
    def __init__(self, nome: str, id_workflow: Optional[uuid.UUID] = None):
        self.id: uuid.UUID = id_workflow or uuid.uuid4()
        self.nome: str = nome
        self.etapas: List[EtapaWorkflowContrato] = []


# ==============================================================================
# DOCUMENTOS E EMBEDDINGS (RAG / PGVECTOR)
# ==============================================================================

class DocumentoEmbedding:
    def __init__(self, chunk_texto: str, vetor_embedding: List[float], numero_pagina: int, modelo_embedding: str, id_emb: Optional[uuid.UUID] = None):
        self.id: uuid.UUID = id_emb or uuid.uuid4()
        self.chunk_texto: str = chunk_texto
        self.vetor_embedding: List[float] = vetor_embedding
        self.numero_pagina: int = numero_pagina
        self.modelo_embedding: str = modelo_embedding


class AnaliseCognitiva:
    def __init__(self, tipo_analise: TipoAnalise, modelo_ia: str, versao_modelo: str, score_confianca: Decimal, resultado_json: Dict[str, Any], id_analise: Optional[uuid.UUID] = None):
        self.id: uuid.UUID = id_analise or uuid.uuid4()
        self.tipo_analise: TipoAnalise = tipo_analise
        self.modelo_ia: str = modelo_ia
        self.versao_modelo: str = versao_modelo
        self.score_confianca: Decimal = score_confianca
        self.resultado_json: Dict[str, Any] = resultado_json
        self.data_analise: datetime = datetime.now()


class Documento:
    def __init__(self, nome_arquivo: str, tipo_documento: str, versao: str, hash_arquivo: str, gcp_storage_url: str, conteudo_texto: Optional[str] = None, id_doc: Optional[uuid.UUID] = None):
        self.id: uuid.UUID = id_doc or uuid.uuid4()
        self.nome_arquivo: str = nome_arquivo
        self.tipo_documento: str = tipo_documento
        self.versao: str = versao
        self.hash_arquivo: str = hash_arquivo
        self.gcp_storage_url: str = gcp_storage_url
        self.conteudo_texto: Optional[str] = conteudo_texto
        self.data_upload: datetime = datetime.now()
        self.embeddings: List[DocumentoEmbedding] = []
        self.analises_cognitivas: List[AnaliseCognitiva] = []


class MinutaContrato(Documento):
    def __init__(self, nome_arquivo: str, tipo_documento: str, versao: str, hash_arquivo: str, gcp_storage_url: str, conteudo_texto: str, id_doc: Optional[uuid.UUID] = None):
        super().__init__(nome_arquivo, tipo_documento, versao, hash_arquivo, gcp_storage_url, conteudo_texto, id_doc)

    def elaborar_minuta(self) -> None:
        pass

    def negociar_termos(self) -> None:
        pass


class ParecerAprovacao(Documento):
    def __init__(self, nome_arquivo: str, tipo_documento: str, versao: str, hash_arquivo: str, gcp_storage_url: str, tipo: str, aprovado: bool, observacoes: str, data_emissao: date, id_doc: Optional[uuid.UUID] = None):
        super().__init__(nome_arquivo, tipo_documento, versao, hash_arquivo, gcp_storage_url, None, id_doc)
        self.tipo: str = tipo
        self.aprovado: bool = aprovado
        self.observacoes: str = observacoes
        self.data_emissao: date = data_emissao


# ==============================================================================
# PROCESSO ADMINISTRATIVO
# ==============================================================================

class SolicitacaoIntake:
    def __init__(self, descricao_objeto: str, valor_estimado: Decimal, data_abertura: date, id_intake: Optional[uuid.UUID] = None):
        self.id: uuid.UUID = id_intake or uuid.uuid4()
        self.descricao_objeto: str = descricao_objeto
        self.valor_estimado: Decimal = valor_estimado
        self.data_abertura: date = data_abertura


class ProcessoAdministrativo:
    def __init__(self, numero_processo: str, objeto: str, unidade_demandante: str, situacao: str, data_abertura: date, workflow: WorkflowProcesso, id_processo: Optional[uuid.UUID] = None):
        self.id: uuid.UUID = id_processo or uuid.uuid4()
        self.numero_processo: str = numero_processo
        self.objeto: str = objeto
        self.unidade_demandante: str = unidade_demandante
        self.situacao: str = situacao
        self.data_abertura: date = data_abertura
        self.workflow: WorkflowProcesso = workflow
        self.solicitacoes_intake: List[SolicitacaoIntake] = []
self.contratos: List['Contrato'] = []
self.documentos: List[Documento] = []

#==============================================================================
#FISCALIZAÇÃO CONTRATUAL
#==============================================================================

class FiscalContrato:
    def init(self, portaria_designacao: str, papel: str, usuario: Usuario, id_fiscal: Optional[uuid.UUID] = None):
	self.id: uuid.UUID = id_fiscal or uuid.uuid4()
	self.portaria_designacao: str = portaria_designacao
	self.papel: str = papel
	self.usuario: Usuario = usuario

class OcorrenciaExecucao:
    def init(self, data_fato: date, tipo: str, gravidade: str, descricao: str, id_ocorrencia: Optional[uuid.UUID] = None):
	self.id: uuid.UUID = id_ocorrencia or uuid.uuid4()
	self.data_fato: date = data_fato
	self.tipo: str = tipo
	self.gravidade: str = gravidade
	self.descricao: str = descricao

#==============================================================================
#COMPLIANCE, INDICADORES, ALERTAS E GESTÃO CONTRATUAL
#==============================================================================

class AnaliseRisco:
    def init(self, nivel_risco: str, descricao: str, plano_mitigacao: str, id_risco: Optional[uuid.UUID] = None):
	self.id: uuid.UUID = id_risco or uuid.uuid4()
	self.nivel_risco: str = nivel_risco
	self.descricao: str = descricao
	self.plano_mitigacao: str = plano_mitigacao

class RegraCompliance14133:
    def init(self, artigo_lei: str, descricao_requisito: str, id_regra: Optional[uuid.UUID] = None):
	self.id_regra: uuid.UUID = id_regra or uuid.uuid4()
	self.artigo_lei: str = artigo_lei
	self.descricao_requisito: str = descricao_requisito
    def validar_documento(self, documento: Documento) -> bool: 
	pass
    def validar_contrato(self, contrato: 'Contrato') -> bool: 
	pass

class AssinaturaEletronica:
    def init(self, data_assinatura: datetime, tipo_certificado: str, icp_brasil: bool, id_assinatura: Optional[uuid.UUID] = None):
	self.id: uuid.UUID = id_assinatura or uuid.uuid4()
	self.data_assinatura: datetime = data_assinatura
	self.tipo_certificado: str = tipo_certificado
	self.icp_brasil: bool = icp_brasil

class MetricaDesempenho:
    def init(self, indicador: str, valor_obtido: Decimal, avaliacao: str, id_metrica: Optional[uuid.UUID] = None):
	self.id: uuid.UUID = id_metrica or uuid.uuid4()
	self.indicador: str = indicador
	self.valor_obtido: Decimal = valor_obtido
	self.avaliacao: str = avaliacao

class Alerta:
    def init(self, tipo: str, mensagem: str, status: str, id_alerta: Optional[uuid.UUID] = None):
	self.id: uuid.UUID = id_alerta or uuid.uuid4()
	self.tipo: str = tipo
	self.mensagem: str = mensagem
	self.data_geracao: datetime = datetime.now()
	self.status: str = status

class AlteracaoContratual:
    def init(self, tipo: str, justificativa: str, valor_ajuste: Decimal, nova_vigencia: date, id_alteracao: Optional[uuid.UUID] = None):
	self.id: uuid.UUID = id_alteracao or uuid.uuid4()
	self.tipo: str = tipo
	self.justificativa: str = justificativa
	self.valor_ajuste: Decimal = valor_ajuste
	self.nova_vigencia: date = nova_vigencia

class ObrigacaoContratual:
    def init(self, descricao: str, data_vencimento: date, status: str, id_obrigacao: Optional[uuid.UUID] = None):
	self.id: uuid.UUID = id_obrigacao or uuid.uuid4()
	self.descricao: str = descricao
	self.data_vencimento: date = data_vencimento
	self.status: str = status
    def validar_cumprimento(self) -> bool:
	pass

#==============================================================================
#CONTRATO PRINCIPAL
#==============================================================================

class Contrato:
    def init(self, numero_contrato: str, objeto: str, valor_global: Decimal, vigencia_inicio: date, vigencia_fim: date, status: StatusContrato, id_contrato: Optional[uuid.UUID] = None):
	self.id: uuid.UUID = id_contrato or uuid.uuid4()
	self.numero_contrato: str = numero_contrato
	self.objeto: str = objeto
	self.valor_global: Decimal = valor_global
	self.data_assinatura: Optional[date] = None
	self.vigencia_inicio: date = vigencia_inicio
	self.vigencia_fim: date = vigencia_fim
	self.status: StatusContrato = status

# Listas de Agregação devidamente inicializadas como vazias []
	self.assinaturas: List[AssinaturaEletronica] = []
	self.obrigacoes: List[ObrigacaoContratual] = []
	self.alteracoes: List[AlteracaoContratual] = []
	self.documentos: List[Documento] = []
	self.fiscais: List[FiscalContrato] = []
	self.ocorrencias: List[OcorrenciaExecucao] = []
	self.analises_risco: List[AnaliseRisco] = []
	self.metricas: List[MetricaDesempenho] = []
	self.alertas: List[Alerta] = []
	self.analises_cognitivas: List[AnaliseCognitiva] = []
    def assinar(self) -> None:
	self.data_assinatura = date.today()
	self.status = StatusContrato.ASSINADO
	def renovar(self, nova_vigencia_fim: date) -> None:
	self.vigencia_fim = nova_vigencia_fim
	self.status = StatusContrato.RENOVADO
    def cerrar(self) -> None:
	self.status = StatusContrato.ENCERRADO

#==============================================================================
#INFRAESTRUTURA DE PROMPTS E BASE NORMATIVA (RAG)
#==============================================================================

class PromptLog:
    def init(self, prompt_enviado: str, resposta_recebida: str, tokens_input: int, tokens_output: int, modelo_utilizado: str, id_log: Optional[uuid.UUID] = None):
	self.id: uuid.UUID = id_log or uuid.uuid4()
	self.prompt_enviado: str = prompt_enviado
	self.resposta_recebida: str = resposta_recebida
	self.tokens_input: int = tokens_input
	self.tokens_output: int = tokens_output
	self.modelo_utilizado: str = modelo_utilizado
	self.data_interacao: datetime = datetime.now()

class EmbeddingNormativo:
    def init(self, chunk_texto: str, vetor_embedding: List[float], id_emb: Optional[uuid.UUID] = None):
	self.id: uuid.UUID = id_emb or uuid.uuid4()
	self.chunk_texto: str = chunk_texto
	self.vetor_embedding: List[float] = vetor_embedding

class DocumentoNormativo:
    def init(self, titulo: str, tipo: str, fonte: str, id_norma: Optional[uuid.UUID] = None):
	self.id: uuid.UUID = id_norma or uuid.uuid4()
	self.titulo: str = titulo
	self.tipo: str = tipo
	self.fonte: str = fonte
	self.embeddings: List[EmbeddingNormativo] = []