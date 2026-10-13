
# main.py
import os
from datetime import date
from decimal import Decimal
from models import Contrato, Documento, DocumentoEmbedding, AnaliseCognitiva, AnaliseRisco, StatusContrato, TipoAnalise
from ai_service import GeminiService
from database import DatabaseRepository

def executar_pipeline_clm():
    print("=================================================================")
    print("=== INICIALIZANDO PIPELINE DO CLM INTELIGENTE (CICLO TOTAL)  ===")
    print("=================================================================\n")
    
    # Configurações de infraestrutura
    os.environ["GEMINI_API_KEY"] = "SUA_CHAVE_AQUI"
    DATABASE_URL = "postgresql://postgres:senha@localhost:5432/clm_db"
    
    ai_service = GeminiService()
    db_repo = DatabaseRepository(DATABASE_URL)
    
    # --------------------------------------------------------------------------
    # FASE 1: DIAGNÓSTICO COGNITIVO (Seu fluxo original estruturado)
    # --------------------------------------------------------------------------
    print("[FASE 1] Inicializando Fase Pré-Contratual de Planejamento...")
    contrato = Contrato(
        numero_contrato="PE-042/2026",
        objeto="Prestação de serviços de computação em nuvem baseada no Google Cloud Platform para o Município.",
        valor_global=Decimal("250000.00"),
        vigencia_inicio=date(2026, 1, 1),
        vigencia_fim=date(2027, 1, 1),
        status=StatusContrato.PLANEJAMENTO
    )
    db_repo.salvar_contrato_inicial(contrato)
    print(f"-> Contrato {contrato.numero_contrato} inserido relacionalmente.")

    texto_minuta_bruta = (
        "Cláusula 5ª: O contratado assume risco integral por instabilidades na nuvem, "
        "sem direito a reajuste orçamentário. Cláusula 12ª: A fiscalização será feita "
        "ocasionalmente por amostragem visual e não haverá indicadores de performance."
    )
    
    doc = Documento(
        nome_arquivo="minuta_bruta_v1.pdf",
        tipo_documento="MinutaContrato",
        versao="1.0",
        hash_arquivo="sha256_mock_hash_xyz_12345",
        gcp_storage_url="gs://meu-bucket-clm/processos/minuta_bruta_v1.pdf",
        conteudo_texto=texto_minuta_bruta
    )
    
    print("-> Enviando fragmento ao Gemini para geração de embedding vetorial...")
    vetor = ai_service.gerar_embedding(doc.conteudo_texto)
    embedding_obj = DocumentoEmbedding(
        chunk_texto=doc.conteudo_texto,
        vetor_embedding=vetor,
        numero_pagina=1,
        modelo_embedding=ai_service.model_embed
    )
    db_repo.salvar_documento_e_embedding(contrato.id, doc, embedding_obj)
    
    print("-> Solicitando análise de riscos à Matriz Cognitiva do Gemini...")
    resultado_diagnostico = ai_service.analisar_riscos_contrato(doc.conteudo_texto)
    
    analise_cognitiva = AnaliseCognitiva(
        tipo_analise=TipoAnalise.ANALISE_RISCO,
        modelo_ia="Google Gemini",
        versao_modelo=ai_service.model_text,
        score_confianca=Decimal(str(resultado_diagnostico["score_confianca"])),
        resultado_json=resultado_diagnostico
    )
    
    riscos_list = []
    nomes_riscos_encontrados = []
    for risco in resultado_diagnostico["riscos_identificados"]:
        risco_obj = AnaliseRisco(
            nivel_risco=risco["nivel_risco"],
            descricao=risco["descricao"],
            plano_mitigacao=risco["plano_mitigacao"]
        )
        riscos_list.append(risco_obj)
        nomes_riscos_encontrados.append(risco["descricao"])

    db_repo.salvar_analise_e_riscos(contrato.id, analise_cognitiva, riscos_list)
    print(f"-> Diagnóstico finalizado. Score de Confiança: {analise_cognitiva.score_confianca}")
    print(f"-> Riscos Persistidos no Postgres: {len(riscos_list)} inconformidades identificadas.\n")

    # --------------------------------------------------------------------------
    # FASE 2: CORREÇÃO ATIVA / RETIFICAÇÃO DA MINUTA (Inclusão Essencial)
    # --------------------------------------------------------------------------
    print("[FASE 2] Ativando Engenharia de Prompt (Perfil: Procurador) para correção...")
    if riscos_list:
        texto_mitigado_consertado = ai_service.gerar_minuta_mitigada(
            texto_original=texto_minuta_bruta,
            riscos_identificados=nomes_riscos_encontrados
        )
        
        # Cria um novo documento representando a versão saneada e corrigida pela IA
        doc_corrigido = Documento(
            nome_arquivo="minuta_saneada_v2.pdf",
            tipo_documento="MinutaContratoCorrigida",
            versao="2.0",
            hash_arquivo="sha256_mock_hash_abc_98765",
            gcp_storage_url="gs://meu-bucket-clm/processos/minuta_saneada_v2.pdf",
            conteudo_texto=texto_mitigado_consertado
        )
        
        # Gera o novo embedding do texto corrigido para atualizar a base RAG
        vetor_corrigido = ai_service.gerar_embedding(doc_corrigido.conteudo_texto)
        embedding_corrigido_obj = DocumentoEmbedding(
            chunk_texto=doc_corrigido.conteudo_texto,
            vetor_embedding=vetor_corrigido,
            numero_pagina=1,
            modelo_embedding=ai_service.model_embed
        )
        db_repo.salvar_documento_e_embedding(contrato.id, doc_corrigido, embedding_corrigido_obj)
        print("-> [Sucesso] Minuta retificada e reescrita autonomamente pelo Gemini.")
        print("-> Nova versão contratual com cláusulas de mitigação salva no PostgreSQL.\n")
        
        # Simula a evolução do status do contrato pelo fluxo humano (Decision Support)
        contrato.assinar()
        print(f"-> Contrato assinado. Novo Status: {contrato.status.descricao}\n")

    # --------------------------------------------------------------------------
    # FASE 3: FISCALIZAÇÃO INTELIGENTE EM TEMPO DE VIGÊNCIA (Inclusão Essencial)
    # --------------------------------------------------------------------------
    print("[FASE 3] Contrato em Vigência. Ativando Módulo de Apoio à Fiscalização...")
    
    # Cláusula fictícia blindada que estaria no contrato após a correção da Fase 2
    clausulas_referencia = (
        "Cláusula 7ª (Do SLA): A contratada deve garantir disponibilidade da nuvem de 99.9%. "
        "Interrupções superiores a 2 horas sem plano de contingência geram multa automática de 5% sobre a medição."
    )
    
    # Simulação do input prático que o Fiscal inseriria no diário de campo do sistema
    diario_campo_fiscal = (
        "Relatório de Ocorrências - Dia 12/10/2026: A infraestrutura do data center ficou "
        "completamente fora do ar por 6 horas contínuas durante o horário de expediente. "
        "A empresa contratada não acionou servidores de backup e não apresentou justificativa técnica."
    )
    
    print("-> Enviando fatos e cláusulas ao Gemini para auditoria semântica...")
    alertas_gerados = ai_service.analisar_ocorrencias_execucao(
        clausulas_contrato=clausulas_referencia,
        relatorio_fiscalizacao=diario_campo_fiscal
    )
    
    # Monta a estrutura da análise cognitiva de acompanhamento para salvar no banco
    analise_fiscalizacao = AnaliseCognitiva(
        tipo_analise=TipoAnalise.AUDITORIA_FISCALIZACAO,
        modelo_ia="Google Gemini",
        versao_modelo=ai_service.model_text,
        score_confianca=Decimal("1.00"),
        resultado_json=alertas_gerados
    )
    
    # Salva os alertas na tabela de análises como payload JSONB nativo para o painel do fiscal
    db_repo.salvar_analise_e_riscos(contrato.id, analise_fiscalizacao, [])
    
    print("-> [Alertas de Fiscalização Gerados pela IA]:")
    for alerta in alertas_gerados["alertas_fiscalizacao"]:
        print(f"   - [{alerta['gravidade']}] Fato: {alerta['fato_relatado']}")
        print(f"     Cláusula Descumprida: {alerta['clausula_violada']}")
        print(f"     Recomendação Legal: {alerta['recomendacao_acao']}")
        
    print("\n=================================================================")
    print("=== PIPELINE INTEGRADO EXECUTADO COM SUCESSO ABSOLUTO!        ===")
    print("=================================================================")

if __name__ == "__main__":
    executar_pipeline_clm()